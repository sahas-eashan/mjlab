"""Collect human-operated Go2+D1 demonstrations with the keyboard."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv

FPS = 20
ARM_SLICE = slice(12, 20)
TASKS = {
  1: "Pick up the red cube and place it in the green tray.",
  2: "Pick up the yellow block and place it in the green tray.",
  3: "Pick up the blue cylinder and place it in the green tray.",
}


def _load_module(filename: str, module_name: str) -> object:
  path = Path(__file__).with_name(filename)
  spec = importlib.util.spec_from_file_location(module_name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


class KeyboardTeleop:
  """Cartesian D1 teleoperation and synchronized episode recording."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    output: Path,
    step_size: float,
    command_gain: float,
  ):
    self.env = env
    self.output = output
    self.step_size = step_size
    self.command_gain = command_gain
    self.robot = env.scene["robot"]
    self.action_term = env.action_manager.get_term("joint_position")

    collector = _load_module("collect_demos.py", "vla_collect_demos")
    self.ik_model = env.sim.mj_model
    self.ik_data = mujoco.MjData(self.ik_model)
    self.expert = collector.ScriptedExpert(self.ik_model)
    self.arm_base_id = mujoco.mj_name2id(
      self.ik_model,
      mujoco.mjtObj.mjOBJ_BODY,
      "robot/d1/base_link",
    )
    if self.arm_base_id < 0:
      raise RuntimeError("Could not find the D1 base body")

    self._lock = threading.Lock()
    self._pending_moves: list[np.ndarray] = []
    self._pending_command: str | None = None
    self._selected_task = 1
    self._gripper = "open"
    self._initialized = False
    self._recording = False
    self._target = np.zeros(3)
    self._held_target = np.zeros(8, dtype=np.float32)
    self._position_jacobian = np.zeros((3, self.ik_model.nv))
    self._unused_rotation_jacobian = np.zeros((3, self.ik_model.nv))
    self._states: list[np.ndarray] = []
    self._actions: list[np.ndarray] = []
    self._ego_frames: list[np.ndarray] = []
    self._wrist_frames: list[np.ndarray] = []
    self._finger_effort: list[np.ndarray] = []

  def reset(self) -> None:
    """Reset teleoperation state after the viewer resets the environment."""
    with self._lock:
      self._pending_moves.clear()
      self._pending_command = None
      self._initialized = False
      self._recording = False
      self._clear_buffers()
    print("Scene reset. Press C when ready to record.")

  def on_key(self, key: int) -> None:
    """Queue keyboard input without touching simulation state."""
    moves = {
      ord("I"): np.array((self.step_size, 0.0, 0.0)),
      ord("K"): np.array((-self.step_size, 0.0, 0.0)),
      ord("J"): np.array((0.0, self.step_size, 0.0)),
      ord("L"): np.array((0.0, -self.step_size, 0.0)),
      ord("U"): np.array((0.0, 0.0, self.step_size)),
      ord("O"): np.array((0.0, 0.0, -self.step_size)),
    }
    with self._lock:
      if key in moves:
        self._pending_moves.append(moves[key])
      elif key == ord("G"):
        self._pending_command = "gripper"
      elif key == ord("C"):
        self._pending_command = "record"
      elif key == ord("V"):
        self._pending_command = "save"
      elif key == ord("X"):
        self._pending_command = "discard"
      elif key in (ord("1"), ord("2"), ord("3")):
        self._selected_task = key - ord("0")
        print(f"Selected task {self._selected_task}: {TASKS[self._selected_task]}")

  def __call__(self, _observation: object) -> torch.Tensor:
    current_state = (
      self.robot.data.joint_pos[0, ARM_SLICE].detach().cpu().numpy().copy()
    )
    self._sync_ik(current_state)

    if not self._initialized:
      self._target = self.expert.grasp_point(self.ik_data).copy()
      self._held_target = current_state.astype(np.float32)
      self._initialized = True

    with self._lock:
      moves = self._pending_moves
      command = self._pending_command
      self._pending_moves = []
      self._pending_command = None

    for move in moves:
      self._target += move
    self._target = np.clip(
      self._target,
      np.array((0.20, -0.42, 0.28)),
      np.array((1.00, 0.42, 0.90)),
    )
    self._limit_reachable_target()

    if command == "gripper":
      self._gripper = "closed" if self._gripper == "open" else "open"
      print(f"Gripper: {self._gripper}")
    elif command == "record":
      self._clear_buffers()
      self._recording = True
      print(f"RECORDING: {TASKS[self._selected_task]}")
    elif command == "save":
      self._save_episode()
    elif command == "discard":
      self._recording = False
      self._clear_buffers()
      print("Demo discarded.")

    error = float(np.linalg.norm(self._target - self.expert.grasp_point(self.ik_data)))
    if moves:
      self._held_target = self._position_command(current_state)
    self._held_target[6:] = (
      np.array((0.028, -0.028), dtype=np.float32) if self._gripper == "open" else 0.0
    )
    target_state = self._clamp_target(self._held_target)

    if self._recording:
      self._record_frame(current_state, target_state)

    raw_action = torch.zeros(
      (self.env.num_envs, self.env.action_manager.total_action_dim),
      device=self.env.device,
    )
    target = torch.as_tensor(target_state, device=self.env.device).unsqueeze(0)
    offset = self.action_term.offset[:, ARM_SLICE]
    scale = self.action_term.scale[:, ARM_SLICE]
    raw_action[:, ARM_SLICE] = (target - offset) / scale

    if moves:
      print(f"Target xyz={self._target.round(3).tolist()}, error={error:.3f} m")
    return raw_action

  def _sync_ik(self, state: np.ndarray) -> None:
    mujoco.mj_resetDataKeyframe(self.ik_model, self.ik_data, 0)
    self.ik_data.qpos[self.expert.state_qpos] = state
    root_pose = self.robot.data.root_link_pose_w[0].detach().cpu().numpy()
    free_joint = mujoco.mj_name2id(
      self.ik_model,
      mujoco.mjtObj.mjOBJ_JOINT,
      "robot/floating_base_joint",
    )
    if free_joint >= 0:
      root_qpos = self.ik_model.jnt_qposadr[free_joint]
      self.ik_data.qpos[root_qpos : root_qpos + 7] = root_pose
    mujoco.mj_forward(self.ik_model, self.ik_data)

  def _clamp_target(self, target: np.ndarray) -> np.ndarray:
    limits = self.robot.data.soft_joint_pos_limits[0, ARM_SLICE].detach().cpu().numpy()
    return np.clip(target, limits[:, 0], limits[:, 1]).astype(np.float32)

  def _position_command(self, current_state: np.ndarray) -> np.ndarray:
    """Compute one position-only IK update, with no wrist orientation correction."""
    grasp_point = self.expert.grasp_point(self.ik_data)
    error = self._target - grasp_point
    if np.linalg.norm(error) > 0.08:
      self._target = grasp_point.copy()
      print("Reach limit hit; holding the current arm pose.")
      return current_state.astype(np.float32)

    mujoco.mj_jac(
      self.ik_model,
      self.ik_data,
      self._position_jacobian,
      self._unused_rotation_jacobian,
      grasp_point,
      self.expert.link_id,
    )
    arm_dofs = self.ik_model.jnt_dofadr[self.expert.arm_joint_ids]
    jacobian = self._position_jacobian[:, arm_dofs]
    damping = 0.06
    delta = jacobian.T @ np.linalg.solve(
      jacobian @ jacobian.T + damping**2 * np.eye(3),
      error,
    )
    target = current_state.astype(np.float32)
    target[:6] += np.clip(
      self.command_gain * delta,
      -0.07,
      0.07,
    ).astype(np.float32)
    return self._clamp_target(target)

  def _limit_reachable_target(self) -> None:
    """Keep the gripper target inside a conservative D1 workspace sphere."""
    arm_base = self.ik_data.xpos[self.arm_base_id]
    displacement = self._target - arm_base
    distance = float(np.linalg.norm(displacement))
    max_reach = 0.54
    if distance > max_reach:
      self._target = arm_base + displacement * (max_reach / distance)

  def _record_frame(self, state: np.ndarray, action: np.ndarray) -> None:
    ego = self.env.scene["ego_camera"].data.rgb[0].detach().cpu().numpy()
    wrist = self.env.scene["wrist_camera"].data.rgb[0].detach().cpu().numpy()
    finger_effort = (
      self.robot.data.actuator_force[0, 18:20].detach().cpu().numpy().copy()
    )
    self._states.append(state.astype(np.float32))
    self._actions.append(action.copy())
    self._ego_frames.append(ego.astype(np.uint8))
    self._wrist_frames.append(wrist.astype(np.uint8))
    self._finger_effort.append(finger_effort.astype(np.float32))

  def _save_episode(self) -> None:
    self._recording = False
    if len(self._actions) < 10:
      print("Nothing saved: record at least 10 frames.")
      return

    self.output.mkdir(parents=True, exist_ok=True)
    existing = [
      int(path.name.removeprefix("episode_"))
      for path in self.output.glob("episode_[0-9][0-9][0-9][0-9]")
      if path.is_dir()
    ]
    index = max(existing, default=-1) + 1
    episode_dir = self.output / f"episode_{index:04d}"
    episode_dir.mkdir()
    np.savez_compressed(
      episode_dir / "trajectory.npz",
      observation_state=np.stack(self._states),
      action=np.stack(self._actions),
      finger_effort=np.stack(self._finger_effort),
      task=np.asarray(TASKS[self._selected_task]),
      success=np.asarray(True),
      fps=np.asarray(FPS),
    )
    imageio.mimwrite(
      episode_dir / "ego_camera.mp4",
      self._ego_frames,
      fps=FPS,
      quality=8,
    )
    imageio.mimwrite(
      episode_dir / "wrist_camera.mp4",
      self._wrist_frames,
      fps=FPS,
      quality=8,
    )
    print(f"Saved successful demo: {episode_dir} ({len(self._actions)} frames)")
    print("Press Enter to reset the scene before the next demonstration.")
    self._clear_buffers()

  def _clear_buffers(self) -> None:
    self._states.clear()
    self._actions.clear()
    self._ego_frames.clear()
    self._wrist_frames.clear()
    self._finger_effort.clear()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).parent / "data" / "go2_d1_multitask_raw",
  )
  parser.add_argument(
    "--step-size",
    type=float,
    default=0.02,
    help="Cartesian movement per key press in metres.",
  )
  parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Initialize teleoperation, step once without a viewer, and exit.",
  )
  parser.add_argument(
    "--command-gain",
    type=float,
    default=2.0,
    help="Multiplier for each arm-joint Cartesian tracking update.",
  )
  args = parser.parse_args()

  scene = _load_module("start_custom_scene.py", "vla_custom_scene")
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = scene.make_env_cfg()
  env_cfg.decimation = 10
  for camera in env_cfg.scene.sensors:
    camera.data_types = ("rgb",)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  teleop = KeyboardTeleop(
    env,
    output=args.output,
    step_size=args.step_size,
    command_gain=args.command_gain,
  )
  observation, _ = env.reset()

  if args.smoke_test:
    env.step(teleop(observation))
    env.close()
    print("Teleoperation smoke test passed.")
    return

  print("Keyboard D1 teleoperation started.")
  print("Move: I/K = forward/back, J/L = left/right, U/O = up/down")
  print("G = open/close gripper | 1/2/3 = red/yellow/blue task")
  print("C = start recording | V = save success | X = discard")
  print("Enter = reset scene | close viewer or Ctrl+C = stop")
  scene.CameraPanelViewer(env, teleop, key_callback=teleop.on_key).run()


if __name__ == "__main__":
  main()
