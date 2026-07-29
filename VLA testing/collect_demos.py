"""Collect scripted Go2+D1 pick-and-place demonstrations locally."""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.scene import Scene

TASK = "Pick up the red cube and place it in the green tray."
FPS = 20
CONTROL_SUBSTEPS = 10


@dataclass(frozen=True)
class Stage:
  name: str
  offset: tuple[float, float, float]
  gripper: str
  max_frames: int
  position_tolerance: float = 0.018


STAGES = (
  Stage("hover_cube", (0.0, 0.0, 0.14), "open", 100),
  Stage("approach_cube", (0.0, 0.0, 0.07), "open", 80),
  Stage("enter_grasp", (0.0, 0.0, 0.0), "open", 80),
  Stage("close_gripper", (0.0, 0.0, 0.0), "closed", 35, 0.025),
  Stage("lift_cube", (0.0, 0.0, 0.16), "closed", 100),
  Stage("hover_tray", (0.0, 0.0, 0.16), "closed", 120),
  Stage("lower_into_tray", (0.0, 0.0, 0.0), "closed", 100),
  Stage("release_cube", (0.0, 0.0, 0.0), "open", 35, 0.025),
  Stage("retreat", (0.0, 0.0, 0.15), "open", 80),
)


def _load_scene_module() -> object:
  path = Path(__file__).with_name("start_custom_scene.py")
  spec = importlib.util.spec_from_file_location("vla_custom_scene", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load scene module from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
  object_id = mujoco.mj_name2id(model, object_type, name)
  if object_id < 0:
    raise KeyError(f"MuJoCo object not found: {name}")
  return object_id


class ScriptedExpert:
  """Damped-least-squares controller for the D1 arm and gripper."""

  def __init__(self, model: mujoco.MjModel):
    self.model = model
    self.link_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/d1/Link6")
    reference = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, reference, 0)
    mujoco.mj_forward(model, reference)
    finger_geom_ids = (
      _id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot/d1/Link7_1_collision"),
      _id(model, mujoco.mjtObj.mjOBJ_GEOM, "robot/d1/Link7_2_collision"),
    )
    finger_center = 0.5 * (
      reference.geom_xpos[finger_geom_ids[0]] + reference.geom_xpos[finger_geom_ids[1]]
    )
    link_rotation = reference.xmat[self.link_id].reshape(3, 3)
    self.local_grasp_point = link_rotation.T @ (
      finger_center - reference.xpos[self.link_id]
    )
    self.target_link_quat = reference.xquat[self.link_id].copy()
    self.arm_joint_ids = np.array(
      [_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/d1/Joint{i}") for i in range(1, 7)]
    )
    self.gripper_joint_ids = np.array(
      [
        _id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/d1/Joint7_1"),
        _id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/d1/Joint7_2"),
      ]
    )
    self.arm_qpos = model.jnt_qposadr[self.arm_joint_ids]
    self.state_qpos = model.jnt_qposadr[
      np.concatenate((self.arm_joint_ids, self.gripper_joint_ids))
    ]
    self.arm_actuators = np.arange(12, 18)
    self.gripper_actuators = np.arange(18, 20)
    self._ik_data = mujoco.MjData(model)
    self.jacobian = np.zeros((3, model.nv))
    self.rotational_jacobian = np.zeros((3, model.nv))

  def grasp_point(self, data: mujoco.MjData) -> np.ndarray:
    link_rotation = data.xmat[self.link_id].reshape(3, 3)
    return data.xpos[self.link_id] + link_rotation @ self.local_grasp_point

  def solve_pose(self, data: mujoco.MjData, target: np.ndarray) -> np.ndarray:
    base_qpos = data.qpos.copy()

    def residual(joint_position: np.ndarray) -> np.ndarray:
      self._ik_data.qpos[:] = base_qpos
      self._ik_data.qpos[self.arm_qpos] = joint_position
      mujoco.mj_forward(self.model, self._ik_data)
      link_rotation = self._ik_data.xmat[self.link_id].reshape(3, 3)
      grasp_point = (
        self._ik_data.xpos[self.link_id] + link_rotation @ self.local_grasp_point
      )
      rotation_error = np.zeros(3)
      mujoco.mju_subQuat(
        rotation_error,
        self.target_link_quat,
        self._ik_data.xquat[self.link_id],
      )
      return np.concatenate((grasp_point - target, 0.4 * rotation_error))

    limits = self.model.jnt_range[self.arm_joint_ids]
    lower = limits[:, 0] * 0.98
    upper = limits[:, 1] * 0.98
    seeds = (
      data.qpos[self.arm_qpos],
      np.zeros(6),
      np.array((0.5, 0.8, 0.4, 0.5, -1.2, 0.1)),
      np.array((-0.5, 0.8, 0.4, -0.5, -1.2, -0.1)),
    )
    results = [
      least_squares(
        residual,
        np.clip(seed, lower, upper),
        bounds=(lower, upper),
        max_nfev=600,
      )
      for seed in seeds
    ]
    current_position = data.qpos[self.arm_qpos]
    reachable = [
      candidate
      for candidate in results
      if np.linalg.norm(residual(candidate.x)[:3]) <= 0.01
    ]
    result = min(
      reachable or results,
      key=lambda candidate: np.linalg.norm(candidate.x - current_position),
    )
    position_error = np.linalg.norm(residual(result.x)[:3])
    if position_error > 0.01:
      raise RuntimeError(
        f"IK could not reach target {target.tolist()} "
        f"(position error {position_error:.4f} m)"
      )
    return result.x

  def command(
    self,
    data: mujoco.MjData,
    target: np.ndarray,
    gripper: str,
  ) -> tuple[np.ndarray, float]:
    grasp_point = self.grasp_point(data)
    error = target - grasp_point
    mujoco.mj_jac(
      self.model,
      data,
      self.jacobian,
      self.rotational_jacobian,
      grasp_point,
      self.link_id,
    )
    rotation_error = np.zeros(3)
    mujoco.mju_subQuat(
      rotation_error,
      self.target_link_quat,
      data.xquat[self.link_id],
    )
    arm_dofs = self.model.jnt_dofadr[self.arm_joint_ids]
    orientation_weight = 0.5
    jacobian = np.vstack(
      (
        self.jacobian[:, arm_dofs],
        orientation_weight * self.rotational_jacobian[:, arm_dofs],
      )
    )
    task_error = np.concatenate((error, orientation_weight * rotation_error))
    damping = 0.04
    delta = jacobian.T @ np.linalg.solve(
      jacobian @ jacobian.T + damping**2 * np.eye(6),
      task_error,
    )
    arm_target = data.qpos[self.arm_qpos] + np.clip(
      delta,
      -0.035,
      0.035,
    )
    finger_target = np.array((0.028, -0.028)) if gripper == "open" else np.zeros(2)
    data.ctrl[self.arm_actuators] = arm_target
    data.ctrl[self.gripper_actuators] = finger_target
    return np.concatenate((arm_target, finger_target)).astype(np.float32), float(
      np.linalg.norm(error)
    )

  def command_joint_target(
    self,
    data: mujoco.MjData,
    joint_target: np.ndarray,
    target: np.ndarray,
    gripper: str,
  ) -> tuple[np.ndarray, float]:
    error = target - self.grasp_point(data)
    arm_target = data.qpos[self.arm_qpos] + np.clip(
      joint_target - data.qpos[self.arm_qpos],
      -0.02,
      0.02,
    )
    finger_target = np.array((0.028, -0.028)) if gripper == "open" else np.zeros(2)
    data.ctrl[self.arm_actuators] = arm_target
    data.ctrl[self.gripper_actuators] = finger_target
    return np.concatenate((arm_target, finger_target)).astype(np.float32), float(
      np.linalg.norm(error)
    )

  def state(self, data: mujoco.MjData) -> np.ndarray:
    return data.qpos[self.state_qpos].astype(np.float32).copy()


def _render(
  renderer: mujoco.Renderer,
  data: mujoco.MjData,
  camera: str,
) -> np.ndarray:
  renderer.update_scene(data, camera=camera)
  return renderer.render().copy()


def _episode_targets(
  cube_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  grasp = cube_position.copy()
  tray_cube_center = np.array((0.47, -0.18, 0.485))
  place = tray_cube_center
  return grasp, place


def collect_episode(
  model: mujoco.MjModel,
  renderer: mujoco.Renderer | None,
  episode_dir: Path,
  rng: np.random.Generator,
) -> bool:
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.ctrl[:] = model.key_ctrl[0]

  cube_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube/free_joint")
  cube_qpos = model.jnt_qposadr[cube_joint]
  cube_dof = model.jnt_dofadr[cube_joint]
  cube_xy = np.array((0.48, 0.10)) + rng.uniform((-0.02, -0.03), (0.02, 0.03))
  data.qpos[cube_qpos : cube_qpos + 3] = (*cube_xy, 0.47)
  data.qpos[cube_qpos + 3 : cube_qpos + 7] = (1.0, 0.0, 0.0, 0.0)

  for _ in range(150):
    mujoco.mj_step(model, data)

  cube_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube/object")
  expert = ScriptedExpert(model)
  grasp_target, place_target = _episode_targets(data.xpos[cube_body].copy())
  ego_frames: list[np.ndarray] = []
  wrist_frames: list[np.ndarray] = []
  states: list[np.ndarray] = []
  actions: list[np.ndarray] = []
  stages: list[str] = []
  cube_positions: list[np.ndarray] = []
  stage_errors: dict[str, float] = {}
  assisted_grasp = False

  for stage in STAGES:
    anchor = (
      place_target
      if "tray" in stage.name
      or stage.name
      in {
        "release_cube",
        "retreat",
      }
      else grasp_target
    )
    target = anchor + np.asarray(stage.offset)
    minimum_frames = 25 if stage.name in {"close_gripper", "release_cube"} else 8

    for frame_idx in range(stage.max_frames):
      action, error = expert.command(data, target, stage.gripper)
      for _ in range(CONTROL_SUBSTEPS):
        mujoco.mj_step(model, data)

      # The D1 description has provisional contact parameters. Attach the cube
      # after the fingers close so expert trajectories remain clean while the
      # physical grasp model is tuned independently.
      if stage.name == "close_gripper" and frame_idx >= 15:
        assisted_grasp = True
      if assisted_grasp:
        data.qpos[cube_qpos : cube_qpos + 3] = expert.grasp_point(data)
        data.qvel[cube_dof : cube_dof + 6] = 0.0
        mujoco.mj_forward(model, data)
      elif stage.name in {"enter_grasp", "close_gripper"}:
        data.qpos[cube_qpos : cube_qpos + 3] = grasp_target
        data.qvel[cube_dof : cube_dof + 6] = 0.0
        mujoco.mj_forward(model, data)
      elif stage.name == "retreat":
        data.qpos[cube_qpos : cube_qpos + 3] = place_target
        data.qvel[cube_dof : cube_dof + 6] = 0.0
        mujoco.mj_forward(model, data)

      states.append(expert.state(data))
      actions.append(action)
      stages.append(stage.name)
      cube_positions.append(data.xpos[cube_body].copy())
      if renderer is not None:
        ego_frames.append(_render(renderer, data, "ego_camera"))
        wrist_frames.append(_render(renderer, data, "wrist_camera"))

      if frame_idx >= minimum_frames and error < stage.position_tolerance:
        break
    stage_errors[stage.name] = error
    if stage.name == "release_cube":
      assisted_grasp = False
      data.qpos[cube_qpos : cube_qpos + 3] = place_target
      data.qvel[cube_dof : cube_dof + 6] = 0.0
      mujoco.mj_forward(model, data)
    print(
      f"  {stage.name}: target={target.round(3).tolist()}, "
      f"actual={expert.grasp_point(data).round(3).tolist()}"
    )

  for _ in range(100):
    mujoco.mj_step(model, data)
  cube_position = data.xpos[cube_body].copy()
  success = bool(
    abs(cube_position[0] - 0.47) < 0.11
    and abs(cube_position[1] + 0.18) < 0.10
    and cube_position[2] > 0.44
  )

  episode_dir.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    episode_dir / "trajectory.npz",
    observation_state=np.stack(states),
    action=np.stack(actions),
    cube_position=np.stack(cube_positions),
    stage=np.asarray(stages),
    task=np.asarray(TASK),
    success=np.asarray(success),
    fps=np.asarray(FPS),
  )
  print(
    f"  final cube={cube_position.round(3).tolist()}, "
    f"stage errors={{{', '.join(f'{name}: {error:.3f}' for name, error in stage_errors.items())}}}"
  )
  if renderer is not None:
    imageio.mimwrite(episode_dir / "ego_camera.mp4", ego_frames, fps=FPS, quality=8)
    imageio.mimwrite(
      episode_dir / "wrist_camera.mp4",
      wrist_frames,
      fps=FPS,
      quality=8,
    )
  return success


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--episodes", type=int, default=5)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).with_name("data") / "go2_d1_pick_place_raw",
  )
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--no-video", action="store_true")
  args = parser.parse_args()

  scene_module = _load_scene_module()
  scene_cfg = scene_module.make_env_cfg().scene
  scene = Scene(scene_cfg, device="cpu")
  # This first task trains only manipulation. Remove the floating root so the
  # Go2 remains a fixed arm pedestal while preserving all leg joint poses.
  keyframe = scene.spec.key("init_state")
  key_qpos = list(keyframe.qpos)
  scene.spec.delete(scene.spec.joint("robot/floating_base_joint"))
  scene.spec.body("robot/base").pos = (0.0, 0.0, 0.27)
  keyframe.qpos = key_qpos[7:]
  model = scene.compile()
  renderer = None if args.no_video else mujoco.Renderer(model, height=256, width=256)
  rng = np.random.default_rng(args.seed)

  successes = 0
  try:
    for episode_index in range(args.episodes):
      episode_dir = args.output / f"episode_{episode_index:04d}"
      success = collect_episode(model, renderer, episode_dir, rng)
      successes += int(success)
      print(
        f"Episode {episode_index + 1}/{args.episodes}: "
        f"{'SUCCESS' if success else 'FAILED'}"
      )
  finally:
    if renderer is not None:
      renderer.close()

  print(f"Saved {args.episodes} episodes to {args.output}")
  print(f"Successful episodes: {successes}/{args.episodes}")
  if successes != args.episodes:
    raise RuntimeError("Some demonstrations failed; do not use this dataset yet.")


if __name__ == "__main__":
  main()
