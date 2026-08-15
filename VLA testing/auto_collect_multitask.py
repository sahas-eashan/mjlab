"""Automatically collect balanced, layout-randomized Go2+D1 demonstrations."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg

FPS = 20
ARM_SLICE = slice(12, 20)
TASKS = {
  "red_cube": "Pick up the red cube and place it in the green tray.",
  "yellow_block": "Pick up the yellow cube and place it in the green tray.",
  "blue_cylinder": "Pick up the blue cube and place it in the green tray.",
}
CONTACT_SENSORS = {name: f"{name}_auto_contact" for name in TASKS}
LAYOUT_SLOTS = np.array(
  (
    (0.41, -0.19),
    (0.47, -0.19),
    (0.53, -0.18),
    (0.42, -0.09),
    (0.48, -0.10),
    (0.54, -0.08),
    (0.43, -0.01),
    (0.51, -0.01),
  ),
  dtype=np.float32,
)


def load_module(filename: str, module_name: str) -> Any:
  path = Path(__file__).with_name(filename)
  spec = importlib.util.spec_from_file_location(module_name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def color_pixels(image: np.ndarray, object_name: str) -> int:
  """Count conservative object-color pixels in an RGB image."""
  rgb = image.astype(np.int16)
  red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
  if object_name == "red_cube":
    mask = (red > 150) & (green < 80) & (blue < 80)
  elif object_name == "blue_cylinder":
    mask = (blue > 150) & (red < 80) & (green < 110)
  else:
    mask = (red > 150) & (green > 110) & (blue < 90)
  return int(np.count_nonzero(mask))


class AutomaticDemo:
  """Position-only expert with contact-triggered visual grasp retention."""

  def __init__(self, env: ManagerBasedRlEnv, object_name: str):
    self.env = env
    self.object_name = object_name
    self.robot = env.scene["robot"]
    self.object = env.scene[object_name]
    self.action_term = env.action_manager.get_term("joint_position")
    collector = load_module("collect_demos.py", "vla_auto_collect_controller")
    self.model = env.sim.mj_model
    self.data = mujoco.MjData(self.model)
    self.expert = collector.ScriptedExpert(self.model)
    self.position_jacobian = np.zeros((3, self.model.nv))
    self.rotation_jacobian = np.zeros((3, self.model.nv))
    self.tray_geom = mujoco.mj_name2id(
      self.model,
      mujoco.mjtObj.mjOBJ_GEOM,
      "target_tray_base",
    )
    if self.tray_geom < 0:
      raise RuntimeError("Could not find target tray geometry")

    self.stage = "hover"
    self.stage_frames = 0
    self.done = False
    self.failed_reason: str | None = None
    self.assist_local_offset: np.ndarray | None = None
    self.states: list[np.ndarray] = []
    self.actions: list[np.ndarray] = []
    self.ego_frames: list[np.ndarray] = []
    self.wrist_frames: list[np.ndarray] = []
    self.finger_effort: list[np.ndarray] = []
    self.stage_labels: list[str] = []
    self.max_ego_pixels = 0
    self.max_wrist_pixels = 0

  def _sync_kinematics(self, state: np.ndarray) -> None:
    mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
    self.data.qpos[self.expert.state_qpos] = state
    root_pose = self.robot.data.root_link_pose_w[0].detach().cpu().numpy()
    free_joint = mujoco.mj_name2id(
      self.model,
      mujoco.mjtObj.mjOBJ_JOINT,
      "robot/floating_base_joint",
    )
    if free_joint >= 0:
      root_qpos = self.model.jnt_qposadr[free_joint]
      self.data.qpos[root_qpos : root_qpos + 7] = root_pose
    mujoco.mj_forward(self.model, self.data)

  def _set_stage(self, stage: str) -> None:
    print(f"    {self.stage} -> {stage}")
    self.stage = stage
    self.stage_frames = 0

  def _contacts(self) -> bool:
    sensor = self.env.scene[CONTACT_SENSORS[self.object_name]].data
    assert sensor.found is not None
    return bool(torch.any(sensor.found[0, :2] > 0))

  def _start_grasp_assist(self, grasp_xyz: np.ndarray) -> None:
    object_xyz = self.object.data.root_link_pos_w[0].detach().cpu().numpy()
    rotation = self.data.xmat[self.expert.link_id].reshape(3, 3)
    self.assist_local_offset = rotation.T @ (object_xyz - grasp_xyz)
    print(f"    contact locked on {self.object_name}")

  def _apply_grasp_assist(self, target_state: np.ndarray) -> None:
    assert self.assist_local_offset is not None
    self.data.qpos[self.expert.state_qpos] = target_state
    mujoco.mj_forward(self.model, self.data)
    grasp_xyz = self.expert.grasp_point(self.data)
    rotation = self.data.xmat[self.expert.link_id].reshape(3, 3)
    desired = grasp_xyz + rotation @ self.assist_local_offset
    current = self.object.data.root_link_pos_w[0, :3].detach().cpu().numpy()
    linear_velocity = np.clip(
      (desired - current) * FPS + np.array((0.0, 0.0, 0.5 * 9.81 / FPS)),
      -0.65,
      0.65,
    )
    velocity = torch.zeros((1, 6), device=self.env.device)
    velocity[0, :3] = torch.as_tensor(linear_velocity, device=self.env.device)
    self.object.write_root_link_velocity_to_sim(velocity)

  def _ik_target(
    self,
    state: np.ndarray,
    target_xyz: np.ndarray,
    gripper_open: bool,
  ) -> tuple[np.ndarray, float]:
    self._sync_kinematics(state)
    grasp_xyz = self.expert.grasp_point(self.data)
    error = target_xyz - grasp_xyz
    mujoco.mj_jac(
      self.model,
      self.data,
      self.position_jacobian,
      self.rotation_jacobian,
      grasp_xyz,
      self.expert.link_id,
    )
    arm_dofs = self.model.jnt_dofadr[self.expert.arm_joint_ids]
    jacobian = self.position_jacobian[:, arm_dofs]
    damping = 0.06
    delta = jacobian.T @ np.linalg.solve(
      jacobian @ jacobian.T + damping**2 * np.eye(3),
      error,
    )
    target = state.astype(np.float32).copy()
    target[:6] += np.clip(delta, -0.04, 0.04).astype(np.float32)
    target[6:] = (0.028, -0.028) if gripper_open else (0.0, 0.0)
    limits = self.robot.data.soft_joint_pos_limits[0, ARM_SLICE]
    limits_np = limits.detach().cpu().numpy()
    target = np.clip(target, limits_np[:, 0], limits_np[:, 1]).astype(np.float32)
    return target, float(np.linalg.norm(error))

  def _stage_command(
    self,
    grasp_xyz: np.ndarray,
    object_xyz: np.ndarray,
    tray_xyz: np.ndarray,
  ) -> tuple[np.ndarray, bool]:
    if self.assist_local_offset is not None and self.stage in {"hover", "descend"}:
      self._set_stage("lift")

    if self.stage == "hover":
      target = object_xyz + np.array((0.0, 0.0, 0.13))
      if np.linalg.norm(grasp_xyz - target) < 0.035 and self.stage_frames >= 4:
        self._set_stage("descend")
      return target, True
    if self.stage == "descend":
      target = object_xyz + np.array((0.0, 0.0, 0.015))
      return target, bool(np.linalg.norm(grasp_xyz - target) >= 0.052)
    if self.stage == "lift":
      target = np.array((grasp_xyz[0], grasp_xyz[1], 0.62))
      if grasp_xyz[2] > 0.59 and self.stage_frames >= 4:
        self._set_stage("transport")
      return target, False
    if self.stage == "transport":
      target = np.array((tray_xyz[0], tray_xyz[1], 0.62))
      if np.linalg.norm(grasp_xyz[:2] - target[:2]) < 0.035:
        self._set_stage("lower")
      return target, False
    if self.stage == "lower":
      target = np.array((tray_xyz[0], tray_xyz[1], 0.515))
      if np.linalg.norm(grasp_xyz - target) < 0.035:
        self._set_stage("release")
      return target, False
    if self.stage == "release":
      if self.stage_frames >= 12:
        self.assist_local_offset = None
        self._set_stage("retreat")
      return grasp_xyz.copy(), True
    if self.stage == "retreat":
      target = np.array((tray_xyz[0], tray_xyz[1], 0.64))
      if grasp_xyz[2] > 0.60 and self.stage_frames >= 8:
        self.done = True
      return target, True
    raise RuntimeError(f"Unknown stage {self.stage}")

  def __call__(self, _observation: object) -> torch.Tensor:
    state = self.robot.data.joint_pos[0, ARM_SLICE].detach().cpu().numpy().copy()
    self._sync_kinematics(state)
    grasp_xyz = self.expert.grasp_point(self.data)
    object_xyz = self.object.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    tray_xyz = self.data.geom_xpos[self.tray_geom].copy()

    if (
      self.stage == "descend" and self._contacts() and self.assist_local_offset is None
    ):
      self._start_grasp_assist(grasp_xyz)

    target_xyz, gripper_open = self._stage_command(
      grasp_xyz,
      object_xyz,
      tray_xyz,
    )
    target_state, _ = self._ik_target(state, target_xyz, gripper_open)

    ego = self.env.scene["ego_camera"].data.rgb[0].detach().cpu().numpy()
    wrist = self.env.scene["wrist_camera"].data.rgb[0].detach().cpu().numpy()
    self.states.append(state.astype(np.float32))
    self.actions.append(target_state.copy())
    self.ego_frames.append(ego.astype(np.uint8))
    self.wrist_frames.append(wrist.astype(np.uint8))
    self.finger_effort.append(
      self.robot.data.actuator_force[0, 18:20].detach().cpu().numpy().copy()
    )
    self.stage_labels.append(self.stage)
    self.max_ego_pixels = max(
      self.max_ego_pixels,
      color_pixels(ego, self.object_name),
    )
    self.max_wrist_pixels = max(
      self.max_wrist_pixels,
      color_pixels(wrist, self.object_name),
    )

    if self.assist_local_offset is not None:
      target_state[6:] = 0.0
      self._apply_grasp_assist(target_state)

    action = torch.zeros(
      (self.env.num_envs, self.env.action_manager.total_action_dim),
      device=self.env.device,
    )
    target = torch.as_tensor(target_state, device=self.env.device).unsqueeze(0)
    offset = self.action_term.offset[:, ARM_SLICE]
    scale = self.action_term.scale[:, ARM_SLICE]
    action[:, ARM_SLICE] = (target - offset) / scale
    self.stage_frames += 1
    if self.stage == "descend" and self.stage_frames > 100:
      self.failed_reason = "no selected-object finger contact"
    elif self.stage_frames > 140:
      self.failed_reason = f"timeout in {self.stage}"
    return action


def randomized_layout(rng: np.random.Generator) -> dict[str, np.ndarray]:
  for _ in range(100):
    indices = rng.choice(len(LAYOUT_SLOTS), size=len(TASKS), replace=False)
    jitter = rng.uniform((-0.008, -0.008), (0.008, 0.008), size=(len(TASKS), 2))
    positions = LAYOUT_SLOTS[indices] + jitter
    distances = [
      np.linalg.norm(positions[first] - positions[second])
      for first in range(len(positions))
      for second in range(first + 1, len(positions))
    ]
    if min(distances) >= 0.08:
      return dict(zip(TASKS, positions, strict=True))
  raise RuntimeError("Could not sample a separated object layout")


def set_layout(
  env: ManagerBasedRlEnv,
  layout: dict[str, np.ndarray],
) -> None:
  for name, xy in layout.items():
    state = torch.zeros((1, 13), device=env.device)
    state[0, :3] = torch.tensor((float(xy[0]), float(xy[1]), 0.48), device=env.device)
    state[0, 3] = 1.0
    env.scene[name].write_root_state_to_sim(state)


def initial_visibility(env: ManagerBasedRlEnv) -> dict[str, int]:
  ego = env.scene["ego_camera"].data.rgb[0].detach().cpu().numpy()
  return {name: color_pixels(ego, name) for name in TASKS}


def next_episode_index(output: Path) -> int:
  indices = [
    int(path.name.removeprefix("episode_"))
    for path in output.glob("episode_[0-9][0-9][0-9][0-9]")
    if path.is_dir()
  ]
  return max(indices, default=-1) + 1


def save_episode(
  output: Path,
  index: int,
  demo: AutomaticDemo,
  layout: dict[str, np.ndarray],
) -> Path:
  episode_dir = output / f"episode_{index:04d}"
  episode_dir.mkdir(parents=True, exist_ok=False)
  np.savez_compressed(
    episode_dir / "trajectory.npz",
    observation_state=np.stack(demo.states),
    action=np.stack(demo.actions),
    finger_effort=np.stack(demo.finger_effort),
    stage=np.asarray(demo.stage_labels),
    grasp_assist=np.asarray(True),
    task=np.asarray(TASKS[demo.object_name]),
    target_object=np.asarray(demo.object_name),
    layout_names=np.asarray(tuple(layout)),
    layout_xy=np.stack(tuple(layout.values())),
    max_ego_object_pixels=np.asarray(demo.max_ego_pixels),
    max_wrist_object_pixels=np.asarray(demo.max_wrist_pixels),
    success=np.asarray(True),
    fps=np.asarray(FPS),
  )
  imageio.mimwrite(
    episode_dir / "ego_camera.mp4",
    demo.ego_frames,
    fps=FPS,
    quality=8,
  )
  imageio.mimwrite(
    episode_dir / "wrist_camera.mp4",
    demo.wrist_frames,
    fps=FPS,
    quality=8,
  )
  return episode_dir


def make_env() -> ManagerBasedRlEnv:
  scene = load_module("start_custom_scene.py", "vla_auto_collect_scene")
  cfg = scene.make_env_cfg()
  cfg.decimation = 10
  for camera in cfg.scene.sensors:
    camera.data_types = ("rgb",)
  finger_match = ContactMatch(
    mode="geom",
    pattern=(
      "robot/d1/Link7_1_collision",
      "robot/d1/Link7_2_collision",
    ),
  )
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    *(
      ContactSensorCfg(
        name=CONTACT_SENSORS[name],
        primary=finger_match,
        secondary=ContactMatch(mode="geom", pattern=f"{name}/collision"),
        fields=("found", "force"),
        reduce="maxforce",
        history_length=10,
      )
      for name in TASKS
    ),
  )
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  return ManagerBasedRlEnv(cfg, device=device)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--episodes-per-task", type=int, default=30)
  parser.add_argument("--seed", type=int, default=20260803)
  parser.add_argument("--max-attempts", type=int, default=400)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).parent / "data" / "go2_d1_multitask_raw",
  )
  parser.add_argument("--smoke-test", action="store_true")
  args = parser.parse_args()
  if args.episodes_per_task <= 0:
    parser.error("--episodes-per-task must be positive")

  env = make_env()
  rng = np.random.default_rng(args.seed)
  accepted = {name: 0 for name in TASKS}
  next_index = next_episode_index(args.output)
  target_total = 1 if args.smoke_test else args.episodes_per_task * len(TASKS)
  attempts = 0

  try:
    while sum(accepted.values()) < target_total and attempts < args.max_attempts:
      if args.smoke_test:
        object_name = "blue_cylinder"
      else:
        object_name = min(accepted, key=lambda name: (accepted[name], name))
      attempts += 1
      observation, _ = env.reset()
      layout = randomized_layout(rng)
      set_layout(env, layout)

      hold = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
      )
      action_term = env.action_manager.get_term("joint_position")
      waiting_arm = torch.zeros((1, 8), device=env.device)
      waiting_arm[0, 6:] = torch.tensor((0.028, -0.028), device=env.device)
      hold[:, ARM_SLICE] = (
        waiting_arm - action_term.offset[:, ARM_SLICE]
      ) / action_term.scale[:, ARM_SLICE]
      for _ in range(35):
        observation = env.step(hold)[0]
      visibility = initial_visibility(env)
      if min(visibility.values()) < 60:
        print(f"Attempt {attempts}: rejected invisible layout {visibility}")
        continue

      demo = AutomaticDemo(env, object_name)
      for _ in range(500):
        action = demo(observation)
        observation = env.step(action)[0]
        if demo.done or demo.failed_reason:
          break
      for _ in range(25):
        observation = env.step(hold)[0]

      final_xyz = (
        env.scene[object_name].data.root_link_pos_w[0].detach().cpu().numpy().copy()
      )
      tray_xy = env.sim.mj_model.geom_pos[demo.tray_geom, :2]
      tray_error = float(np.linalg.norm(final_xyz[:2] - tray_xy))
      success = bool(
        demo.done
        and demo.failed_reason is None
        and tray_error < 0.075
        and final_xyz[2] > 0.44
        and demo.max_ego_pixels >= 60
        and demo.max_wrist_pixels >= 100
      )
      if not success:
        print(
          f"Attempt {attempts}: REJECTED {object_name}; "
          f"reason={demo.failed_reason}, tray_error={tray_error:.3f}, "
          f"pixels=({demo.max_ego_pixels}, {demo.max_wrist_pixels})"
        )
        continue

      episode_dir = save_episode(args.output, next_index, demo, layout)
      accepted[object_name] += 1
      next_index += 1
      print(
        f"Attempt {attempts}: SAVED {episode_dir.name}, {object_name}, "
        f"{len(demo.actions)} frames, tray_error={tray_error:.3f}, "
        f"counts={accepted}"
      )
  finally:
    env.close()

  print(f"Accepted automatic demonstrations: {accepted}")
  if sum(accepted.values()) != target_total:
    raise RuntimeError(
      f"Collected {sum(accepted.values())}/{target_total} accepted episodes "
      f"after {attempts} attempts"
    )


if __name__ == "__main__":
  main()
