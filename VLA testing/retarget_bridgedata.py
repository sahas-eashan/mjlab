"""Retarget real BridgeData V2 Cartesian demonstrations to the simulated D1."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd

from mjlab.scene import Scene

FPS = 20
SOURCE_FPS = 5
TIME_SCALE = 2.0
CONTROL_SUBSTEPS = 10
TASK = "Pick up the red cube and place it in the green tray."


def load_local_module(name: str, filename: str) -> object:
  path = Path(__file__).with_name(filename)
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--episode-index", type=int, default=4)
  parser.add_argument(
    "--bridge-root",
    type=Path,
    default=Path(__file__).parent / "datasets" / "bridgedata_v2",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).parent / "data" / "bridgedata_d1_retargeted",
  )
  parser.add_argument("--no-video", action="store_true")
  return parser.parse_args()


def load_bridge_episode(
  root: Path, episode_index: int
) -> tuple[np.ndarray, np.ndarray]:
  data_path = root / "data" / "chunk-000" / "file-000.parquet"
  if not data_path.is_file():
    raise FileNotFoundError(
      f"{data_path} is missing. Download the first BridgeData action shard."
    )
  frame = pd.read_parquet(
    data_path,
    filters=[("episode_index", "=", episode_index)],
    columns=["state", "action", "frame_index"],
  ).sort_values("frame_index")
  if frame.empty:
    raise ValueError(f"Episode {episode_index} is not in the first data shard")
  return np.stack(frame["state"]), np.stack(frame["action"])


def gripper_events(actions: np.ndarray) -> tuple[int, int]:
  closed = actions[:, 6] < 0.5
  close_candidates = np.flatnonzero(closed)
  if close_candidates.size == 0:
    raise ValueError("BridgeData episode never closes the gripper")
  close_index = int(close_candidates[0])
  release_candidates = np.flatnonzero((np.arange(len(actions)) > close_index) & ~closed)
  if release_candidates.size == 0:
    raise ValueError("BridgeData episode never reopens the gripper")
  return close_index, int(release_candidates[0])


def retarget_positions(
  source_positions: np.ndarray,
  close_index: int,
  release_index: int,
) -> np.ndarray:
  # The cube settles with its center at z=0.4475 on the 0.425 m table top.
  # The 64 mm-tall fingers must sit slightly above that center so their lower
  # edges clear the table while retaining ample overlap with the 45 mm cube.
  # The coupled D1 fingers' physical opening is offset from the Link6-derived
  # grasp point when fully open, so compensate to center the gap on y=0.10.
  target_grasp = np.array((0.48, 0.109, 0.488))
  target_release = np.array((0.47, -0.18, 0.49))
  target_lift = np.array((target_grasp[0], target_grasp[1], 0.57))
  source_grasp = source_positions[close_index]
  source_release = source_positions[release_index]
  lift_index = close_index + int(
    np.argmax(source_positions[close_index : release_index + 1, 2])
  )
  source_lift = source_positions[lift_index]

  source_xy = source_release[:2] - source_grasp[:2]
  target_xy = target_release[:2] - target_grasp[:2]
  source_angle = np.arctan2(source_xy[1], source_xy[0])
  target_angle = np.arctan2(target_xy[1], target_xy[0])
  angle = target_angle - source_angle
  rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))

  retargeted = np.empty_like(source_positions)
  for index, source in enumerate(source_positions):
    if index <= close_index:
      offset = source - source_grasp
      # The D1 open gripper only has a few millimetres of clearance around
      # this cube. Approach vertically above its center instead of carrying
      # over the WidowX's embodiment-specific lateral approach.
      retargeted[index, :2] = target_grasp[:2]
      retargeted[index, 2] = target_grasp[2] + max(0.0, 0.7 * offset[2])
    elif index <= lift_index:
      progress = (index - close_index) / max(lift_index - close_index, 1)
      source_baseline = (1 - progress) * source_grasp + progress * source_lift
      target_baseline = (1 - progress) * target_grasp + progress * target_lift
      residual = source - source_baseline
      # Do not drag the cube sideways while the grasp is still seating.
      retargeted[index, :2] = target_grasp[:2] + 0.15 * rotation @ residual[:2]
      retargeted[index, 2] = target_baseline[2] + 0.3 * residual[2]
    elif index < release_index:
      progress = (index - lift_index) / max(release_index - lift_index, 1)
      source_baseline = (1 - progress) * source_lift + progress * source_release
      target_baseline = (1 - progress) * target_lift + progress * target_release
      residual = source - source_baseline
      retargeted[index, :2] = target_baseline[:2] + 0.4 * rotation @ residual[:2]
      retargeted[index, 2] = target_baseline[2] + 0.4 * residual[2]
    else:
      offset = source - source_release
      retargeted[index, :2] = target_release[:2] + 0.7 * rotation @ offset[:2]
      retargeted[index, 2] = target_release[2] + 0.7 * offset[2]
  return retargeted


def upsample_trajectory(
  positions: np.ndarray,
  actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  source_times = np.arange(len(positions)) / SOURCE_FPS
  duration = source_times[-1] * TIME_SCALE
  target_times = np.arange(round(duration * FPS) + 1) / FPS
  sample_times = target_times / TIME_SCALE
  upsampled_positions = np.column_stack(
    [np.interp(sample_times, source_times, positions[:, axis]) for axis in range(3)]
  )
  source_indices = np.minimum(
    (sample_times * SOURCE_FPS).astype(int),
    len(actions) - 1,
  )
  gripper_open = actions[source_indices, 6] >= 0.5
  return upsampled_positions, gripper_open


def build_model() -> mujoco.MjModel:
  scene_module = load_local_module("vla_custom_scene", "start_custom_scene.py")
  scene = Scene(scene_module.make_env_cfg().scene, device="cpu")
  keyframe = scene.spec.key("init_state")
  key_qpos = list(keyframe.qpos)
  scene.spec.delete(scene.spec.joint("robot/floating_base_joint"))
  scene.spec.body("robot/base").pos = (0.0, 0.0, 0.27)
  keyframe.qpos = key_qpos[7:]
  return scene.compile()


def render(
  renderer: mujoco.Renderer,
  data: mujoco.MjData,
  camera: str,
) -> np.ndarray:
  renderer.update_scene(data, camera=camera)
  return renderer.render().copy()


def run_retargeted_episode(
  model: mujoco.MjModel,
  positions: np.ndarray,
  gripper_open: np.ndarray,
  renderer: mujoco.Renderer | None,
  output_dir: Path,
) -> bool:
  collector = load_local_module("vla_collect_demos", "collect_demos.py")
  expert = collector.ScriptedExpert(model)
  object_id = collector._id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube/object")
  object_geom_ids = set(
    range(
      model.body_geomadr[object_id],
      model.body_geomadr[object_id] + model.body_geomnum[object_id],
    )
  )
  cube_joint = collector._id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube/free_joint")
  cube_qpos = model.jnt_qposadr[cube_joint]
  cube_dof = model.jnt_dofadr[cube_joint]

  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  data.ctrl[:] = model.key_ctrl[0]

  # Solve embodiment-specific joint waypoints while preserving the retargeted
  # Cartesian path and the D1's initial wrist orientation.
  waypoint_indices = np.unique(
    np.append(np.arange(0, len(positions), 8), len(positions) - 1)
  )
  waypoint_joints: list[np.ndarray] = []
  ik_data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, ik_data, 0)
  for index in waypoint_indices:
    joint_target = expert.solve_pose(ik_data, positions[index])
    waypoint_joints.append(joint_target)
    ik_data.qpos[expert.arm_qpos] = joint_target
    mujoco.mj_forward(model, ik_data)
  joint_targets = np.column_stack(
    [
      np.interp(
        np.arange(len(positions)), waypoint_indices, np.stack(waypoint_joints)[:, i]
      )
      for i in range(6)
    ]
  )

  def apply_joint_target(joint_target: np.ndarray, gripper: str) -> np.ndarray:
    finger_target = np.array((0.03, -0.03)) if gripper == "open" else np.zeros(2)
    data.ctrl[expert.arm_actuators] = joint_target
    data.ctrl[expert.gripper_actuators] = finger_target
    return np.concatenate((joint_target, finger_target)).astype(np.float32)

  # Keep the object out of the arm's unrecorded initialization sweep.
  data.qpos[cube_qpos : cube_qpos + 3] = (2.0, 2.0, 0.50)
  data.qpos[cube_qpos + 3 : cube_qpos + 7] = (1.0, 0.0, 0.0, 0.0)

  # Move to the real demonstration's initial pose before recording.
  for _ in range(150):
    apply_joint_target(joint_targets[0], "open")
    for _ in range(CONTROL_SUBSTEPS):
      mujoco.mj_step(model, data)

  data.qpos[cube_qpos : cube_qpos + 3] = (0.48, 0.10, 0.4475)
  data.qpos[cube_qpos + 3 : cube_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
  data.qvel[cube_dof : cube_dof + 6] = 0.0
  mujoco.mj_forward(model, data)
  for _ in range(100):
    apply_joint_target(joint_targets[0], "open")
    mujoco.mj_step(model, data)

  states: list[np.ndarray] = []
  commands: list[np.ndarray] = []
  cube_positions: list[np.ndarray] = []
  ego_frames: list[np.ndarray] = []
  wrist_frames: list[np.ndarray] = []
  was_open = True
  reported_contact = False

  for frame_index, (_target, joint_target, is_open) in enumerate(
    zip(positions, joint_targets, gripper_open, strict=True)
  ):
    gripper = "open" if is_open else "closed"
    dwell = 20 if was_open and not is_open else 1
    for _ in range(dwell):
      command = apply_joint_target(joint_target, gripper)
      for _ in range(CONTROL_SUBSTEPS):
        mujoco.mj_step(model, data)
        if not reported_contact:
          for contact in data.contact:
            if contact.geom1 in object_geom_ids or contact.geom2 in object_geom_ids:
              other = (
                contact.geom2 if contact.geom1 in object_geom_ids else contact.geom1
              )
              other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
              if other_name != "work_table_top":
                print(
                  f"First non-table cube contact at frame {frame_index}: "
                  f"{other_name} (gripper={gripper})"
                )
                reported_contact = True
                break
      states.append(expert.state(data))
      commands.append(command)
      cube_positions.append(data.xpos[object_id].copy())
      if renderer is not None:
        ego_frames.append(render(renderer, data, "ego_camera"))
        wrist_frames.append(render(renderer, data, "wrist_camera"))
    was_open = bool(is_open)

  for _ in range(100):
    mujoco.mj_step(model, data)
  cube_position = data.xpos[object_id].copy()
  success = bool(
    abs(cube_position[0] - 0.47) < 0.11
    and abs(cube_position[1] + 0.18) < 0.10
    and cube_position[2] > 0.44
  )

  output_dir.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    output_dir / "trajectory.npz",
    observation_state=np.stack(states),
    action=np.stack(commands),
    cube_position=np.stack(cube_positions),
    task=np.asarray(TASK),
    success=np.asarray(success),
    fps=np.asarray(FPS),
    source=np.asarray("BridgeData V2 real WidowX trajectory"),
  )
  if renderer is not None:
    imageio.mimwrite(output_dir / "ego_camera.mp4", ego_frames, fps=FPS, quality=8)
    imageio.mimwrite(
      output_dir / "wrist_camera.mp4",
      wrist_frames,
      fps=FPS,
      quality=8,
    )
  print(f"Final cube position: {cube_position.round(4).tolist()}")
  print(f"Physical-contact result: {'SUCCESS' if success else 'FAILED'}")
  return success


def main() -> None:
  args = parse_args()
  states, actions = load_bridge_episode(args.bridge_root, args.episode_index)
  close_index, release_index = gripper_events(actions)
  retargeted = retarget_positions(states[:, :3], close_index, release_index)
  positions, gripper_open = upsample_trajectory(retargeted, actions)
  print(
    f"BridgeData episode {args.episode_index}: {len(states)} real frames, "
    f"close={close_index}, release={release_index}"
  )

  model = build_model()
  renderer = None if args.no_video else mujoco.Renderer(model, height=256, width=256)
  output_dir = args.output / f"episode_{args.episode_index:06d}"
  try:
    success = run_retargeted_episode(
      model,
      positions,
      gripper_open,
      renderer,
      output_dir,
    )
  finally:
    if renderer is not None:
      renderer.close()
  if not success:
    raise RuntimeError(
      "The retargeted real demonstration did not physically place the cube; "
      "it was retained for contact debugging and must not be used for training."
    )


if __name__ == "__main__":
  main()
