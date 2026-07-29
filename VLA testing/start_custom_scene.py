"""Launch a custom Go2+D1 mobile-manipulation scene in mjlab."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from mjlab.asset_zoo.robots import (
  GO2_D1_ACTION_SCALE,
  get_go2_d1_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import NativeMujocoViewer, ViewerConfig


class CameraPanelViewer(NativeMujocoViewer):
  """Native viewer with live robot-camera images on the right side."""

  _camera_names = ("ego_camera", "wrist_camera")
  _panel_size = 256
  _panel_margin = 12

  def sync_env_to_viewer(self) -> None:
    super().sync_env_to_viewer()
    viewer = self.viewer
    if viewer is None or not viewer.is_running():
      return

    window = viewer.viewport
    panel_x = max(self._panel_margin, window.width - self._panel_size - 12)
    panels: list[tuple[mujoco.MjrRect, np.ndarray]] = []

    for index, camera_name in enumerate(self._camera_names):
      rgb = self.env.unwrapped.scene[camera_name].data.rgb
      if rgb is None:
        continue

      panel_y = window.height - (index + 1) * (self._panel_size + self._panel_margin)
      panel_y = max(self._panel_margin, panel_y)
      viewport = mujoco.MjrRect(
        panel_x,
        panel_y,
        self._panel_size,
        self._panel_size,
      )
      image = rgb[self.env_idx].detach().cpu().numpy()
      panels.append((viewport, image))

    if panels:
      viewer.set_images(panels)


def _make_pickable_spec(
  *,
  geom_type: mujoco.mjtGeom,
  size: tuple[float, float, float],
  rgba: tuple[float, float, float, float],
) -> mujoco.MjSpec:
  """Create a single movable object."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="object")
  body.add_freejoint(name="free_joint")
  geom = body.add_geom(name="collision")
  geom.type = geom_type
  geom.size = size
  geom.rgba = rgba
  geom.density = 450.0
  geom.friction = (0.9, 0.02, 0.002)
  geom.condim = 4
  return spec


def _pickable_cfg(
  *,
  position: tuple[float, float, float],
  geom_type: mujoco.mjtGeom,
  size: tuple[float, float, float],
  rgba: tuple[float, float, float, float],
) -> EntityCfg:
  def spec_fn() -> mujoco.MjSpec:
    return _make_pickable_spec(
      geom_type=geom_type,
      size=size,
      rgba=rgba,
    )

  return EntityCfg(
    spec_fn=spec_fn,
    init_state=EntityCfg.InitialStateCfg(
      pos=position,
      joint_pos={},
      joint_vel={},
    ),
  )


def _add_box(
  spec: mujoco.MjSpec,
  *,
  name: str,
  pos: tuple[float, float, float],
  size: tuple[float, float, float],
  rgba: tuple[float, float, float, float],
) -> None:
  geom = spec.worldbody.add_geom(name=name)
  geom.type = mujoco.mjtGeom.mjGEOM_BOX
  geom.pos = pos
  geom.size = size
  geom.rgba = rgba
  geom.friction = (0.9, 0.02, 0.002)


def _customize_scene(spec: mujoco.MjSpec) -> None:
  """Add a low work table, target tray, and navigation obstacles."""
  wood = (0.42, 0.24, 0.10, 1.0)
  table_center = (0.72, 0.0, 0.40)
  _add_box(
    spec,
    name="work_table_top",
    pos=table_center,
    size=(0.32, 0.34, 0.025),
    rgba=wood,
  )
  for index, (x, y) in enumerate(
    ((0.45, -0.27), (0.45, 0.27), (0.99, -0.27), (0.99, 0.27))
  ):
    _add_box(
      spec,
      name=f"work_table_leg_{index}",
      pos=(x, y, 0.20),
      size=(0.025, 0.025, 0.20),
      rgba=wood,
    )

  # A shallow green goal tray inside the D1 arm's reachable workspace.
  tray_color = (0.12, 0.55, 0.22, 1.0)
  _add_box(
    spec,
    name="target_tray_base",
    pos=(0.47, -0.18, 0.435),
    size=(0.10, 0.09, 0.01),
    rgba=tray_color,
  )
  for index, (pos, size) in enumerate(
    (
      ((0.47, -0.265, 0.46), (0.10, 0.008, 0.025)),
      ((0.47, -0.095, 0.46), (0.10, 0.008, 0.025)),
      ((0.565, -0.18, 0.46), (0.008, 0.09, 0.025)),
      ((0.375, -0.18, 0.46), (0.008, 0.09, 0.025)),
    )
  ):
    _add_box(
      spec,
      name=f"target_tray_wall_{index}",
      pos=pos,
      size=size,
      rgba=tray_color,
    )

  # Obstacles leave a clear approach lane between the robot and table.
  obstacle_color = (0.26, 0.30, 0.36, 1.0)
  for name, pos, size in (
    ("obstacle_left", (0.05, 0.70, 0.16), (0.18, 0.12, 0.16)),
    ("obstacle_right", (0.05, -0.70, 0.13), (0.14, 0.16, 0.13)),
    ("obstacle_back_left", (-0.55, 0.52, 0.20), (0.13, 0.13, 0.20)),
    ("obstacle_back_right", (-0.55, -0.52, 0.11), (0.20, 0.10, 0.11)),
  ):
    _add_box(
      spec,
      name=name,
      pos=pos,
      size=size,
      rgba=obstacle_color,
    )


def _fixed_base_robot_cfg() -> EntityCfg:
  """Create the Go2+D1 configuration used for arm-only VLA experiments."""
  robot_cfg = get_go2_d1_robot_cfg()
  source_spec_fn = robot_cfg.spec_fn

  def fixed_base_spec() -> mujoco.MjSpec:
    spec = source_spec_fn()
    free_joint = next(
      joint for joint in spec.joints if joint.type == mujoco.mjtJoint.mjJNT_FREE
    )
    spec.delete(free_joint)
    # A free joint ignores the body's MJCF reference position, while the
    # fixed-base mocap wrapper adds init_state.pos. Remove that added offset
    # from the body reference so the compiled pose matches the free robot:
    # base z=0.445 m and foot centers z=0.019 m.
    base = spec.body("base")
    base.pos = (
      base.pos[0] - robot_cfg.init_state.pos[0],
      base.pos[1] - robot_cfg.init_state.pos[1],
      base.pos[2] - robot_cfg.init_state.pos[2],
    )
    return spec

  robot_cfg.spec_fn = fixed_base_spec
  return robot_cfg


def make_env_cfg(*, fixed_base: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the custom scene configuration."""
  cameras = (
    CameraSensorCfg(
      name="ego_camera",
      parent_body="robot/base",
      pos=(0.25, 0.0, 0.20),
      # MuJoCo cameras look along local -Z. This points forward and slightly down.
      quat=(-0.5425, -0.4536, 0.4536, 0.5425),
      fovy=75.0,
      width=256,
      height=256,
      data_types=("rgb", "depth"),
    ),
    CameraSensorCfg(
      name="wrist_camera",
      parent_body="robot/d1/Link6",
      pos=(0.05, 0.0, 0.03),
      quat=(-0.0281, 0.7065, -0.7065, 0.0281),
      fovy=90.0,
      width=256,
      height=256,
      data_types=("rgb", "depth"),
    ),
  )
  entities = {
    "robot": _fixed_base_robot_cfg() if fixed_base else get_go2_d1_robot_cfg(),
    "red_cube": _pickable_cfg(
      position=(0.48, 0.10, 0.46),
      geom_type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.0225, 0.0225, 0.0225),
      rgba=(0.85, 0.08, 0.06, 1.0),
    ),
    "blue_cylinder": _pickable_cfg(
      position=(0.75, 0.12, 0.47),
      geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      size=(0.028, 0.04, 0.0),
      rgba=(0.05, 0.25, 0.85, 1.0),
    ),
    "yellow_block": _pickable_cfg(
      position=(0.88, 0.10, 0.455),
      geom_type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(0.045, 0.022, 0.022),
      rgba=(0.95, 0.72, 0.05, 1.0),
    ),
  }
  actions: dict[str, ActionTermCfg] = {
    "joint_position": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=GO2_D1_ACTION_SCALE,
      use_default_offset=True,
    )
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities=entities,
      sensors=cameras,
      num_envs=1,
      env_spacing=3.0,
      spec_fn=_customize_scene,
    ),
    actions=actions,
    decimation=4,
    episode_length_s=0.0,
    sim=SimulationCfg(
      nconmax=120,
      njmax=800,
      mujoco=MujocoCfg(timestep=0.005),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=2.4,
      elevation=-18.0,
      azimuth=135.0,
    ),
  )


def main() -> None:
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(make_env_cfg(), device=device)
  action_dim = env.action_manager.total_action_dim

  def zero_policy(_observation: object) -> torch.Tensor:
    return torch.zeros((env.num_envs, action_dim), device=env.device)

  env.reset()
  print("Custom Go2+D1 scene started in MuJoCo Warp.")
  print("Sensors: ego_camera and wrist_camera (256x256 RGB + depth).")
  print("Close the viewer or press Ctrl+C to stop.")
  CameraPanelViewer(env, zero_policy).run()


if __name__ == "__main__":
  main()
