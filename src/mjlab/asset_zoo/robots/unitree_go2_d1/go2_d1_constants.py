"""Unitree Go2 EDU with mounted Unitree D1 arm constants."""

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_d1.d1_constants import (
  D1_ARTICULATION,
  D1_GRIPPER_ACTUATOR_CFG,
  D1_URDF,
)
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  FULL_COLLISION as GO2_FULL_COLLISION,
)
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  HIP_ACTUATOR,
  KNEE_ACTUATOR,
)
from mjlab.asset_zoo.robots.unitree_go2.go2_constants import (
  get_spec as get_go2_spec,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# Mount pose from the public go2_d1_integration xacro:
# go2_to_d1_mount origin xyz="-0.0071 0 0.087" rpy="0 0 0".
D1_MOUNT_POS = (-0.0071, 0.0, 0.087)


def _d1_collision_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(D1_URDF))
  for geom in spec.geoms:
    body_name = geom.parent.name
    geom.name = f"{body_name}_collision"
    geom.contype = 1
    geom.conaffinity = 1
    geom.condim = 4
    geom.friction = (
      (1.2, 0.02, 0.002) if body_name in {"Link7_1", "Link7_2"} else (0.8, 0.02, 0.002)
    )
  return spec


def get_spec() -> mujoco.MjSpec:
  spec = get_go2_spec()
  base = spec.body("base")
  frame = base.add_frame(pos=D1_MOUNT_POS)
  spec.attach(_d1_collision_spec(), prefix="d1/", frame=frame)
  return spec


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.27),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*hip_joint": 0.0,
    ".*Joint[1-6]": 0.0,
    ".*Joint7_1": 0.0,
    ".*Joint7_2": 0.0,
  },
  joint_vel={".*": 0.0},
)

GO2_D1_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=60.0,
  damping=5.0,
  effort_limit=HIP_ACTUATOR.effort_limit,
  armature=HIP_ACTUATOR.reflected_inertia,
)
GO2_D1_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=100.0,
  damping=8.0,
  effort_limit=KNEE_ACTUATOR.effort_limit,
  armature=KNEE_ACTUATOR.reflected_inertia,
)

GO2_D1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_D1_HIP_ACTUATOR_CFG,
    GO2_D1_KNEE_ACTUATOR_CFG,
    *D1_ARTICULATION.actuators,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_d1_robot_cfg() -> EntityCfg:
  """Get a fresh Go2+D1 mobile manipulator configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(GO2_FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_D1_ARTICULATION,
  )


GO2_D1_ACTION_SCALE: dict[str, float] = {}
for a in (GO2_D1_HIP_ACTUATOR_CFG, GO2_D1_KNEE_ACTUATOR_CFG):
  assert a.effort_limit is not None
  for name in a.target_names_expr:
    GO2_D1_ACTION_SCALE[name] = 0.25 * a.effort_limit / a.stiffness

for a in (D1_GRIPPER_ACTUATOR_CFG,):
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    GO2_D1_ACTION_SCALE[n] = 0.25 * e / s

for n in D1_ARTICULATION.actuators[0].target_names_expr:
  GO2_D1_ACTION_SCALE[n] = 0.05
