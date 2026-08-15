"""Run a language-conditioned SmolVLA policy on the Go2+D1 test scene."""

from __future__ import annotations

import argparse
import importlib.util
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg

VLA_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = (
  VLA_DIR
  / "outputs"
  / "smolvla_multitask_50k"
  / "checkpoints"
  / "last"
  / "pretrained_model"
)
ARM_SLICE = slice(12, 20)
ARM_MAX_STEP = torch.tensor((0.04,) * 6 + (0.004, 0.004))
OBJECT_CONTACT_SENSORS = {
  "red_cube": "red_cube_finger_contact",
  "yellow_block": "yellow_block_finger_contact",
  "blue_cylinder": "blue_cylinder_finger_contact",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "instruction",
    nargs="?",
    default="Pick up the red cube and place it in the green tray.",
  )
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument(
    "--execute",
    action="store_true",
    help="Apply SmolVLA actions. Without this flag, predictions are dry-run only.",
  )
  parser.add_argument(
    "--device",
    choices=("cuda", "cpu"),
    default="cuda" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Run one held-action inference without opening the viewer.",
  )
  parser.add_argument(
    "--replan-steps",
    type=int,
    default=10,
    help="Execute this many predicted actions before observing the cameras again.",
  )
  parser.add_argument(
    "--arm-speed-scale",
    type=float,
    default=1.0,
    help="Scale the per-control-step D1 joint-motion limit (0 < scale <= 1).",
  )
  parser.add_argument(
    "--async-inference",
    action="store_true",
    help="Run SmolVLA chunk inference off the viewer/control thread.",
  )
  parser.add_argument(
    "--grasp-assist",
    action="store_true",
    help="Use the same contact-triggered grasp follower as assisted demos.",
  )
  parser.add_argument(
    "--presentation-assist",
    action="store_true",
    help=(
      "Use an explicitly assisted language-selected Cartesian pick/place "
      "supervisor for presentation rollouts."
    ),
  )
  parser.add_argument(
    "--vla-assist",
    action="store_true",
    help=(
      "Keep SmolVLA in control and apply only short-range object/tray alignment "
      "plus contact grasp assistance."
    ),
  )
  parser.add_argument(
    "--interactive-instructions",
    action="store_true",
    help="Keep the loaded model running and accept new tasks from the terminal.",
  )
  parser.add_argument(
    "--compare-instructions",
    action="store_true",
    help="Compare initial predictions for red, yellow, and blue tasks.",
  )
  parser.add_argument(
    "--rollout-steps",
    type=int,
    default=0,
    help="Run this many control steps headlessly, then report object motion.",
  )
  args = parser.parse_args()
  if not 0.0 < args.arm_speed_scale <= 1.0:
    parser.error("--arm-speed-scale must be greater than 0 and at most 1")
  if args.presentation_assist and args.vla_assist:
    parser.error("--presentation-assist and --vla-assist are mutually exclusive")
  if args.async_inference and args.compare_instructions:
    parser.error("--async-inference cannot be used with --compare-instructions")
  if args.async_inference and args.smoke_test:
    parser.error("--async-inference cannot be used with --smoke-test")
  if args.async_inference and not args.execute:
    parser.error("--async-inference requires --execute")
  return args


def load_scene_module() -> Any:
  path = VLA_DIR / "start_custom_scene.py"
  spec = importlib.util.spec_from_file_location("vla_runtime_scene", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load scene from {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def selected_object(instruction: str) -> str:
  instruction = instruction.lower()
  if "yellow" in instruction:
    return "yellow_block"
  if "blue" in instruction or "cylinder" in instruction:
    return "blue_cylinder"
  return "red_cube"


def load_policy(
  checkpoint: Path,
  device: str,
  replan_steps: int,
) -> tuple[SmolVLAPolicy, Any, Any]:
  checkpoint = checkpoint.resolve()
  if not (checkpoint / "config.json").is_file():
    raise FileNotFoundError(f"SmolVLA checkpoint is missing config.json: {checkpoint}")

  config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
  if not isinstance(config, SmolVLAConfig):
    raise TypeError(f"Expected a SmolVLA checkpoint, got policy type {config.type!r}")
  if not 1 <= replan_steps <= config.chunk_size:
    raise ValueError(
      f"replan_steps must be between 1 and {config.chunk_size}, got {replan_steps}"
    )
  config.device = device
  config.n_action_steps = replan_steps
  config.pretrained_path = checkpoint
  policy = SmolVLAPolicy.from_pretrained(
    checkpoint,
    config=config,
    local_files_only=True,
  )
  policy.eval()
  policy.reset()

  preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=config,
    pretrained_path=str(checkpoint),
    preprocessor_overrides={"device_processor": {"device": device}},
  )
  return policy, preprocessor, postprocessor


class SafeSmolVLAPolicy:
  """Adapt SmolVLA's eight absolute D1 targets to mjlab's 20 actions."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    policy: SmolVLAPolicy,
    preprocessor: Any,
    postprocessor: Any,
    instruction: str,
    *,
    execute: bool,
    grasp_assist: bool,
    presentation_assist: bool,
    vla_assist: bool,
    interactive_instructions: bool,
    arm_speed_scale: float,
    async_inference: bool,
    inference_chunk_steps: int,
  ):
    self.env = env
    self.policy = policy
    self.preprocessor = preprocessor
    self.postprocessor = postprocessor
    self.instruction = instruction
    self.execute = execute
    self.presentation_assist = presentation_assist
    self.vla_assist = vla_assist
    self.grasp_assist = grasp_assist or presentation_assist or vla_assist
    self.interactive_instructions = interactive_instructions
    self.arm_speed_scale = arm_speed_scale
    self.async_inference = async_inference
    self.inference_chunk_steps = inference_chunk_steps
    self._async_low_watermark = max(2, int(0.8 * inference_chunk_steps))
    self._instruction_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
    self._instruction_active = not interactive_instructions
    self._instruction_ready = threading.Event()
    if interactive_instructions:
      self._instruction_ready.set()
    self.selected_object = selected_object(instruction)
    self.robot = env.scene["robot"]
    self.action_term = env.action_manager._terms["joint_position"]
    self.step_count = 0
    self._warned_stability = False
    self._assist_local_offset: np.ndarray | None = None
    self._assist_object_quat: torch.Tensor | None = None
    self._assist_seen_close = False
    self._assist_release_latched = False
    self._tray_release_frames = 0
    self._presentation_stage = "approach"
    self._completion_pending = False
    self._waiting_arm_target = torch.zeros((1, 8), device=self.env.device)
    self._waiting_arm_target[0, 6:] = torch.tensor(
      (0.028, -0.028),
      device=self.env.device,
    )
    self._async_condition = threading.Condition()
    self._async_observation: dict[str, Any] | None = None
    self._async_actions: deque[torch.Tensor] = deque()
    self._async_desired: torch.Tensor | None = None
    self._async_reset_requested = False
    self._async_generation = 0
    self._async_error: BaseException | None = None
    self._async_stopping = False
    self._async_worker: threading.Thread | None = None

    collector_path = VLA_DIR / "collect_demos.py"
    collector_spec = importlib.util.spec_from_file_location(
      "vla_runtime_collector",
      collector_path,
    )
    if collector_spec is None or collector_spec.loader is None:
      raise RuntimeError(f"Could not load controller from {collector_path}")
    collector = importlib.util.module_from_spec(collector_spec)
    sys.modules[collector_spec.name] = collector
    collector_spec.loader.exec_module(collector)
    self.ik_model = env.sim.mj_model
    self.ik_data = mujoco.MjData(self.ik_model)
    self.expert = collector.ScriptedExpert(self.ik_model)
    self._position_jacobian = np.zeros((3, self.ik_model.nv))
    self._rotation_jacobian = np.zeros((3, self.ik_model.nv))
    self._tray_geom = mujoco.mj_name2id(
      self.ik_model,
      mujoco.mjtObj.mjOBJ_GEOM,
      "target_tray_base",
    )
    if self._tray_geom < 0:
      raise RuntimeError("Could not find target tray geometry")
    if self.async_inference:
      self._async_worker = threading.Thread(
        target=self._inference_worker,
        name="smolvla-inference",
        daemon=True,
      )
      self._async_worker.start()

  def reset(self) -> None:
    self._reset_task_state()
    if self.interactive_instructions:
      while True:
        try:
          self._instruction_queue.get_nowait()
        except queue.Empty:
          break
      self._instruction_active = False
      self._instruction_ready.set()
      print("\nScene and console reset. Arm holding the waiting pose.")

  def _reset_task_state(self) -> None:
    if self.async_inference:
      with self._async_condition:
        self._async_generation += 1
        self._async_observation = None
        self._async_actions.clear()
        self._async_desired = None
        self._async_reset_requested = True
        self._async_condition.notify_all()
    else:
      self.policy.reset()
    self.step_count = 0
    self._assist_local_offset = None
    self._assist_object_quat = None
    self._assist_seen_close = False
    self._assist_release_latched = False
    self._tray_release_frames = 0
    self._completion_pending = False
    self._set_presentation_stage("approach")

  def _inference_worker(self) -> None:
    """Generate action chunks without blocking simulation or viewer rendering."""
    while True:
      with self._async_condition:
        self._async_condition.wait_for(
          lambda: self._async_stopping
          or (
            self._async_observation is not None
            and len(self._async_actions) <= self._async_low_watermark
          )
        )
        if self._async_stopping:
          return
        observation = self._async_observation
        self._async_observation = None
        generation = self._async_generation
        reset_requested = self._async_reset_requested
        self._async_reset_requested = False

      assert observation is not None
      try:
        if reset_requested:
          self.policy.reset()
        processed = self.preprocessor(observation)
        chunk: list[torch.Tensor] = []
        with torch.inference_mode():
          for _ in range(self.inference_chunk_steps):
            action = self.postprocessor(self.policy.select_action(processed))
            chunk.append(action.detach().clone())
      except BaseException as error:
        with self._async_condition:
          self._async_error = error
        return

      with self._async_condition:
        if generation != self._async_generation:
          continue
        discarded = len(self._async_actions)
        self._async_actions.clear()
        self._async_actions.extend(chunk)
        print(
          f"Async SmolVLA: queued {len(chunk)} camera-conditioned actions "
          f"(replaced {discarded} stale actions)."
        )

  def _async_arm_target(self) -> torch.Tensor:
    observation = self._policy_observation()
    with self._async_condition:
      if self._async_error is not None:
        raise RuntimeError("Asynchronous SmolVLA inference failed") from (
          self._async_error
        )
      self._async_observation = observation
      if self._async_actions:
        self._async_desired = self._async_actions.popleft()
      self._async_condition.notify_all()
      desired = self._async_desired
    if desired is None:
      return self._safe_arm_target(self._waiting_arm_target)
    return self._safe_arm_target(desired)

  def close(self) -> None:
    if not self.async_inference:
      return
    with self._async_condition:
      self._async_stopping = True
      self._async_condition.notify_all()
    if self._async_worker is not None:
      self._async_worker.join(timeout=5.0)

  def submit_instruction(self, instruction: str) -> None:
    """Queue a task change from the terminal input thread."""
    self._instruction_ready.clear()
    self._instruction_queue.put(instruction)

  def wait_until_instruction_ready(self) -> None:
    """Block the terminal thread until the arm is ready for another task."""
    self._instruction_ready.wait()

  def _apply_pending_instruction(self) -> None:
    instruction: str | None = None
    while True:
      try:
        instruction = self._instruction_queue.get_nowait()
      except queue.Empty:
        break
    if instruction is None:
      return

    self.instruction = instruction
    self.selected_object = selected_object(instruction)
    self._reset_task_state()
    self._instruction_active = True
    print(f"\nNew task: {instruction}")
    print(f"Selected object: {self.selected_object}")

  def _set_presentation_stage(self, stage: str) -> None:
    if stage != self._presentation_stage:
      print(f"Presentation assist: {self._presentation_stage} -> {stage}")
    self._presentation_stage = stage

  def _cartesian_target(
    self,
    target_xyz: np.ndarray,
    *,
    gripper_open: bool,
  ) -> torch.Tensor:
    """Return a position-only D1 IK step while preserving free wrist rotation."""
    current = self.robot.data.joint_pos[0, ARM_SLICE].detach().cpu().numpy()
    self._sync_kinematics(current)
    grasp_point = self.expert.grasp_point(self.ik_data)
    mujoco.mj_jac(
      self.ik_model,
      self.ik_data,
      self._position_jacobian,
      self._rotation_jacobian,
      grasp_point,
      self.expert.link_id,
    )
    arm_dofs = self.ik_model.jnt_dofadr[self.expert.arm_joint_ids]
    jacobian = self._position_jacobian[:, arm_dofs]
    error = target_xyz - grasp_point
    damping = 0.06
    delta = jacobian.T @ np.linalg.solve(
      jacobian @ jacobian.T + damping**2 * np.eye(3),
      error,
    )
    result = current.copy()
    result[:6] += np.clip(delta, -0.04, 0.04)
    result[6:] = (0.028, -0.028) if gripper_open else (0.0, 0.0)
    return self._safe_arm_target(
      torch.as_tensor(result, device=self.env.device).unsqueeze(0)
    )

  def _presentation_override(self) -> torch.Tensor:
    """Run the disclosed deterministic supervisor used for presentation video."""
    entity = self.env.scene[self.selected_object]
    object_xyz = entity.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    current = self.robot.data.joint_pos[0, ARM_SLICE].detach().cpu().numpy()
    self._sync_kinematics(current)
    tray_xyz = self.ik_data.geom_xpos[self._tray_geom].copy()
    grasp_xyz = self.expert.grasp_point(self.ik_data)

    if self._assist_local_offset is not None and self._presentation_stage in {
      "approach",
      "descend",
    }:
      self._set_presentation_stage("lift")

    stage = self._presentation_stage
    if stage == "approach":
      target = object_xyz + np.array((0.0, 0.0, 0.13))
      if np.linalg.norm(grasp_xyz - target) < 0.045:
        self._set_presentation_stage("descend")
      return self._cartesian_target(target, gripper_open=True)

    if stage == "descend":
      target = object_xyz + np.array((0.0, 0.0, 0.015))
      close = np.linalg.norm(grasp_xyz - target) < 0.055
      return self._cartesian_target(target, gripper_open=not close)

    if stage == "lift":
      target = np.array((grasp_xyz[0], grasp_xyz[1], 0.62))
      if grasp_xyz[2] > 0.59:
        self._set_presentation_stage("transport")
      return self._cartesian_target(target, gripper_open=False)

    if stage == "transport":
      target = np.array((tray_xyz[0], tray_xyz[1], 0.62))
      if np.linalg.norm(grasp_xyz[:2] - target[:2]) < 0.045:
        self._set_presentation_stage("lower")
      return self._cartesian_target(target, gripper_open=False)

    if stage == "lower":
      target = np.array((tray_xyz[0], tray_xyz[1], 0.515))
      if np.linalg.norm(grasp_xyz - target) < 0.04:
        self._set_presentation_stage("release")
      return self._cartesian_target(target, gripper_open=False)

    if stage == "release":
      target = grasp_xyz.copy()
      self._set_presentation_stage("done")
      if self.interactive_instructions:
        self._instruction_active = False
        self._completion_pending = True
      return self._cartesian_target(target, gripper_open=True)

    target = np.array((grasp_xyz[0], grasp_xyz[1], 0.62))
    return self._cartesian_target(target, gripper_open=True)

  def _local_vla_correction(self, arm_target: torch.Tensor) -> torch.Tensor:
    """Correct the local grasp/release geometry around a VLA trajectory."""
    current = self.robot.data.joint_pos[0, ARM_SLICE].detach().cpu().numpy()
    self._sync_kinematics(current)
    grasp_xyz = self.expert.grasp_point(self.ik_data)
    entity = self.env.scene[self.selected_object]
    object_xyz = entity.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    tray_xyz = self.ik_data.geom_xpos[self._tray_geom].copy()
    model_gripper_open = bool(torch.max(torch.abs(arm_target[0, 6:])) > 0.012)

    if self._assist_release_latched:
      released = arm_target.clone()
      released[0, 6:] = torch.tensor(
        (0.028, -0.028),
        device=self.env.device,
      )
      return self._safe_arm_target(released)

    correction_target: np.ndarray | None = None
    correction_name: str | None = None
    blend = 0.0
    corrected_gripper_open = model_gripper_open
    if self._assist_local_offset is None:
      object_distance = float(np.linalg.norm(grasp_xyz - object_xyz))
      if object_distance < 0.20:
        # Enter above the object before descending so the fingers do not push
        # it sideways. Inside the final grasp zone, local geometry—not a noisy
        # open/close prediction—decides when to close the fingers.
        height_offset = 0.075 if object_distance >= 0.10 else 0.015
        correction_target = object_xyz + np.array((0.0, 0.0, height_offset))
        correction_name = "local grasp approach"
        target_error = float(np.linalg.norm(grasp_xyz - correction_target))
        corrected_gripper_open = target_error >= 0.052
        blend = 0.65 if object_distance >= 0.10 else 1.0
    else:
      tray_xy_distance = float(np.linalg.norm(object_xyz[:2] - tray_xyz[:2]))
      if tray_xy_distance < 0.12:
        correction_target = np.array((tray_xyz[0], tray_xyz[1], 0.515))
        correction_name = "tray alignment"
        blend = 0.25
        corrected_gripper_open = model_gripper_open

    if correction_target is None:
      return arm_target

    corrected = self._cartesian_target(
      correction_target,
      gripper_open=corrected_gripper_open,
    )
    result = arm_target.clone()
    result[:, :6] = (1.0 - blend) * arm_target[:, :6] + blend * corrected[:, :6]
    if self._assist_local_offset is None:
      result[:, 6:] = corrected[:, 6:]
    if self.step_count % 20 == 0:
      error = float(np.linalg.norm(grasp_xyz - correction_target))
      gripper = "open" if corrected_gripper_open else "closing"
      print(
        f"VLA assist: {correction_name}, blend={blend:.2f}, "
        f"error={error:.3f} m, gripper={gripper}"
      )
    return self._safe_arm_target(result)

  def _sync_kinematics(self, state: np.ndarray) -> None:
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

  def _update_grasp_assist(self, arm_target: torch.Tensor) -> None:
    finger_open = bool(torch.max(torch.abs(arm_target[0, 6:])) > 0.012)
    sensor = self.env.scene[OBJECT_CONTACT_SENSORS[self.selected_object]]
    assert sensor.data.found is not None
    any_contact = bool(torch.any(sensor.data.found[0, :2] > 0))
    current_state = self.robot.data.joint_pos[0, ARM_SLICE].detach().cpu().numpy()
    self._sync_kinematics(current_state)
    entity = self.env.scene[self.selected_object]

    if self.vla_assist and self._assist_local_offset is not None:
      tray_xyz = self.ik_data.geom_xpos[self._tray_geom]
      object_xyz = entity.data.root_link_pos_w[0, :3].detach().cpu().numpy()
      inside_release_region = bool(
        np.linalg.norm(object_xyz[:2] - tray_xyz[:2]) < 0.10 and object_xyz[2] < 0.56
      )
      self._tray_release_frames = (
        self._tray_release_frames + 1 if inside_release_region else 0
      )
      if self._tray_release_frames >= 8:
        print("Tray dwell complete: opening gripper and releasing object.")
        arm_target[0, 6:] = torch.tensor(
          (0.028, -0.028),
          device=self.env.device,
        )
        self._assist_local_offset = None
        self._assist_object_quat = None
        self._assist_seen_close = False
        self._assist_release_latched = True
        return

    if (
      any_contact
      and self._assist_local_offset is None
      and not self._assist_release_latched
    ):
      object_pose = entity.data.root_link_pose_w[0].detach().clone()
      grasp_point = self.expert.grasp_point(self.ik_data)
      rotation = self.ik_data.xmat[self.expert.link_id].reshape(3, 3)
      self._assist_local_offset = rotation.T @ (
        object_pose[:3].cpu().numpy() - grasp_point
      )
      self._assist_object_quat = object_pose[3:7]
      self._assist_seen_close = True
      print(f"Contact detected: auto-closing on {self.selected_object}.")

    if self._assist_local_offset is None:
      return

    if not finger_open:
      self._assist_seen_close = True
    elif self._assist_seen_close:
      allow_release = True
      if self.vla_assist:
        tray_xyz = self.ik_data.geom_xpos[self._tray_geom]
        object_xyz = entity.data.root_link_pos_w[0, :3].detach().cpu().numpy()
        allow_release = bool(
          np.linalg.norm(object_xyz[:2] - tray_xyz[:2]) < 0.12 and object_xyz[2] < 0.56
        )
      if allow_release:
        source = "presentation supervisor" if self.presentation_assist else "SmolVLA"
        print(f"Grasp assist released by {source} open command.")
        self._assist_local_offset = None
        self._assist_object_quat = None
        self._assist_seen_close = False
        self._assist_release_latched = True
        arm_target[0, 6:] = torch.tensor(
          (0.028, -0.028),
          device=self.env.device,
        )
        return

    # Keep the fingers closed after contact while waiting for the model to
    # enter its learned closed/transport phase.
    arm_target[0, 6:] = 0.0

    predicted = arm_target[0].detach().cpu().numpy()
    self.ik_data.qpos[self.expert.state_qpos] = predicted
    mujoco.mj_forward(self.ik_model, self.ik_data)
    grasp_point = self.expert.grasp_point(self.ik_data)
    rotation = self.ik_data.xmat[self.expert.link_id].reshape(3, 3)
    desired = grasp_point + rotation @ self._assist_local_offset
    current = entity.data.root_link_pose_w[0, :3].detach().cpu().numpy()
    velocity_np = np.clip(
      (desired - current) * 20.0 + np.array((0.0, 0.0, 0.5 * 9.81 / 20.0)),
      -0.65,
      0.65,
    )
    velocity = torch.zeros((1, 6), device=self.env.device)
    velocity[0, :3] = torch.as_tensor(velocity_np, device=self.env.device)
    entity.write_root_link_velocity_to_sim(velocity)

  def _is_stable(self) -> bool:
    position = self.robot.data.root_link_pos_w[0]
    quaternion = self.robot.data.root_link_quat_w[0]
    upright = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
    return bool(position[2] > 0.18 and upright > 0.75)

  def _policy_observation(self) -> dict[str, Any]:
    state = self.robot.data.joint_pos[:, ARM_SLICE].detach().cpu()
    ego = self.env.scene["ego_camera"].data.rgb
    wrist = self.env.scene["wrist_camera"].data.rgb
    if ego is None or wrist is None:
      raise RuntimeError("Robot camera RGB data is unavailable")

    def prepare_image(image: torch.Tensor) -> torch.Tensor:
      if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(
          f"Expected batched HWC uint8 camera image, got {image.shape}/{image.dtype}"
        )
      return image.detach().cpu().permute(0, 3, 1, 2).float().div(255.0)

    return {
      "observation.state": state,
      "observation.images.ego": prepare_image(ego),
      "observation.images.wrist": prepare_image(wrist),
      "task": [self.instruction],
    }

  def _safe_arm_target(self, predicted: torch.Tensor) -> torch.Tensor:
    predicted = predicted.to(self.env.device, dtype=torch.float32).reshape(1, 8)
    if not torch.isfinite(predicted).all():
      raise RuntimeError("SmolVLA produced a non-finite action")

    current = self.robot.data.joint_pos[:, ARM_SLICE]
    max_step = ARM_MAX_STEP.to(self.env.device).unsqueeze(0) * self.arm_speed_scale
    target = torch.clamp(predicted, current - max_step, current + max_step)
    limits = self.robot.data.joint_pos_limits[:, ARM_SLICE]
    return torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])

  def _to_environment_action(self, arm_target: torch.Tensor) -> torch.Tensor:
    actions = torch.zeros(
      (self.env.num_envs, self.env.action_manager.total_action_dim),
      device=self.env.device,
    )
    offset = self.action_term._offset[:, ARM_SLICE]
    scale = self.action_term._scale[:, ARM_SLICE]
    actions[:, ARM_SLICE] = (arm_target - offset) / scale
    return actions

  def prepare_initial_pose(self, steps: int = 30) -> object:
    """Match the open-gripper initial state used by teleoperated demos."""
    arm_target = self.robot.data.joint_pos[:, ARM_SLICE].clone()
    arm_target[:, 6:] = torch.tensor(
      (0.028, -0.028),
      device=self.env.device,
    )
    observation: object = {}
    for _ in range(steps):
      observation = self.env.step(self._to_environment_action(arm_target))[0]
    if self.async_inference:
      with self._async_condition:
        self._async_reset_requested = True
        self._async_condition.notify_all()
    else:
      self.policy.reset()
    self.step_count = 0
    print("Initial pose prepared: arm held, gripper open.")
    return observation

  def __call__(self, _observation: object) -> torch.Tensor:
    self._apply_pending_instruction()
    if not self._instruction_active:
      arm_target = self._safe_arm_target(self._waiting_arm_target)
      return self._to_environment_action(arm_target)

    if self.async_inference:
      arm_target = self._async_arm_target()
    else:
      observation = self.preprocessor(self._policy_observation())
      with torch.inference_mode():
        predicted = self.postprocessor(self.policy.select_action(observation))
      arm_target = self._safe_arm_target(predicted)

    if self.step_count % 20 == 0:
      mode = "EXECUTE" if self.execute else "DRY RUN"
      values = np.round(arm_target[0].detach().cpu().numpy(), 3).tolist()
      print(f"[{mode}] step={self.step_count}, D1 target={values}")
    self.step_count += 1

    if not self.execute:
      arm_target = self.robot.data.joint_pos[:, ARM_SLICE].clone()
    elif not self._is_stable():
      arm_target = self.robot.data.joint_pos[:, ARM_SLICE].clone()
      if not self._warned_stability:
        print("Safety stop: Go2 tilt/height limit reached; holding the D1 arm.")
        self._warned_stability = True

    if self.execute and self.presentation_assist and self._is_stable():
      arm_target = self._presentation_override()
    elif self.execute and self.vla_assist and self._is_stable():
      arm_target = self._local_vla_correction(arm_target)

    if self.execute and self.grasp_assist:
      self._update_grasp_assist(arm_target)

    if self._completion_pending:
      self._completion_pending = False
      self._instruction_ready.set()
      print("\nTask complete. Arm returning to its waiting pose.")

    # Zero leg actions map to the configured standing-pose offsets.
    return self._to_environment_action(arm_target)


def main() -> None:
  args = parse_args()
  scene_module = load_scene_module()
  config = scene_module.make_env_cfg()
  config.decimation = 10  # 0.005 s simulation step -> 20 Hz VLA control.
  if args.grasp_assist or args.presentation_assist or args.vla_assist:
    finger_match = ContactMatch(
      mode="geom",
      pattern=(
        "robot/d1/Link7_1_collision",
        "robot/d1/Link7_2_collision",
      ),
    )
    contact_sensors = tuple(
      ContactSensorCfg(
        name=sensor_name,
        primary=finger_match,
        secondary=ContactMatch(
          mode="geom",
          pattern=f"{object_name}/collision",
        ),
        fields=("found", "force"),
        reduce="maxforce",
        history_length=10,
      )
      for object_name, sensor_name in OBJECT_CONTACT_SENSORS.items()
    )
    config.scene.sensors = (*config.scene.sensors, *contact_sensors)
  env = ManagerBasedRlEnv(config, device=args.device)
  observation, _ = env.reset()

  print(f"Loading SmolVLA checkpoint: {args.checkpoint.resolve()}")
  policy, preprocessor, postprocessor = load_policy(
    args.checkpoint,
    args.device,
    args.replan_steps,
  )
  runtime_policy = SafeSmolVLAPolicy(
    env,
    policy,
    preprocessor,
    postprocessor,
    args.instruction,
    execute=args.execute,
    grasp_assist=args.grasp_assist,
    presentation_assist=args.presentation_assist,
    vla_assist=args.vla_assist,
    interactive_instructions=args.interactive_instructions,
    arm_speed_scale=args.arm_speed_scale,
    async_inference=args.async_inference,
    inference_chunk_steps=args.replan_steps,
  )
  observation = runtime_policy.prepare_initial_pose()
  mode = "EXECUTION ENABLED" if args.execute else "DRY RUN (arm held)"
  print(f"Instruction: {args.instruction}")
  print(f"Mode: {mode}")
  if args.grasp_assist:
    print(f"Grasp assist: enabled for {selected_object(args.instruction)}")
  if args.presentation_assist:
    print(
      "Presentation assist: ENABLED (language-selected deterministic Cartesian "
      "supervisor; not end-to-end VLA autonomy)"
    )
  if args.vla_assist:
    print(
      "VLA assist: ENABLED (SmolVLA trajectory with local alignment/contact "
      "patches; failures remain possible)"
    )
  if args.interactive_instructions:
    print("Interactive instructions: waiting for a task in this terminal.")
    print("Type red, yellow, blue, or a complete instruction, then press Enter.")

    def read_instructions() -> None:
      aliases = {
        "red": "Pick up the red cube and place it in the green tray.",
        "yellow": "Pick up the yellow cube and place it in the green tray.",
        "blue": "Pick up the blue cube and place it in the green tray.",
      }
      while True:
        runtime_policy.wait_until_instruction_ready()
        try:
          command = input("task> ").strip()
        except (EOFError, KeyboardInterrupt):
          return
        if not command:
          continue
        runtime_policy.submit_instruction(aliases.get(command.lower(), command))

    threading.Thread(target=read_instructions, daemon=True).start()
  print(
    f"Camera replanning: every {args.replan_steps} actions "
    f"({args.replan_steps / 20:.2f} s)"
  )
  print(f"D1 arm speed scale: {args.arm_speed_scale:.2f}x")
  if args.async_inference:
    print(
      "Async inference: enabled; simulation remains at 20 Hz while SmolVLA "
      f"prefetches {args.replan_steps}-action chunks."
    )
  if args.compare_instructions:
    print("Comparing instructions from the identical initial observation:")
    for instruction in (
      "Pick up the red cube and place it in the green tray.",
      "Pick up the yellow cube and place it in the green tray.",
      "Pick up the blue cube and place it in the green tray.",
    ):
      runtime_policy.instruction = instruction
      runtime_policy.reset()
      print(f"Task: {instruction}")
      runtime_policy({})
    print("Instruction comparison complete.")
    runtime_policy.close()
    env.close()
    return
  if args.rollout_steps > 0:
    if not args.execute:
      raise ValueError("--rollout-steps requires --execute")
    object_entity = env.scene[selected_object(args.instruction)]
    initial_position = (
      object_entity.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    )
    max_height = float(initial_position[2])
    for _ in range(args.rollout_steps):
      step_started = time.perf_counter()
      action = runtime_policy(observation)
      observation = env.step(action)[0]
      position = object_entity.data.root_link_pos_w[0]
      max_height = max(max_height, float(position[2]))
      if args.async_inference:
        time.sleep(max(0.0, 0.05 - (time.perf_counter() - step_started)))
    final_position = object_entity.data.root_link_pos_w[0].detach().cpu().numpy().copy()
    tray_geom = mujoco.mj_name2id(
      env.sim.mj_model,
      mujoco.mjtObj.mjOBJ_GEOM,
      "target_tray_base",
    )
    if tray_geom < 0:
      raise RuntimeError("Could not find target tray geometry")
    tray_center = env.sim.mj_model.geom_pos[tray_geom, :2]
    tray_xy_error = float(np.linalg.norm(final_position[:2] - tray_center))
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"Object initial position: {initial_position.round(3).tolist()}")
    print(f"Object final position: {final_position.round(3).tolist()}")
    print(f"Object maximum height: {max_height:.3f} m")
    print(f"Final tray XY error: {tray_xy_error:.3f} m")
    print(f"Grasp assist locked: {runtime_policy._assist_local_offset is not None}")
    runtime_policy.close()
    env.close()
    return
  if args.smoke_test:
    action = runtime_policy({})
    print(f"SmolVLA smoke test passed: environment action shape={tuple(action.shape)}")
    runtime_policy.close()
    env.close()
    return
  print("Close the viewer or press Ctrl+C to stop.")
  try:
    scene_module.CameraPanelViewer(env, runtime_policy, frame_rate=20).run()
  finally:
    runtime_policy.close()


if __name__ == "__main__":
  main()
