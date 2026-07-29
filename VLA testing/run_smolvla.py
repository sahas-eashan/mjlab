"""Run a language-conditioned SmolVLA policy on the Go2+D1 test scene."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy

from mjlab.envs import ManagerBasedRlEnv

VLA_DIR = Path(__file__).parent
DEFAULT_CHECKPOINT = (
  VLA_DIR / "outputs" / "smolvla_full_20k" / "checkpoints" / "last" / "pretrained_model"
)
ARM_SLICE = slice(12, 20)
ARM_MAX_STEP = torch.tensor((0.04,) * 6 + (0.004, 0.004))


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
  return parser.parse_args()


def load_scene_module() -> Any:
  path = VLA_DIR / "start_custom_scene.py"
  spec = importlib.util.spec_from_file_location("vla_runtime_scene", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load scene from {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


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
  ):
    self.env = env
    self.policy = policy
    self.preprocessor = preprocessor
    self.postprocessor = postprocessor
    self.instruction = instruction
    self.execute = execute
    self.robot = env.scene["robot"]
    self.action_term = env.action_manager._terms["joint_position"]
    self.step_count = 0
    self._warned_stability = False

  def reset(self) -> None:
    self.policy.reset()
    self.step_count = 0

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
    max_step = ARM_MAX_STEP.to(self.env.device).unsqueeze(0)
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

  def __call__(self, _observation: object) -> torch.Tensor:
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

    # Zero leg actions map to the configured standing-pose offsets.
    return self._to_environment_action(arm_target)


def main() -> None:
  args = parse_args()
  scene_module = load_scene_module()
  config = scene_module.make_env_cfg()
  config.decimation = 10  # 0.005 s simulation step -> 20 Hz VLA control.
  env = ManagerBasedRlEnv(config, device=args.device)
  env.reset()

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
  )
  mode = "EXECUTION ENABLED" if args.execute else "DRY RUN (arm held)"
  print(f"Instruction: {args.instruction}")
  print(f"Mode: {mode}")
  print(
    f"Camera replanning: every {args.replan_steps} actions "
    f"({args.replan_steps / 20:.2f} s)"
  )
  if args.smoke_test:
    action = runtime_policy({})
    print(f"SmolVLA smoke test passed: environment action shape={tuple(action.shape)}")
    env.close()
    return
  print("Close the viewer or press Ctrl+C to stop.")
  scene_module.CameraPanelViewer(env, runtime_policy, frame_rate=20).run()


if __name__ == "__main__":
  main()
