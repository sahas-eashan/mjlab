"""Evaluate SmolVLA actions on recorded observations without rollout drift."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset

VLA_DIR = Path(__file__).parent
RAW_ROOT = VLA_DIR / "data" / "go2_d1_multitask_raw"
DATASET_ROOT = VLA_DIR / "data" / "go2_d1_multitask_lerobot"
DEFAULT_CHECKPOINT = (
  VLA_DIR
  / "outputs"
  / "smolvla_multitask_50k"
  / "checkpoints"
  / "last"
  / "pretrained_model"
)


def _load_runtime() -> Any:
  path = VLA_DIR / "run_smolvla.py"
  spec = importlib.util.spec_from_file_location("vla_teacher_runtime", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load runtime from {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _discover_representative_episodes(require_key: str | None) -> list[int]:
  selected: dict[str, int] = {}
  for episode_dir in sorted(RAW_ROOT.glob("episode_*")):
    with np.load(episode_dir / "trajectory.npz") as data:
      if require_key and require_key not in data:
        continue
      task = str(data["task"].item())
    color = next(color for color in ("red", "yellow", "blue") if color in task)
    selected.setdefault(color, int(episode_dir.name.removeprefix("episode_")))
  missing = {"red", "yellow", "blue"}.difference(selected)
  if missing:
    raise RuntimeError(f"No episodes found for: {sorted(missing)}")
  return [selected[color] for color in ("red", "yellow", "blue")]


def _sample_frames(actions: np.ndarray) -> list[tuple[str, int]]:
  gripper_closed = np.max(np.abs(actions[:, 6:8]), axis=1) < 0.012
  close_candidates = np.flatnonzero(gripper_closed[1:] & ~gripper_closed[:-1]) + 1
  if not len(close_candidates):
    raise RuntimeError("Episode has no open-to-closed gripper transition")
  close = int(close_candidates[0])
  release_candidates = np.flatnonzero(
    ~gripper_closed[close + 1 :] & gripper_closed[close:-1]
  )
  release = (
    close + 1 + int(release_candidates[0])
    if len(release_candidates)
    else len(actions) - 1
  )
  samples = (
    ("start", 0),
    ("approach", max(0, close // 2)),
    ("pre_grasp", max(0, close - 10)),
    ("close", close),
    ("transport", min(len(actions) - 1, (close + release) // 2)),
    ("pre_release", max(close, release - 10)),
    ("release", release),
  )
  unique: dict[int, str] = {}
  for stage, frame in samples:
    unique.setdefault(frame, stage)
  return [(stage, frame) for frame, stage in sorted(unique.items())]


def _batched_observation(sample: dict[str, Any]) -> dict[str, Any]:
  return {
    "observation.state": sample["observation.state"].unsqueeze(0),
    "observation.images.ego": sample["observation.images.ego"].unsqueeze(0),
    "observation.images.wrist": sample["observation.images.wrist"].unsqueeze(0),
    "task": [sample["task"]],
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument("--episodes", type=int, nargs="+")
  parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
  parser.add_argument("--repo-id", default="local/go2_d1_multitask")
  parser.add_argument("--raw-episode-offset", type=int, default=0)
  parser.add_argument("--require-key")
  parser.add_argument(
    "--device",
    choices=("cuda", "cpu"),
    default="cuda" if torch.cuda.is_available() else "cpu",
  )
  args = parser.parse_args()

  runtime = _load_runtime()
  policy, preprocessor, postprocessor = runtime.load_policy(
    args.checkpoint,
    args.device,
    replan_steps=1,
  )
  episode_indices = args.episodes or _discover_representative_episodes(args.require_key)
  all_arm_errors: list[float] = []
  gripper_matches: list[bool] = []

  for episode_index in episode_indices:
    raw_path = RAW_ROOT / f"episode_{episode_index:04d}" / "trajectory.npz"
    with np.load(raw_path) as raw:
      actions = raw["action"].copy()
      task = str(raw["task"].item())
    dataset = LeRobotDataset(
      repo_id=args.repo_id,
      root=args.dataset_root,
      episodes=[episode_index - args.raw_episode_offset],
      video_backend="pyav",
    )
    if len(dataset) != len(actions):
      raise RuntimeError(
        f"Episode {episode_index} frame mismatch: {len(dataset)} vs {len(actions)}"
      )

    print(f"\nEpisode {episode_index:04d}: {task} ({len(dataset)} frames)")
    print("stage          frame  arm_MAE  demo_grip  predicted_grip")
    for stage, frame in _sample_frames(actions):
      sample = dataset[frame]
      demonstrated = sample["action"].cpu().numpy()
      policy.reset()
      torch.manual_seed(1000 + episode_index * 10000 + frame)
      observation = preprocessor(_batched_observation(sample))
      with torch.inference_mode():
        predicted = postprocessor(policy.select_action(observation))
      predicted_np = predicted.reshape(-1).detach().cpu().numpy()[:8]
      arm_mae = float(np.mean(np.abs(predicted_np[:6] - demonstrated[:6])))
      demo_closed = bool(np.max(np.abs(demonstrated[6:8])) < 0.012)
      predicted_closed = bool(np.max(np.abs(predicted_np[6:8])) < 0.012)
      all_arm_errors.append(arm_mae)
      gripper_matches.append(demo_closed == predicted_closed)
      print(
        f"{stage:<14} {frame:>5}  {arm_mae:>7.4f}  "
        f"{'closed' if demo_closed else 'open':>9}  "
        f"{'closed' if predicted_closed else 'open':>14}"
      )

  print("\nTeacher-forced summary")
  print(f"Mean arm action MAE: {np.mean(all_arm_errors):.4f} rad")
  print(
    "Gripper-state accuracy: "
    f"{sum(gripper_matches)}/{len(gripper_matches)} "
    f"({100 * np.mean(gripper_matches):.1f}%)"
  )
  if np.mean(gripper_matches) < 0.8:
    print("Diagnosis: gripper/action sequence is not learned reliably.")
  elif np.mean(all_arm_errors) > 0.12:
    print("Diagnosis: arm actions are inaccurate even on recorded observations.")
  else:
    print("Diagnosis: recorded behavior is learned; rollout drift is the main issue.")


if __name__ == "__main__":
  main()
