"""Adapt a downloaded SmolVLA checkpoint to the Go2+D1 dataset schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INPUT_FEATURES = {
  "observation.state": {"type": "STATE", "shape": [8]},
  "observation.images.ego": {"type": "VISUAL", "shape": [3, 256, 256]},
  "observation.images.wrist": {"type": "VISUAL", "shape": [3, 256, 256]},
}
OUTPUT_FEATURES = {
  "action": {"type": "ACTION", "shape": [8]},
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--model-dir",
    type=Path,
    default=Path(__file__).parent / "models" / "smolvla_base",
    help="Downloaded local SmolVLA checkpoint directory.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  config_path = args.model_dir.resolve() / "config.json"
  if not config_path.is_file():
    raise FileNotFoundError(f"SmolVLA config not found: {config_path}")

  config = json.loads(config_path.read_text(encoding="utf-8"))
  if config.get("type") != "smolvla":
    raise ValueError(f"{config_path} is not a SmolVLA config")
  if config.get("max_state_dim", 0) < 8 or config.get("max_action_dim", 0) < 8:
    raise ValueError("Checkpoint cannot accommodate eight-dimensional D1 controls")

  config["input_features"] = INPUT_FEATURES
  config["output_features"] = OUTPUT_FEATURES
  config["push_to_hub"] = False
  config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

  print(f"Adapted {config_path}")
  print("Inputs: ego RGB, wrist RGB, and 8-value D1 state")
  print("Output: 8-value D1 arm/gripper action")


if __name__ == "__main__":
  main()
