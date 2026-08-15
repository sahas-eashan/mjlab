"""Convert raw Go2+D1 demonstrations into a local LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterator
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

CAMERA_FILES = {
  "observation.images.ego": "ego_camera.mp4",
  "observation.images.wrist": "wrist_camera.mp4",
}
STATE_NAMES = (
  "arm_joint_1",
  "arm_joint_2",
  "arm_joint_3",
  "arm_joint_4",
  "arm_joint_5",
  "arm_joint_6",
  "gripper",
  "gripper_mirror",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input",
    type=Path,
    default=Path(__file__).parent / "data" / "go2_d1_pick_place_raw",
    help="Folder containing raw episode_* directories.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).parent / "data" / "go2_d1_pick_place_lerobot",
    help="Destination for the local LeRobot v3 dataset.",
  )
  parser.add_argument(
    "--repo-id",
    default="local/go2_d1_pick_place",
    help="Dataset identifier stored in LeRobot metadata (nothing is uploaded).",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Replace an existing output dataset.",
  )
  parser.add_argument(
    "--require-key",
    help=(
      "Convert only episodes whose trajectory.npz contains this metadata key "
      "(for example, layout_xy for randomized automatic demonstrations)."
    ),
  )
  return parser.parse_args()


def discover_episodes(input_dir: Path) -> list[Path]:
  episodes = sorted(
    path
    for path in input_dir.glob("episode_*")
    if path.is_dir() and (path / "trajectory.npz").is_file()
  )
  if not episodes:
    raise FileNotFoundError(f"No raw episodes found under {input_dir}")
  return episodes


def read_video(path: Path) -> Iterator[np.ndarray]:
  reader = imageio.get_reader(path)
  try:
    for frame in reader:
      yield np.asarray(frame, dtype=np.uint8)
  finally:
    reader.close()


def validate_episode(
  episode_dir: Path,
) -> tuple[dict[str, np.ndarray], int, str]:
  missing = [
    filename
    for filename in (*CAMERA_FILES.values(), "trajectory.npz")
    if not (episode_dir / filename).is_file()
  ]
  if missing:
    raise FileNotFoundError(f"{episode_dir.name} is missing: {', '.join(missing)}")

  with np.load(episode_dir / "trajectory.npz") as loaded:
    trajectory = {key: loaded[key].copy() for key in loaded.files}

  required = {"observation_state", "action", "task", "success", "fps"}
  if missing_keys := required.difference(trajectory):
    raise ValueError(f"{episode_dir.name} is missing arrays: {sorted(missing_keys)}")
  if not bool(trajectory["success"]):
    raise ValueError(f"{episode_dir.name} is not marked successful")

  states = trajectory["observation_state"]
  actions = trajectory["action"]
  if states.ndim != 2 or actions.ndim != 2:
    raise ValueError(f"{episode_dir.name} state and action must be 2D")
  if states.shape != actions.shape or states.shape[1] != len(STATE_NAMES):
    raise ValueError(
      f"{episode_dir.name} expected matching (frames, 8) state/action, got "
      f"{states.shape} and {actions.shape}"
    )

  fps = int(trajectory["fps"])
  task = str(trajectory["task"].item()).strip()
  if fps <= 0 or not task:
    raise ValueError(f"{episode_dir.name} has invalid FPS or task")
  return trajectory, fps, task


def convert(args: argparse.Namespace) -> None:
  try:
    from lerobot.datasets import LeRobotDataset
  except ImportError as error:
    raise RuntimeError(
      "LeRobot is not installed. Run this script with the README's isolated "
      "`uv run --no-project --with` command."
    ) from error

  input_dir = args.input.resolve()
  output_dir = args.output.resolve()
  episodes = discover_episodes(input_dir)
  validated = [(path, *validate_episode(path)) for path in episodes]
  if args.require_key:
    validated = [item for item in validated if args.require_key in item[1]]
    if not validated:
      raise ValueError(f"No episodes contain required key {args.require_key!r}")
    print(
      f"Selected {len(validated)}/{len(episodes)} episodes containing "
      f"{args.require_key!r}"
    )

  fps_values = {fps for _, _, fps, _ in validated}
  if len(fps_values) != 1:
    raise ValueError(f"All episodes must use one FPS; found {sorted(fps_values)}")
  fps = fps_values.pop()

  if output_dir.exists():
    if not args.overwrite:
      raise FileExistsError(
        f"{output_dir} already exists; pass --overwrite to replace it"
      )
    shutil.rmtree(output_dir)

  features = {
    camera_key: {
      "dtype": "video",
      "shape": (256, 256, 3),
      "names": ["height", "width", "channels"],
    }
    for camera_key in CAMERA_FILES
  }
  features["observation.state"] = {
    "dtype": "float32",
    "shape": (len(STATE_NAMES),),
    "names": list(STATE_NAMES),
  }
  features["action"] = {
    "dtype": "float32",
    "shape": (len(STATE_NAMES),),
    "names": list(STATE_NAMES),
  }

  dataset = LeRobotDataset.create(
    repo_id=args.repo_id,
    root=output_dir,
    fps=fps,
    robot_type="unitree_go2_d1_sim",
    features=features,
    use_videos=True,
  )

  try:
    for episode_dir, trajectory, _, task in validated:
      video_streams = {
        key: read_video(episode_dir / filename)
        for key, filename in CAMERA_FILES.items()
      }
      frame_count = trajectory["action"].shape[0]
      for frame_index in range(frame_count):
        frame = {
          "observation.state": trajectory["observation_state"][frame_index].astype(
            np.float32
          ),
          "action": trajectory["action"][frame_index].astype(np.float32),
          "task": task,
        }
        for camera_key, stream in video_streams.items():
          try:
            image = next(stream)
          except StopIteration as error:
            raise ValueError(
              f"{episode_dir.name}/{CAMERA_FILES[camera_key]} ended before "
              f"trajectory frame {frame_index}"
            ) from error
          if image.shape != (256, 256, 3):
            raise ValueError(
              f"{episode_dir.name}/{CAMERA_FILES[camera_key]} frame "
              f"{frame_index} has shape {image.shape}"
            )
          frame[camera_key] = image

        dataset.add_frame(frame)

      for camera_key, stream in video_streams.items():
        try:
          next(stream)
        except StopIteration:
          continue
        raise ValueError(
          f"{episode_dir.name}/{CAMERA_FILES[camera_key]} has more than "
          f"{frame_count} frames"
        )

      dataset.save_episode()
      print(f"Converted {episode_dir.name}: {frame_count} frames")
  finally:
    dataset.finalize()

  print(f"LeRobot v3 dataset written to {output_dir}")
  print(f"Local repo id: {args.repo_id}")


def main() -> None:
  convert(parse_args())


if __name__ == "__main__":
  main()
