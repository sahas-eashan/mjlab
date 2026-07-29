# VLA testing

This folder contains isolated experiments for connecting open-source robot
environments and VLA policies to mjlab.

## Current work

- Added a source snapshot of Meta-World under `environments/metaworld`.
- Pinned the snapshot to commit
  `f571cd00d85af4dc1264a31ed85407ec23495d89`.
- Added a continuously rendered launcher for Meta-World's `pick-place-v3` task.
- Added a custom mjlab scene with the Go2+D1 robot, a low work table, three
  movable objects, a target tray, and navigation obstacles.
- Added robot-mounted ego and wrist cameras. Both produce 256x256 RGB and depth;
  SmolVLA will consume RGB, while depth is retained for expert demonstrations,
  debugging, and future perception work.
- The custom native viewer shows both live RGB camera feeds on the right side of
  the simulation window.
- Added a scripted pick-and-place demonstration collector. It randomizes the red
  cube, records synchronized ego/wrist RGB, D1 state, and D1 actions, and rejects
  unsuccessful episodes.
- The first task keeps the Go2 base fixed and uses a temporary grasp assist while
  the D1 contact model is refined.
- Added an experimental BridgeData V2 retargeter for real WidowX Cartesian
  demonstrations. It maps a real pick-and-place path to D1 pose-IK waypoints,
  runs it through normal MuJoCo contacts, and marks failed physical replays as
  unusable for training. No object attachment or in-episode teleport is used.
- Corrected the D1 finger collider axes and added contact diagnostics. The first
  BridgeData replay is still being calibrated and has not produced an accepted
  episode yet.
- Added keyboard Cartesian teleoperation for collecting successful,
  language-labelled demonstrations of the red cube, yellow block, and blue
  cylinder without scripted object attachment or teleporting.

Meta-World is maintained by the Farama Foundation and its downloaded source
retains the upstream MIT license.

## Start

From the mjlab repository root, run:

```powershell
uv run --no-project --with-editable ".\VLA testing\environments\metaworld" python ".\VLA testing\start_metaworld.py"
```

Meta-World does not officially support Windows. If the viewer fails on Windows,
run the same command in Linux or WSL with GPU/GUI support.

To start the custom Go2+D1 scene from the repository root:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
uv run python ".\VLA testing\start_custom_scene.py"
```

To collect five local demonstration episodes:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
uv run --no-sync python ".\VLA testing\collect_demos.py" --episodes 5
```

Episodes are saved under `VLA testing/data/` and are intentionally ignored by
Git. Camera rendering is currently slow on Windows, so a five-episode collection
can take a while. Add `--no-video` when validating only the controller.

To collect demonstrations manually with the D1 arm:

```powershell
& ".\VLA testing\setup_teleop_cuda.ps1"
& ".\VLA testing\teleop_collect.ps1"
```

Use `I/K` for forward/back, `J/L` for left/right, `U/O` for up/down, and `G`
to open or close the gripper. Select the instruction with `1` (red cube), `2`
(yellow block), or `3` (blue cylinder). Press `C` to start a recording, `V` to
save a successful demonstration, or `X` to discard it. Press `Enter` to reset
the scene between attempts. Record many varied successes for every instruction;
failed or ambiguous attempts should be discarded.

Run the CUDA setup once. The project environment otherwise uses CPU-only
PyTorch, which makes the two-camera simulation respond far below real time.
The teleoperator captures RGB only because SmolVLA does not consume depth,
and uses faster Cartesian tracking than the scripted demonstration controller.

The teleoperator records both camera views, the eight D1 joint values and
commands, the instruction, and finger actuator effort. It never teleports or
attaches an object. Raw episodes are written to
`VLA testing/data/go2_d1_multitask_raw`.

Convert those demonstrations with:

```powershell
uv run --no-project --with "lerobot[dataset]==0.6.0" --with "imageio[ffmpeg]" python ".\VLA testing\convert_to_lerobot.py" --input ".\VLA testing\data\go2_d1_multitask_raw" --output ".\VLA testing\data\go2_d1_multitask_lerobot" --repo-id "local/go2_d1_multitask"
```

To test the locally downloaded BridgeData V2 episode through physical contacts:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
uv run --no-sync python ".\VLA testing\retarget_bridgedata.py" --episode-index 4 --no-video
```

The command exits with an error when the cube is not physically placed. Such an
episode is retained only for debugging and must not be converted or trained on.

## Convert to LeRobot v3

Convert successful raw episodes into a local LeRobot v3 dataset:

```powershell
uv run --no-project --with "lerobot[dataset]==0.6.0" --with "imageio[ffmpeg]" python ".\VLA testing\convert_to_lerobot.py"
```

This writes `VLA testing/data/go2_d1_pick_place_lerobot`. It stores both RGB
views, the eight-value D1 state, the eight-value action, and the language task.
The command is local only and does not upload a dataset.

Use `--input`, `--output`, and `--repo-id` to choose other locations. Pass
`--overwrite` only when replacing an existing converted dataset.

For a first SmolVLA smoke test after collecting enough demonstrations:

```powershell
uv run --no-project --with "huggingface-hub" hf download lerobot/smolvla_base --local-dir ".\VLA testing\models\smolvla_base"

uv run python ".\VLA testing\adapt_smolvla_config.py"

uv run --no-project --index "https://download.pytorch.org/whl/cu126" --index-strategy unsafe-best-match --with "torch==2.9.0+cu126" --with "torchvision==0.24.0+cu126" --with "lerobot[training,smolvla,scipy-dep]==0.6.0" lerobot-train --policy.path=".\VLA testing\models\smolvla_base" --policy.push_to_hub=false --dataset.repo_id=local/go2_d1_pick_place --dataset.root=".\VLA testing\data\go2_d1_pick_place_lerobot" --dataset.video_backend=pyav --batch_size=1 --num_workers=0 --persistent_workers=false --steps=10 --output_dir=".\VLA testing\outputs\smolvla_smoke" --job_name=go2_d1_smolvla_smoke --policy.device=cuda --wandb.enable=false
```

Ten steps only verify that the training pipeline starts. A useful policy needs
substantially more varied successful episodes; SmolVLA recommends about 50 as a
starting point.

## Run SmolVLA from a language instruction

Run one safe smoke-test inference using the local checkpoint:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
uv run --no-sync --index "https://download.pytorch.org/whl/cu126" --index-strategy unsafe-best-match --with "torch==2.9.0+cu126" --with "torchvision==0.24.0+cu126" --with "lerobot[smolvla,scipy-dep]==0.6.0" python ".\VLA testing\run_smolvla.py" "Pick up the red cube and place it in the green tray." --smoke-test
```

Omit `--smoke-test` to open the scene in dry-run mode. It displays both camera
views and prints SmolVLA predictions while holding the arm:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
uv run --no-sync --index "https://download.pytorch.org/whl/cu126" --index-strategy unsafe-best-match --with "torch==2.9.0+cu126" --with "torchvision==0.24.0+cu126" --with "lerobot[smolvla,scipy-dep]==0.6.0" python ".\VLA testing\run_smolvla.py" "Pick up the red cube and place it in the green tray."
```

Add `--execute` only after training a useful checkpoint. Execution keeps Go2's
legs at their standing targets, limits each D1 action step, clamps joint limits,
and holds the arm if the base height or tilt crosses the safety threshold. The
runner now loads `outputs/smolvla_full_20k/checkpoints/last/pretrained_model`
by default. Use `--checkpoint` to evaluate another saved policy.

The shorter PowerShell launcher runs the same command:

```powershell
& ".\VLA testing\run_smolvla.ps1"
```

After checking the dry-run predictions and scene, enable D1 movement with:

```powershell
& ".\VLA testing\run_smolvla.ps1" -Execute
```

The SmolVLA runtime uses a floating Go2 base with normal gravity and foot
contacts. Payload-specific leg position gains support the D1 arm while SmolVLA
controls only the eight arm and gripper joints.

The runtime executes ten actions (0.5 seconds at 20 Hz) from each predicted
action chunk before processing fresh camera frames. Change this receding-horizon
interval with `-ReplanSteps`; lower values react more frequently but require
more GPU inference.

## Full local fine-tuning

Start the first 20,000-step local fine-tune with:

```powershell
& ".\VLA testing\train_smolvla_full.ps1"
```

Checkpoints are saved every 2,000 steps under
`VLA testing/outputs/smolvla_full_20k`. The current dataset has five scripted
episodes for one instruction, so this run is useful for validating the complete
training/inference loop but is not enough for a robust manipulation policy.

The initially downloaded `smolvla_base` files are pretrained foundation weights.
Fine-tuning reads them once and writes adapted weights plus the normalization and
tokenization processors into each checkpoint. Hugging Face may also populate its
local cache with SmolVLM configuration/tokenizer files. Cached files are reused;
they are not downloaded again for every run.
