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
Object-filtered contact sensors independently monitor both D1 fingers. When
both fingers contact the selected object, a small capped over-close command
maintains grip pressure; the existing actuator effort limit remains the hard
safety bound.

For presentation video only, lock a detected object to the closed gripper:

```powershell
& ".\VLA testing\teleop_collect.ps1" --visual-grasp-assist
```

Assisted episodes can be saved and include `grasp_assist=True` in their raw
metadata. They are intended for the matching assisted SmolVLA runtime, not for
evaluating unassisted physical grasping or transferring directly to hardware.

The teleoperator records both camera views, the eight D1 joint values and
commands, the instruction, and finger actuator effort. It never teleports or
attaches an object. Raw episodes are written to
`VLA testing/data/go2_d1_multitask_raw`.

Generate additional balanced demonstrations automatically with randomized,
camera-visible object layouts:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
uv run --no-sync python ".\VLA testing\auto_collect_multitask.py" --episodes-per-task 30
```

The automatic collector alternates red, yellow, and blue tasks, independently
randomizes color-to-position assignments, rejects layouts where any cube has
too few ego-camera pixels, requires target visibility in both camera streams,
and saves only successful tray placements. It uses the same disclosed
contact-triggered visual grasp retention as the assisted runtime.

Create a clean LeRobot dataset containing only these randomized episodes:

```powershell
uv run --no-project --with "lerobot[dataset]==0.6.0" --with "imageio[ffmpeg]" python ".\VLA testing\convert_to_lerobot.py" --input ".\VLA testing\data\go2_d1_multitask_raw" --output ".\VLA testing\data\go2_d1_multitask_randomized_lerobot" --repo-id "local/go2_d1_multitask_randomized" --require-key layout_xy
```

Fine-tune a fresh SmolVLA base policy on the randomized subset:

```powershell
& ".\VLA testing\train_smolvla_randomized.ps1"
```

This intentionally starts from the local SmolVLA base weights instead of the
red-biased 100K checkpoint. It trains for 50,000 steps and saves every 5,000
steps under `outputs/smolvla_randomized_50k`.

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
runner now loads the multitask checkpoint at
`outputs/smolvla_multitask_50k/checkpoints/last/pretrained_model` by default.
Use `--checkpoint` to evaluate another saved policy.

The shorter PowerShell launcher runs the same command:

```powershell
& ".\VLA testing\run_smolvla.ps1"
```

After checking the dry-run predictions and scene, enable D1 movement with:

```powershell
& ".\VLA testing\run_smolvla.ps1" -Execute
```

Run a policy trained on assisted demonstrations with the matching low-level
grasp follower:

```powershell
& ".\VLA testing\run_smolvla.ps1" -Execute -GraspAssist -Instruction "Pick up the yellow block and place it in the green tray."
```

The instruction selects the red cube, yellow block, or blue cylinder contact
sensor. The follower activates only after a closed gripper contacts that object
and releases when SmolVLA commands the gripper open.

For a reliable presentation rollout, enable the disclosed hybrid supervisor:

```powershell
& ".\VLA testing\run_smolvla.ps1" -Execute -PresentationAssist -Instruction "Pick up the blue cube and place it in the green tray."
```

SmolVLA still processes both live cameras and the language instruction. A
deterministic position-only Cartesian supervisor routes the arm to the object
named by the instruction, while the contact follower retains it and the
supervisor places it in the tray. This mode is assisted VLA behavior and must
not be presented as end-to-end autonomous SmolVLA control.

Load the model once and enter presentation tasks continuously from the terminal:

```powershell
& ".\VLA testing\run_smolvla.ps1" -Execute -PresentationAssist -InteractiveInstructions
```

After the viewer opens, type `red`, `yellow`, `blue`, or a complete instruction
at the `task>` prompt. The viewer remains open and each new instruction resets
the task supervisor without reloading the model or resetting the scene, so a
single continuous screen recording can contain several commands. The prompt
returns only after the current placement completes. Pressing Enter in the
viewer resets both the scene and interactive console; while waiting, the D1 is
actively commanded to a fixed neutral pose with its gripper open.

To evaluate SmolVLA with substantially smaller low-level patches, use:

```powershell
& ".\VLA testing\run_smolvla.ps1" -Execute -VlaAssist -Instruction "Pick up the blue cube and place it in the green tray."
```

In this mode SmolVLA controls the broad arm trajectory and must independently
enter a 20 cm neighborhood of the requested object and transport it near the
tray. Position-only IK then contributes a bounded local approach, keeps the
fingers open until the hand enters the final grasp zone, and closes them before
contact-triggered retention takes over. The retention latch rejects premature
open predictions during transport. After the held object remains low and within
10 cm of the tray centre for eight control frames, the local controller opens
the fingers once and prevents immediate re-grasping. A 25% correction remains
available within 12 cm of the tray. This is hybrid VLA behavior—not end-to-end autonomous
grasping—and can still fail if the learned policy selects the wrong object or
never enters the local grasp neighborhood.

The SmolVLA runtime uses a floating Go2 base with normal gravity and foot
contacts. Payload-specific leg position gains support the D1 arm while SmolVLA
controls only the eight arm and gripper joints.

The runtime executes ten actions (0.5 seconds at 20 Hz) from each predicted
action chunk before processing fresh camera frames. Change this receding-horizon
interval with `-ReplanSteps`; lower values react more frequently but require
more GPU inference.

For visually closed-loop evaluation, use `-ReplanSteps 1` so every control step
uses a fresh camera-conditioned prediction. Use `-ArmSpeedScale 0.35` to reduce
the D1 per-step joint limit to 35% without changing the policy outputs.

For smooth real-time viewing, run SmolVLA action-chunk generation in a background
worker. This is the recommended command for the balanced randomized checkpoint:

```powershell
& ".\VLA testing\run_smolvla.ps1" -Checkpoint ".\VLA testing\outputs\smolvla_randomized_50k\checkpoints\050000\pretrained_model" -Execute -VlaAssist -AsyncInference -ReplanSteps 50 -Instruction "Pick up the blue cube and place it in the green tray."
```

MuJoCo and both camera panels continue at 20 Hz while the model predicts the
next 50-action chunk from the latest available ego, wrist, state, and language
observation. The worker starts prefetching before the current chunk is empty. If
the GPU is still late, the controller safely holds its last target instead of
freezing the viewer or applying a burst of actions. This fixes execution timing;
it does not conceal autonomous selection or trajectory failures.

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
