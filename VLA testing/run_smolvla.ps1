param(
  [string]$Instruction = "Pick up the red cube and place it in the green tray.",
  [string]$Checkpoint = ".\VLA testing\outputs\smolvla_multitask_50k\checkpoints\last\pretrained_model",
  [switch]$Execute,
  [switch]$GraspAssist,
  [switch]$PresentationAssist,
  [switch]$VlaAssist,
  [switch]$InteractiveInstructions,
  [switch]$CompareInstructions,
  [switch]$SmokeTest,
  [switch]$AsyncInference,
  [int]$RolloutSteps = 0,
  [int]$ReplanSteps = 10,
  [double]$ArmSpeedScale = 1.0
)

$ErrorActionPreference = "Stop"
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"

$runtimeArguments = @(
  "VLA testing\run_smolvla.py",
  $Instruction,
  "--checkpoint",
  $Checkpoint,
  "--replan-steps",
  $ReplanSteps,
  "--arm-speed-scale",
  $ArmSpeedScale
)
if ($Execute) {
  $runtimeArguments += "--execute"
}
if ($GraspAssist) {
  $runtimeArguments += "--grasp-assist"
}
if ($PresentationAssist) {
  $runtimeArguments += "--presentation-assist"
}
if ($VlaAssist) {
  $runtimeArguments += "--vla-assist"
}
if ($InteractiveInstructions) {
  $runtimeArguments += "--interactive-instructions"
}
if ($CompareInstructions) {
  $runtimeArguments += "--compare-instructions"
}
if ($SmokeTest) {
  $runtimeArguments += "--smoke-test"
}
if ($AsyncInference) {
  $runtimeArguments += "--async-inference"
}
if ($RolloutSteps -gt 0) {
  $runtimeArguments += "--rollout-steps"
  $runtimeArguments += $RolloutSteps
}

uv run --no-sync `
  --index "https://download.pytorch.org/whl/cu126" `
  --index-strategy unsafe-best-match `
  --with "torch==2.9.0+cu126" `
  --with "torchvision==0.24.0+cu126" `
  --with "lerobot[smolvla,scipy-dep]==0.6.0" `
  python @runtimeArguments
