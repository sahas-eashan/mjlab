param(
  [string]$Instruction = "Pick up the red cube and place it in the green tray.",
  [switch]$Execute,
  [switch]$SmokeTest,
  [int]$ReplanSteps = 10
)

$ErrorActionPreference = "Stop"
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"

$runtimeArguments = @(
  "VLA testing\run_smolvla.py",
  $Instruction,
  "--replan-steps",
  $ReplanSteps
)
if ($Execute) {
  $runtimeArguments += "--execute"
}
if ($SmokeTest) {
  $runtimeArguments += "--smoke-test"
}

uv run --no-sync `
  --index "https://download.pytorch.org/whl/cu126" `
  --index-strategy unsafe-best-match `
  --with "torch==2.9.0+cu126" `
  --with "torchvision==0.24.0+cu126" `
  --with "lerobot[smolvla,scipy-dep]==0.6.0" `
  python @runtimeArguments
