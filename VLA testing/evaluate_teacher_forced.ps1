param(
  [int[]]$Episodes
)

$ErrorActionPreference = "Stop"
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"

$runtimeArguments = @("VLA testing\evaluate_teacher_forced.py")
if ($Episodes) {
  $runtimeArguments += "--episodes"
  $runtimeArguments += $Episodes
}

uv run --no-sync `
  --index "https://download.pytorch.org/whl/cu126" `
  --index-strategy unsafe-best-match `
  --with "torch==2.9.0+cu126" `
  --with "torchvision==0.24.0+cu126" `
  --with "lerobot[smolvla,scipy-dep]==0.6.0" `
  python @runtimeArguments
