param(
  [int]$Steps = 50000,
  [int]$SaveFreq = 5000
)

$ErrorActionPreference = "Stop"
$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"

uv run --no-project `
  --index "https://download.pytorch.org/whl/cu126" `
  --index-strategy unsafe-best-match `
  --with "torch==2.9.0+cu126" `
  --with "torchvision==0.24.0+cu126" `
  --with "lerobot[training,smolvla,scipy-dep]==0.6.0" `
  lerobot-train `
  --policy.path=".\VLA testing\models\smolvla_base" `
  --policy.push_to_hub=false `
  --dataset.repo_id=local/go2_d1_multitask_randomized `
  --dataset.root=".\VLA testing\data\go2_d1_multitask_randomized_lerobot" `
  --dataset.video_backend=pyav `
  --batch_size=1 `
  --num_workers=0 `
  --persistent_workers=false `
  --steps=$Steps `
  --save_freq=$SaveFreq `
  --log_freq=20 `
  --output_dir=".\VLA testing\outputs\smolvla_randomized_50k" `
  --job_name=go2_d1_smolvla_randomized_50k `
  --policy.device=cuda `
  --wandb.enable=false
