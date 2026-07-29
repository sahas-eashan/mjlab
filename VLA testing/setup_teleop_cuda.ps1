uv pip install `
  --python ".venv-windows\Scripts\python.exe" `
  --index "https://download.pytorch.org/whl/cu126" `
  --index-strategy unsafe-best-match `
  "torch==2.9.0+cu126" `
  "torchvision==0.24.0+cu126"
