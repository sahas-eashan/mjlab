$env:UV_PROJECT_ENVIRONMENT = ".venv-windows"
uv run --no-sync python ".\VLA testing\teleop_collect.py" @args
