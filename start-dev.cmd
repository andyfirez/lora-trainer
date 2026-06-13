@echo off
cd /d "%~dp0"

echo API:      http://127.0.0.1:8000  (docs: /docs)
echo Frontend: http://localhost:3000
echo Worker:   polls DB every 5s and spawns training subprocesses
echo.
echo Copy config.example.toml to config.toml and edit before first run.
echo.

start "LoRA-API" cmd /k "cd /d ""%~dp0"" && uv run uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000"
start "LoRA-Worker" cmd /k "cd /d ""%~dp0"" && uv run python -m scripts.worker"
start "LoRA-Frontend" cmd /k "cd /d ""%~dp0frontend"" && npm run dev"
