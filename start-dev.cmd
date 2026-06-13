@echo off
cd /d "%~dp0"

echo API:      http://127.0.0.1:8000  (docs: /docs)
echo Frontend: http://localhost:3000
echo Worker:   embedded in API process (polls DB every 5s)
echo.
echo Copy config.example.toml to config.toml and edit before first run.
echo Avoid editing API code during a running job when using --reload.
echo.

start "LoRA-API" cmd /k "cd /d ""%~dp0"" && uv run uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000"
start "LoRA-Frontend" cmd /k "cd /d ""%~dp0frontend"" && npm run dev"
