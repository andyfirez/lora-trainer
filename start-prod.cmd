@echo off
cd /d "%~dp0"

echo API:      http://127.0.0.1:8000  (docs: /docs)  [production, no --reload]
echo Frontend: http://localhost:3000  (next start)
echo Worker:   polls DB every 5s and spawns training subprocesses
echo.
echo Build UI first if needed: cd frontend ^&^& npm run build
echo.

start "LoRA-API-Prod" cmd /k "cd /d ""%~dp0"" && uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000"
start "LoRA-Worker-Prod" cmd /k "cd /d ""%~dp0"" && uv run python -m scripts.worker"
start "LoRA-Frontend-Prod" cmd /k "cd /d ""%~dp0frontend"" && npm run start"
