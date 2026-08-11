@echo off
cd /d "%~dp0"

echo Building frontend (next build)...
echo.

pushd frontend
call npm run build
set "BUILD_EXIT=%ERRORLEVEL%"
popd

if not "%BUILD_EXIT%"=="0" (
  echo.
  echo Frontend build failed.
  pause
  exit /b %BUILD_EXIT%
)

echo.
echo Frontend build complete. Run start-prod.cmd to start production.
pause
exit /b 0
