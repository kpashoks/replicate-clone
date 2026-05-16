@echo off
REM Start both dev servers in separate windows.
REM Backend on http://localhost:8000, frontend on http://localhost:3000.
REM
REM Run from the repo root: just double-click this file, or `start-dev.bat`
REM from cmd. Each server gets its own titled window so you can read logs
REM and Ctrl+C to stop independently.

setlocal
set "REPO=%~dp0"

REM Make sure the venv exists before we try to use it.
if not exist "%REPO%backend\.venv\Scripts\activate.bat" (
  echo [start-dev] ERROR: backend\.venv not found.
  echo            Did you run the one-time setup? See README step 2.
  pause
  exit /b 1
)

if not exist "%REPO%frontend\node_modules" (
  echo [start-dev] ERROR: frontend\node_modules not found.
  echo            Run 'cd frontend ^&^& npm install' first ^(README step 3^).
  pause
  exit /b 1
)

echo [start-dev] Starting backend on port 8000...
start "replicate-local backend (8000)" cmd /k "cd /d %REPO%backend && .venv\Scripts\activate && uvicorn main:app --port 8000 --reload"

REM Small delay so the two windows don't fight for focus.
timeout /t 2 /nobreak >nul

echo [start-dev] Starting frontend on port 3000...
start "replicate-local frontend (3000)" cmd /k "cd /d %REPO%frontend && npm run dev"

echo.
echo [start-dev] Both servers launching in separate windows.
echo            Backend:  http://localhost:8000/api/health
echo            Frontend: http://localhost:3000
echo            Ctrl+C inside each window to stop that server.
echo.
endlocal
