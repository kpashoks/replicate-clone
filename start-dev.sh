#!/usr/bin/env bash
# Start both dev servers (macOS / Linux / Git Bash on Windows).
#
# Backend on http://localhost:8000, frontend on http://localhost:3000.
# Press Ctrl+C in this terminal to stop both.
#
# Run from the repo root:
#   bash start-dev.sh
#   ./start-dev.sh   (if executable)

set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$REPO/backend/.venv/bin/activate" ] && \
   [ ! -f "$REPO/backend/.venv/Scripts/activate" ]; then
  echo "[start-dev] ERROR: backend/.venv not found."
  echo "            Did you run the one-time setup? See README step 2."
  exit 1
fi

if [ ! -d "$REPO/frontend/node_modules" ]; then
  echo "[start-dev] ERROR: frontend/node_modules not found."
  echo "            Run 'cd frontend && npm install' first (README step 3)."
  exit 1
fi

# Stop both children on Ctrl+C.
cleanup() {
  echo
  echo "[start-dev] Stopping..."
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
  wait 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

echo "[start-dev] Starting backend on port 8000..."
(
  cd "$REPO/backend"
  if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  else
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
  fi
  exec uvicorn main:app --port 8000 --reload
) &
BACKEND_PID=$!

sleep 2

echo "[start-dev] Starting frontend on port 3000..."
(
  cd "$REPO/frontend"
  exec npm run dev
) &
FRONTEND_PID=$!

echo
echo "[start-dev] Both servers running:"
echo "            Backend:  http://localhost:8000/api/health  (PID $BACKEND_PID)"
echo "            Frontend: http://localhost:3000             (PID $FRONTEND_PID)"
echo "            Ctrl+C here to stop both."
echo

wait
