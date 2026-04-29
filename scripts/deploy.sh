#!/bin/bash
# ============================================
# AI Hologram Medical - Complete Deployment Script
# Supports: Windows (Git Bash/WSL), Linux, macOS
# ============================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
DATA_DIR="$PROJECT_DIR/data"
WEIGHTS_DIR="$PROJECT_DIR/models/weights"

echo "
╔══════════════════════════════════════════════════════╗
║  AI-Driven Hologram Medical Visualization System     ║
║  Deployment Script v2.0 (Windows + Linux/Mac)        ║
╚══════════════════════════════════════════════════════╝
"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
err()  { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "  ► $*"; }

# ─────────────────────────────────────────────
# 1. Detect correct Python command
# ─────────────────────────────────────────────
detect_python() {
    PYTHON=""
    PIP=""

    for cmd in python python3 python3.13 python3.12 python3.11 python3.10; do
        if command -v "$cmd" &>/dev/null; then
            VER=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            MAJ=$(echo "$VER" | cut -d. -f1)
            MIN=$(echo "$VER" | cut -d. -f2)
            if [ "$MAJ" -ge 3 ] && [ "$MIN" -ge 10 ]; then
                PYTHON="$cmd"
                # Find matching pip
                PIP_CMD="${cmd/python/pip}"
                if command -v "$PIP_CMD" &>/dev/null; then
                    PIP="$PIP_CMD"
                else
                    PIP="$cmd -m pip"
                fi
                log "Found Python $VER → '$PYTHON'"
                return 0
            fi
        fi
    done

    # Also try Windows py launcher
    if command -v py &>/dev/null; then
        VER=$(py -3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        MAJ=$(echo "$VER" | cut -d. -f1)
        MIN=$(echo "$VER" | cut -d. -f2)
        if [ "$MAJ" -ge 3 ] && [ "$MIN" -ge 10 ]; then
            PYTHON="py -3"
            PIP="py -3 -m pip"
            log "Found Python via Windows launcher: $VER"
            return 0
        fi
    fi

    echo ""
    echo "  ✗ Python 3.10+ not found in PATH."
    echo ""
    echo "  Your system Python: $(python --version 2>/dev/null || echo 'not found')"
    echo ""
    echo "  To fix on Windows:"
    echo "    1. Download Python 3.13 from https://www.python.org/downloads/"
    echo "    2. During install: ✔ CHECK 'Add Python to PATH'"
    echo "    3. Restart Git Bash and try again."
    echo ""
    exit 1
}

# ─────────────────────────────────────────────
# 2. Detect OS → venv activation path
# ─────────────────────────────────────────────
detect_venv_paths() {
    VENV_DIR="$PROJECT_DIR/venv"

    if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "win32" || \
          "${OSTYPE:-}" == "cygwin" || -n "${WINDIR:-}" ]]; then
        IS_WINDOWS=true
        VENV_ACTIVATE="$VENV_DIR/Scripts/activate"
        VENV_PYTHON="$VENV_DIR/Scripts/python"
        VENV_PIP="$VENV_DIR/Scripts/pip"
    else
        IS_WINDOWS=false
        VENV_ACTIVATE="$VENV_DIR/bin/activate"
        VENV_PYTHON="$VENV_DIR/bin/python"
        VENV_PIP="$VENV_DIR/bin/pip"
    fi
}

# ─────────────────────────────────────────────
# Install
# ─────────────────────────────────────────────
setup_venv() {
    log "Creating virtual environment at: $VENV_DIR"
    if [ ! -d "$VENV_DIR" ]; then
        $PYTHON -m venv "$VENV_DIR"
        log "Virtual environment created."
    else
        log "Virtual environment already exists — skipping creation."
    fi

    source "$VENV_ACTIVATE"
    python -m pip install --upgrade pip setuptools wheel --quiet
    log "Installing Python dependencies..."
    pip install -r "$PROJECT_DIR/requirements.txt" --quiet
    log "All dependencies installed."
}

setup_directories() {
    log "Creating project directories..."
    mkdir -p "$LOG_DIR" "$WEIGHTS_DIR"
    mkdir -p "$DATA_DIR/uploads" "$DATA_DIR/outputs"
    mkdir -p "$DATA_DIR/train/images" "$DATA_DIR/train/masks"
    log "Directories ready."
}

setup_frontend() {
    log "Setting up frontend..."
    if command -v npm &>/dev/null; then
        cd "$PROJECT_DIR/frontend"
        npm install --silent
        npm run build
        log "Frontend built."
        cd "$PROJECT_DIR"
    else
        log "npm not found — skipping frontend build (optional)."
    fi
}

db_init() {
    log "Initializing database..."
    source "$VENV_ACTIVATE"
    python -c "
import asyncio, sys
sys.path.insert(0, '.')
from backend.database import Database
async def init():
    db = Database()
    await db.connect()
    print('  ✓ Database initialized.')
    await db.disconnect()
asyncio.run(init())
"
}

# ─────────────────────────────────────────────
# Start / Stop
# ─────────────────────────────────────────────
start_backend() {
    log "Starting FastAPI backend..."
    source "$VENV_ACTIVATE"
    cd "$PROJECT_DIR"
    nohup python -m uvicorn backend.main:app \
        --host 0.0.0.0 --port 8000 \
        --workers 1 --log-level info \
        > "$LOG_DIR/backend.log" 2>&1 &
    echo $! > "$LOG_DIR/backend.pid"
    log "Backend PID: $(cat "$LOG_DIR/backend.pid")"
    info "API docs: http://localhost:8000/docs"
    info "Logs:     $LOG_DIR/backend.log"
}

stop_backend() {
    if [ -f "$LOG_DIR/backend.pid" ]; then
        PID=$(cat "$LOG_DIR/backend.pid")
        kill "$PID" 2>/dev/null && log "Stopped (PID $PID)." || log "Not running."
        rm -f "$LOG_DIR/backend.pid"
    else
        log "No PID file found."
    fi
}

check_health() {
    sleep 3
    if command -v curl &>/dev/null; then
        curl -sf http://localhost:8000/health > /dev/null 2>&1 \
            && log "✓ Backend healthy." \
            || log "⚠ Backend health check failed — check $LOG_DIR/backend.log"
    fi
}

# ─────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────
CMD="${1:-help}"
detect_venv_paths

case "$CMD" in
    install)
        detect_python
        setup_directories
        setup_venv
        setup_frontend
        db_init
        log "✓ Installation complete! Run: ./scripts/deploy.sh start"
        ;;
    start)
        [ -f "$VENV_ACTIVATE" ] || err "Not installed. Run: ./scripts/deploy.sh install"
        start_backend && check_health
        ;;
    stop)
        stop_backend
        ;;
    restart)
        stop_backend; sleep 1; start_backend; check_health
        ;;
    test)
        [ -f "$VENV_ACTIVATE" ] || err "Not installed."
        source "$VENV_ACTIVATE"
        python -m pytest tests/ -v --tb=short 2>&1 | tee "$LOG_DIR/test_results.log"
        ;;
    train)
        [ -f "$VENV_ACTIVATE" ] || err "Not installed."
        MODEL="${2:-unet}"
        source "$VENV_ACTIVATE"
        python scripts/train.py --model "$MODEL" --data_dir "$DATA_DIR/train" "${@:3}"
        ;;
    logs)
        tail -f "$LOG_DIR/backend.log"
        ;;
    status)
        [ -f "$LOG_DIR/backend.pid" ] \
            && kill -0 "$(cat "$LOG_DIR/backend.pid")" 2>/dev/null \
            && log "✓ Running (PID $(cat "$LOG_DIR/backend.pid"))" \
            || log "✗ Not running."
        ;;
    clean)
        rm -rf "$DATA_DIR/outputs"/* "$LOG_DIR"/*.log 2>/dev/null || true
        log "Cleaned."
        ;;
    check)
        detect_python
        log "Python: $($PYTHON --version)"
        command -v npm &>/dev/null && log "npm: $(npm --version)" || log "npm: not found"
        ;;
    help|*)
        cat << 'HELP'

Usage: ./scripts/deploy.sh <command>

  install     Install all deps, create venv, init database
  start       Start the FastAPI backend
  stop        Stop the backend
  restart     Restart the backend
  test        Run pytest test suite
  train       Train AI models  (unet | esrgan | 3dcnn)
  logs        Tail backend logs
  status      Check if server is running
  clean       Remove output files
  check       Verify Python and npm versions
  help        Show this message

Windows (Git Bash) Quick Fix:
  If you see "Python not found", your Python is not in PATH.
  1. Open Windows Settings → Apps → Advanced app execution aliases
  2. Disable the "python.exe" Microsoft Store alias
  3. Reinstall Python 3.13 with "Add to PATH" checked
  4. Restart Git Bash, then: ./scripts/deploy.sh install

HELP
        ;;
esac