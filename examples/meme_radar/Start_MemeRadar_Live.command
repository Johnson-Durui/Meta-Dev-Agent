#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
python3 -m uvicorn server:app --app-dir "$SCRIPT_DIR" --host 127.0.0.1 --port 8124
