#!/bin/bash
# LPLH Framework - WSL Run Script
# Runs the LPLH agent on IF games via WSL2
#
# Usage (from Windows PowerShell):
#   wsl -e bash /mnt/c/.../IFGames/run_wsl.sh --game zork1 --epochs 1 --steps 5
#
# Usage (from within WSL):
#   ./run_wsl.sh --game zork1 --epochs 1 --steps 50

DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate the WSL virtual environment
source "$DIR/.wsl_venv/bin/activate"

# Ollama runs on Windows host, accessible via localhost (mirrored networking)
export OLLAMA_BASE_URL="http://localhost:11434"

# Run the LPLH framework
python3 "$DIR/run_game.py" "$@"
