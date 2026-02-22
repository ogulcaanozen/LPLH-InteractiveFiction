# LPLH: Learning to Play Like Humans

A cognitively inspired framework that enables LLM agents to play Interactive Fiction (IF) games through structured map building, action learning, and feedback-driven experience analysis.

Based on the paper: *"Learning to Play Interactive Fiction Games with LLM Guidance"* (ACL 2025).

## Architecture

The framework consists of three core modules that work together:

| Module | Description |
|--------|-------------|
| **KG-Map** | Dynamic Knowledge Graph that tracks locations, objects, and spatial relationships |
| **Action Space** | Learns valid verb-object pairings from gameplay to suggest context-appropriate actions |
| **Experience Library** | Stores and retrieves past experiences via RAG (ChromaDB) for reflective decision-making |

```
User ──▶ run_wsl.sh ──▶ LPLH Agent ──▶ Jericho Engine (Z-Machine)
                           │
                    ┌──────┼──────┐
                    ▼      ▼      ▼
                 KG-Map  Action  Experience
                         Space   Library
                    │      │      │
                    └──────┼──────┘
                           ▼
                     LLM (Ollama)
```

## Quick Start

### Prerequisites

- **Windows 10/11** with WSL2 (Ubuntu)
- **Python 3.12+** in WSL
- **Ollama** installed on Windows with a model (e.g., `qwen3:8b`)
- Game ROM files (e.g., `zork1.z5`) in the `games/` directory

### Setup

```bash
# 1. In WSL, create a virtual environment and install dependencies
cd /mnt/c/path/to/IFGames
python3 -m venv .wsl_venv
source .wsl_venv/bin/activate
pip install jericho ollama openai chromadb sentence-transformers numpy

# 2. Place game ROMs in games/ directory
mkdir -p games
# Download zork1.z5 and place it in games/

# 3. Ensure Ollama is running on Windows with your chosen model
ollama pull qwen3:8b
```

### Running

```bash
# From Windows PowerShell:
wsl -e bash /mnt/c/path/to/IFGames/run_wsl.sh --game zork1 --epochs 3 --steps 100

# Quick test (5 steps):
wsl -e bash /mnt/c/path/to/IFGames/run_wsl.sh --game zork1 --epochs 1 --steps 5

# With a different model:
wsl -e bash /mnt/c/path/to/IFGames/run_wsl.sh --game zork1 --model llama3:8b

# List available games:
wsl -e bash /mnt/c/path/to/IFGames/run_wsl.sh --list-games
```

### Configuration

Key settings in `lplh/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend (`ollama` or `openai`) |
| `LLM_MODEL` | `qwen3:8b` | Model name |
| `NUM_EPOCHS` | `10` | Training epochs |
| `MAX_STEPS_PER_EPOCH` | `250` | Steps per epoch |
| `LLM_TEMPERATURE` | `0.6` | Agent generation temperature |
| `EXPERIENCE_TOP_K` | `3` | RAG retrieval count |

All settings can be overridden via environment variables (prefix: `LPLH_`).

## Project Structure

```
IFGames/
├── lplh/
│   ├── config.py            # Central configuration
│   ├── prompts.py           # Prompt templates (Tables 4-9 from paper)
│   ├── llm_client.py        # Unified Ollama/OpenAI client
│   ├── kg_map.py            # Module 1: Dynamic Knowledge Graph
│   ├── action_space.py      # Module 2: Action Space Learning
│   ├── experience_lib.py    # Module 3: Experience Library (RAG)
│   ├── agent.py             # LPLH Agent (orchestrates all modules)
│   └── game_runner.py       # Multi-epoch game loop
├── games/                   # Game ROM files (.z5, .z8)
├── data/                    # Results and ChromaDB storage
├── run_game.py              # CLI entry point
├── run_wsl.sh               # WSL launch script
└── requirements.txt         # Python dependencies
```

## How It Works

Each game step follows this pipeline:

1. **Relation Extraction** — LLM extracts `(subject, relation, object)` triples from game text
2. **KG-Map Update** — Triples update the spatial knowledge graph
3. **Action Validation** — Previous action validated and stored in action space
4. **Experience Summarization** — Score changes trigger experience capture
5. **RAG Retrieval** — Relevant past experiences retrieved from ChromaDB
6. **Command Generation** — LLM generates next command using map + actions + experiences

## License

This project is for research purposes. Game ROM files are not included and must be obtained separately.
