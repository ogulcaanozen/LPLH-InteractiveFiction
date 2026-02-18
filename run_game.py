"""LPLH Framework - Entry Point.

Run an LPLH agent on an Interactive Fiction game.

Usage:
    python run_game.py --game zork1 --epochs 3 --steps 250 --verbose
    python run_game.py --game detective --epochs 1 --steps 50
    python run_game.py --list-games
"""

import os
import sys
import argparse
import logging

# Add parent dir to path so we can import lplh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lplh import config
from lplh.game_runner import GameRunner


def find_game(game_name: str) -> str:
    """Find the game ROM file in the games directory."""
    games_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
    
    # Try exact filename first
    if os.path.isfile(game_name):
        return game_name

    # Common IF game extensions
    extensions = [".z5", ".z8", ".z3", ".z6", ".zblorb", ".ulx", ".dat"]
    
    for ext in extensions:
        path = os.path.join(games_dir, game_name + ext)
        if os.path.isfile(path):
            return path
    
    # Try without extension
    for f in os.listdir(games_dir) if os.path.isdir(games_dir) else []:
        if f.lower().startswith(game_name.lower()):
            return os.path.join(games_dir, f)
    
    return None


def list_games():
    """List all available game ROMs."""
    games_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games")
    if not os.path.isdir(games_dir):
        print(f"Games directory not found: {games_dir}")
        print("Please download game ROM files and place them in the 'games/' directory.")
        return

    print("Available games:")
    for f in sorted(os.listdir(games_dir)):
        if any(f.endswith(ext) for ext in [".z5", ".z8", ".z3", ".z6", ".zblorb", ".ulx", ".dat"]):
            name = os.path.splitext(f)[0]
            size = os.path.getsize(os.path.join(games_dir, f))
            print(f"  {name:20s} ({f}, {size/1024:.0f}KB)")


def main():
    parser = argparse.ArgumentParser(
        description="LPLH Framework - Play IF games like humans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_game.py --game zork1 --epochs 3 --steps 250
  python run_game.py --game detective --epochs 1 --steps 50 --verbose
  python run_game.py --list-games
  
Environment variables:
  LPLH_LLM_PROVIDER   "ollama" (default) or "openai"
  LPLH_LLM_MODEL      Model name (default: "qwen3:8b")
  OPENAI_API_KEY       Required if using OpenAI provider
        """
    )
    
    parser.add_argument("--game", type=str, default="zork1",
                        help="Game name or path to ROM file (default: zork1)")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of epochs (default: {config.NUM_EPOCHS})")
    parser.add_argument("--steps", type=int, default=None,
                        help=f"Max steps per epoch (default: {config.MAX_STEPS_PER_EPOCH})")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print step-by-step output (default: True)")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    parser.add_argument("--list-games", action="store_true",
                        help="List available games and exit")
    parser.add_argument("--clear-experiences", action="store_true",
                        help="Clear stored experiences before running")
    parser.add_argument("--model", type=str, default=None,
                        help="Override LLM model name")
    parser.add_argument("--provider", type=str, default=None,
                        help="Override LLM provider (ollama/openai)")

    args = parser.parse_args()

    if args.list_games:
        list_games()
        return

    # Override config if specified
    if args.model:
        config.LLM_MODEL = args.model
    if args.provider:
        config.LLM_PROVIDER = args.provider

    # Setup logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Find the game
    game_path = find_game(args.game)
    if not game_path:
        print(f"Error: Game '{args.game}' not found.")
        print(f"Place game ROM files in: {os.path.join(os.path.dirname(__file__), 'games')}")
        print("Run with --list-games to see available games.")
        sys.exit(1)

    print(f"Game found: {game_path}")

    # Clear experiences if requested
    if args.clear_experiences:
        from lplh.experience_lib import ExperienceLib
        exp = ExperienceLib()
        exp.clear_collection()
        print("Experiences cleared.")

    # Run the game
    runner = GameRunner(
        game_path=game_path,
        num_epochs=args.epochs,
        max_steps=args.steps,
        verbose=not args.quiet,
    )

    try:
        results = runner.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        logging.exception("Game run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
