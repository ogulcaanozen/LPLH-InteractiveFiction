"""Game Runner - Orchestrates the full game loop.

Manages epochs, steps, scoring, and logging for running
LPLH agents on IF games via the Jericho environment.
"""

import os
import json
import time
import logging
from datetime import datetime
from . import config
from .agent import LPLHAgent
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class GameRunner:
    """Runs LPLH agent on an IF game for multiple epochs."""

    def __init__(self, game_path: str, num_epochs: int = None,
                 max_steps: int = None, verbose: bool = True):
        """
        Args:
            game_path: Path to the game ROM file (e.g., "games/zork1.z5")
            num_epochs: Number of epochs to run (default from config)
            max_steps: Max steps per epoch (default from config)
            verbose: Whether to print step-by-step output
        """
        self.game_path = game_path
        self.num_epochs = num_epochs or config.NUM_EPOCHS
        self.max_steps = max_steps or config.MAX_STEPS_PER_EPOCH
        self.verbose = verbose

        # Results tracking
        self.epoch_results = []
        self.all_scores = []

    def run(self) -> dict:
        """Run the full game experiment.
        
        Returns:
            Dictionary with all results and statistics.
        """
        import jericho

        # Initialize Jericho environment
        env = jericho.FrotzEnv(self.game_path)
        game_name = os.path.splitext(os.path.basename(self.game_path))[0]

        # Initialize LPLH agent
        llm = LLMClient()
        agent = LPLHAgent(llm_client=llm)

        logger.info(f"Starting LPLH on '{game_name}': "
                     f"{self.num_epochs} epochs, {self.max_steps} steps/epoch")
        print(f"\n{'='*60}")
        print(f"  LPLH Framework - Playing: {game_name}")
        print(f"  Epochs: {self.num_epochs}, Steps/epoch: {self.max_steps}")
        print(f"  LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_result = self._run_epoch(env, agent, epoch, game_name)
            self.epoch_results.append(epoch_result)
            self.all_scores.append(epoch_result["final_score"])

            print(f"\n--- Epoch {epoch}/{self.num_epochs} Complete ---")
            print(f"  Final Score: {epoch_result['final_score']}")
            print(f"  Max Score: {epoch_result['max_score']}")
            print(f"  Steps Used: {epoch_result['steps_used']}")
            print(f"  Rooms Visited: {epoch_result['rooms_visited']}")
            print(f"  Actions Learned: {epoch_result['actions_learned']}")
            print(f"  Experiences: {epoch_result['experiences_stored']}")
            print()

        elapsed = time.time() - start_time

        # Compute summary statistics
        results = self._compute_summary(game_name, elapsed)

        # Save results
        self._save_results(results, game_name)

        print(f"\n{'='*60}")
        print(f"  LPLH Results for '{game_name}'")
        print(f"  Average Score (all): {results['avg_score_all']:.1f}")
        print(f"  Average Score (last 3): {results['avg_score_last3']:.1f}")
        print(f"  Max Score: {results['max_score']}")
        print(f"  Total Time: {elapsed:.1f}s")
        print(f"{'='*60}\n")

        return results

    def _run_epoch(self, env, agent: LPLHAgent, epoch: int, game_name: str) -> dict:
        """Run a single epoch of gameplay.
        
        Args:
            env: Jericho FrotzEnv
            agent: LPLH agent
            epoch: Current epoch number
            game_name: Name of the game
            
        Returns:
            Dictionary with epoch results
        """
        # Reset environment and agent (keep experiences across epochs!)
        observation, info = env.reset()
        agent.reset(keep_experiences=True)

        # Set to verbose mode (paper specifies this)
        try:
            verbose_obs, _, _, _ = env.step("verbose")
            if verbose_obs and "verbose" not in verbose_obs.lower():
                observation = verbose_obs
        except Exception:
            pass

        max_score = 0
        score = 0
        step_log = []

        if self.verbose:
            print(f"\n=== Epoch {epoch} ===")
            print(f"Initial: {observation[:200]}...")

        for step in range(1, self.max_steps + 1):
            # Agent decides the next action
            action = agent.act(observation, score, False, info)

            # Execute action in the game
            try:
                observation, reward, done, info = env.step(action)
            except Exception as e:
                logger.warning(f"Jericho step failed: {e}")
                observation = "Nothing happens."
                reward = 0
                done = False
                info = {}

            score = info.get("score", score + reward)
            max_score = max(max_score, score)

            # Log the step
            step_log.append({
                "step": step,
                "action": action,
                "observation": observation[:500],
                "score": score,
                "reward": reward,
            })

            if self.verbose and step <= 20:  # Only print first 20 steps in detail
                print(f"  [{step}] > {action}")
                obs_preview = observation[:150].replace("\n", " ")
                print(f"      obs: {obs_preview}")
                print(f"      score: {score} (change: {reward:+d})")
            elif self.verbose and step == 21:
                print(f"  ... (continuing quietly, will show summary at end)")

            if done:
                logger.info(f"Epoch {epoch} ended at step {step}")
                break

        stats = agent.get_stats()

        return {
            "epoch": epoch,
            "final_score": score,
            "max_score": max_score,
            "steps_used": step,
            "rooms_visited": stats["rooms_visited"],
            "actions_learned": stats["actions_learned"],
            "experiences_stored": stats["experiences_stored"],
            "game": game_name,
        }

    def _compute_summary(self, game_name: str, elapsed: float) -> dict:
        """Compute summary statistics across all epochs."""
        all_scores = [r["final_score"] for r in self.epoch_results]
        all_max = [r["max_score"] for r in self.epoch_results]

        # Paper uses last 3 epochs as "learning outcomes"
        last3_scores = all_scores[-3:] if len(all_scores) >= 3 else all_scores

        return {
            "game": game_name,
            "num_epochs": self.num_epochs,
            "max_steps": self.max_steps,
            "llm_provider": config.LLM_PROVIDER,
            "llm_model": config.LLM_MODEL,
            "avg_score_all": sum(all_scores) / len(all_scores),
            "avg_score_last3": sum(last3_scores) / len(last3_scores),
            "max_score": max(all_max),
            "all_epoch_results": self.epoch_results,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

    def _save_results(self, results: dict, game_name: str):
        """Save results to a JSON file."""
        os.makedirs(config.DATA_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{game_name}_{timestamp}.json"
        filepath = os.path.join(config.DATA_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        print(f"Results saved to: {filepath}")
