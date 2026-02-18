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


def _trunc(text: str, length: int = 200) -> str:
    """Truncate text for display."""
    if not text:
        return ""
    t = text.replace("\n", " ").strip()
    return t[:length] + "..." if len(t) > length else t


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
        print(f"\n{'='*70}")
        print(f"  LPLH Framework - Playing: {game_name}")
        print(f"  Epochs: {self.num_epochs}, Steps/epoch: {self.max_steps}")
        print(f"  LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
        print(f"  LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
        print(f"{'='*70}\n")
        sys.stdout.flush()

        # Integrity Check
        if os.path.getsize(self.game_path) == 0:
            logger.error("Game file is empty (0 bytes)! Upload failed?")
            print("\n❌ CRITICAL ERROR: Game file is 0 bytes. Please re-upload zork1.z5 correctly.\n")
            return {}


        start_time = time.time()
        all_step_logs = []   # detailed logs across epochs
        
        try:
            for epoch in range(1, self.num_epochs + 1):
                epoch_result, epoch_steps = self._run_epoch(env, agent, epoch, game_name)
                self.epoch_results.append(epoch_result)
                self.all_scores.append(epoch_result["final_score"])
                all_step_logs.append({
                    "epoch": epoch,
                    "steps": epoch_steps,
                })
    
                print(f"\n{'─'*70}")
                print(f"  Epoch {epoch}/{self.num_epochs} Complete")
                print(f"  Final Score: {epoch_result['final_score']}  |  "
                      f"Max Score: {epoch_result['max_score']}  |  "
                      f"Steps Used: {epoch_result['steps_used']}")
                print(f"  Rooms: {epoch_result['rooms_visited']}  |  "
                      f"Actions Learned: {epoch_result['actions_learned']}  |  "
                      f"Experiences: {epoch_result['experiences_stored']}")
                print(f"{'─'*70}\n")
        except KeyboardInterrupt:
            print("\n\n🛑 Run interrupted by user (Ctrl+C). Saving partial results...")
            logger.warning("Run interrupted by user.")
        finally:
            elapsed = time.time() - start_time
    
            # Compute summary statistics (if any epochs finished)
            if self.epoch_results:
                results = self._compute_summary(game_name, elapsed)
                # Save results (summary)
                self._save_results(results, game_name)
            
            # Save detailed step log (whatever we have)
            if all_step_logs:
                self._save_step_log(all_step_logs, game_name)
    
            print(f"\n{'='*70}")
            if self.epoch_results:
                print(f"  LPLH Results for '{game_name}' (Partial)")
                print(f"  Average Score (all): {results['avg_score_all']:.1f}")
                print(f"  Max Score: {results['max_score']}")
            else:
                print(f"  LPLH Results: No full epochs completed.")
            print(f"  Total Time: {elapsed:.1f}s")
            print(f"{'='*70}\n")
            
            if self.epoch_results:
                return results
            return {}


    def _run_epoch(self, env, agent: LPLHAgent, epoch: int, game_name: str) -> tuple:
        """Run a single epoch of gameplay.
        
        Returns:
            Tuple of (epoch_result_dict, step_details_list)
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

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  EPOCH {epoch}")
            print(f"{'='*70}")
            print(f"\n  📍 Initial Observation:")
            print(f"  {_trunc(observation, 300)}")
            print()

        for step in range(1, self.max_steps + 1):
            # Agent decides the next action
            action = agent.act(observation, score, False, info)

            # Execute action in the game
            # Execute action in the game
            try:
                # print(f"DEBUG: Executing step {step} with action '{action}'...", flush=True)
                observation, reward, done, info = env.step(action)
                # print(f"DEBUG: Step returned done={done}", flush=True)
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                logger.error(f"CRITICAL: Jericho step failed/crashed: {e} (Type: {type(e)})")
                print(f"\n❌ CRITICAL GAME CRASH at Step {step}: {e}")
                import traceback
                traceback.print_exc()
                done = True
                observation = "Game crashed."
                reward = 0
                info = {}
                break  # Stop the loop on crash


            score = info.get("score", score + reward)
            max_score = max(max_score, score)

            # ── Live console output ───────────────────────────
            if self.verbose:
                self._print_step_detail(agent, step, action, observation, score, reward)

            if done:
                logger.info(f"Epoch {epoch} ended at step {step}")
                if self.verbose:
                    print(f"\n  🏁 GAME OVER at step {step}")
                break

        stats = agent.get_stats()
        step_details = agent.get_step_details()

        epoch_result = {
            "epoch": epoch,
            "final_score": score,
            "max_score": max_score,
            "steps_used": step,
            "rooms_visited": stats["rooms_visited"],
            "actions_learned": stats["actions_learned"],
            "experiences_stored": stats["experiences_stored"],
            "game": game_name,
        }

        return epoch_result, step_details

    def _print_step_detail(self, agent: LPLHAgent, step: int, action: str,
                            observation: str, score: int, reward: int):
        """Print detailed per-step info to console during the run."""
        # Get the latest step detail from the agent
        if not agent.step_details:
            return
        d = agent.step_details[-1]
        modules = d.get("modules", {})

        # ── Header ────────────────────────────────────────────
        reward_str = f" ({'+' if reward >= 0 else ''}{reward})" if reward != 0 else ""
        print(f"  ┌─ Step {step}  │  Score: {score}{reward_str}  │  "
              f"Location: {modules.get('kg_map', {}).get('current_location', '?')}")
        print(f"  │")

        # ── Command & Observation ─────────────────────────────
        print(f"  │  🎮 Command: {action}")
        print(f"  │  📜 Observation: {_trunc(observation, 250)}")

        # ── Module 1: KG-Map ──────────────────────────────────
        kg = modules.get("kg_map", {})
        triples = kg.get("extracted_triples", [])
        if triples:
            print(f"  │")
            print(f"  │  🗺️  KG-Map ({len(kg.get('rooms_visited', []))} rooms)")
            for s, r, o in triples[:5]:
                print(f"  │     Triple: ({s}, {r}, {o})")
            if len(triples) > 5:
                print(f"  │     ... +{len(triples)-5} more")
        inv = kg.get("inventory", [])
        if inv:
            print(f"  │     Inventory: {', '.join(inv)}")

        # ── Module 2: Action Space ────────────────────────────
        act_mod = modules.get("action_space", {})
        valid = act_mod.get("prev_action_valid")
        split = act_mod.get("action_split")
        if valid is not None:
            status = "✅ Valid" if valid is True else ("❌ Invalid" if valid is False else f"⚠️ {valid}")
            split_str = f" → verb='{split['verb']}' obj={split['objects']}" if isinstance(split, dict) else ""
            print(f"  │  ⚡ Action Space ({act_mod.get('total_actions_learned', 0)} learned): "
                  f"prev_action {status}{split_str}")
            if act_mod.get("all_verbs"):
                print(f"  │     Verbs: {', '.join(act_mod['all_verbs'][:10])}"
                      f"{'...' if len(act_mod.get('all_verbs', [])) > 10 else ''}")

        # ── Module 3: Experience Library ──────────────────────
        exp = modules.get("experience_lib", {})
        if exp.get("score_changed"):
            summary = exp.get("new_experience_summary", "")
            if summary and not str(summary).startswith("ERROR"):
                print(f"  │  💡 New Experience: {_trunc(str(summary), 150)}")
        retrieved = exp.get("retrieved_experiences", "")
        if retrieved and retrieved != "No relevant experiences found yet.":
            print(f"  │  📚 Retrieved: {_trunc(str(retrieved), 150)}")

        # ── LLM Response ──────────────────────────────────────
        gen = modules.get("action_generation", {})
        raw = gen.get("llm_raw_response", "")
        if raw and not str(raw).startswith("ERROR"):
            print(f"  │  🤖 LLM Response: {_trunc(str(raw), 200)}")

        print(f"  └{'─'*69}")
        print()

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

    def _save_step_log(self, all_step_logs: list, game_name: str):
        """Save detailed per-step log to a JSON file for post-run analysis."""
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"steplog_{game_name}_{timestamp}.json"
        filepath = os.path.join(config.LOGS_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_step_logs, f, indent=2, default=str)

        print(f"Detailed step log saved to: {filepath}")
