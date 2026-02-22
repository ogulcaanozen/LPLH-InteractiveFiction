"""Game Runner - Orchestrates the full game loop.

Manages epochs, steps, scoring, and logging for running
LPLH agents on IF games via the Jericho environment.
"""

import os
import re
import sys
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
        self._log_file = None

    def run(self) -> dict:
        """Run the full game experiment.
        
        Returns:
            Dictionary with all results and statistics.
        """
        import jericho

        # Initialize Jericho environment
        env = jericho.FrotzEnv(self.game_path)
        game_name = os.path.splitext(os.path.basename(self.game_path))[0]

        # Open human-readable run log
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        log_path = os.path.join(config.LOGS_DIR, f"run_log_{game_name}_{self._run_timestamp}.txt")
        self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        self._log_file.write(f"LPLH Run Log — {game_name}\n")
        self._log_file.write(f"Epochs: {self.num_epochs} | Steps/epoch: {self.max_steps} | Model: {config.LLM_PROVIDER}/{config.LLM_MODEL}\n")
        self._log_file.write("=" * 70 + "\n")
        print(f"Run log: {log_path}")

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
        except RuntimeError as e:
            if "unreachable" in str(e).lower() or "connect" in str(e).lower():
                print(f"\n\n❌ OLLAMA SERVER LOST: {e}")
                print("The Ollama server crashed or became unreachable. Stopping run.")
                print("Partial results (completed epochs) will be saved.")
            else:
                raise
            logger.error(f"Run stopped: {e}")
        finally:
            if self._log_file:
                self._log_file.close()
                self._log_file = None
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

        # Set to verbose mode (paper specifies this).
        # We discard the verbose confirmation ("Maximum verbosity.") and keep
        # the real room description from env.reset() as the starting observation.
        try:
            env.step("verbose")
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
            self._log_step(epoch, step, action, observation, score, reward, agent)

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
              f"Location: {modules.get('kg_map', {}).get('current_location', '?')}", flush=True)
        print(f"  │", flush=True)

        # ── Command & Observation ─────────────────────────────
        print(f"  │  🎮 Command: {action}", flush=True)
        print(f"  │  📜 Observation: {_trunc(observation, 250)}", flush=True)

        # ── Module 1: KG-Map ──────────────────────────────────
        kg = modules.get("kg_map", {})
        triples = kg.get("extracted_triples", [])
        if triples:
            print(f"  │", flush=True)
            print(f"  │  🗺️  KG-Map ({len(kg.get('rooms_visited', []))} rooms)", flush=True)
            for s, r, o in triples[:5]:
                print(f"  │     Triple: ({s}, {r}, {o})", flush=True)
            if len(triples) > 5:
                print(f"  │     ... +{len(triples)-5} more", flush=True)
        inv = kg.get("inventory", [])
        if inv:
            print(f"  │     Inventory: {', '.join(inv)}", flush=True)

        # ── Module 2: Action Space ────────────────────────────
        act_mod = modules.get("action_space", {})
        valid = act_mod.get("prev_action_valid")
        split = act_mod.get("action_split")
        if valid is not None:
            status = "✅ Valid" if valid is True else ("❌ Invalid" if valid is False else f"⚠️ {valid}")
            split_str = f" → verb='{split['verb']}' obj={split['objects']}" if isinstance(split, dict) else ""
            print(f"  │  ⚡ Action Space ({act_mod.get('total_actions_learned', 0)} learned): "
                  f"prev_action {status}{split_str}", flush=True)
            if act_mod.get("all_verbs"):
                print(f"  │     Verbs: {', '.join(act_mod['all_verbs'][:10])}"
                      f"{'...' if len(act_mod.get('all_verbs', [])) > 10 else ''}", flush=True)

        # ── Module 3: Experience Library ──────────────────────
        exp = modules.get("experience_lib", {})
        if exp.get("score_changed"):
            summary = exp.get("new_experience_summary", "")
            if summary and not str(summary).startswith("ERROR"):
                print(f"  │  💡 New Experience: {_trunc(str(summary), 150)}", flush=True)
        retrieved = exp.get("retrieved_experiences", "")
        if retrieved and retrieved != "No relevant experiences found yet.":
            print(f"  │  📚 Retrieved: {_trunc(str(retrieved), 150)}", flush=True)

        # ── LLM Response ──────────────────────────────────────
        gen = modules.get("action_generation", {})
        raw = gen.get("llm_raw_response", "")
        if raw and not str(raw).startswith("ERROR"):
            print(f"  │  🤖 LLM Response: {_trunc(str(raw), 200)}", flush=True)

        print(f"  └{'─'*69}", flush=True)
        print(flush=True)

    def _log_step(self, epoch: int, step: int, action: str, observation: str,
                  score: int, reward: int, agent: LPLHAgent):
        """Write a comprehensive per-step log to the run log file (real-time)."""
        if not self._log_file or not agent.step_details:
            return
        d = agent.step_details[-1]
        modules = d.get("modules", {})
        kg = modules.get("kg_map", {})
        act_mod = modules.get("action_space", {})
        gen = modules.get("action_generation", {})
        exp = modules.get("experience_lib", {})

        reward_str = f" ({'+' if reward >= 0 else ''}{reward})" if reward != 0 else ""
        location = kg.get("current_location") or "?"
        W = 80

        def section(title):
            return f"\n--- {title} {'─' * max(0, W - len(title) - 6)}\n"

        lines = []
        lines.append("\n" + "=" * W)
        lines.append(f"[EPOCH {epoch} | STEP {step:03d}]  Score: {score}{reward_str}  |  Location: {location}")
        lines.append("=" * W)

        # ── Command ───────────────────────────────────────────
        lines.append(section("COMMAND"))
        lines.append(action)

        # ── Game Response ─────────────────────────────────────
        lines.append(section("GAME RESPONSE"))
        lines.append(observation)

        # ── KG Map (full JSON) ────────────────────────────────
        lines.append(section("KG MAP (JSON)"))
        lines.append(gen.get("prompt_kg_map", kg.get("room_info", "N/A")))

        # ── Action Space ──────────────────────────────────────
        lines.append(section("ACTION SPACE"))
        lines.append(f"Total verbs learned: {len(act_mod.get('all_verbs', []))}")
        lines.append(f"Total actions learned: {act_mod.get('total_actions_learned', 0)}")
        lines.append("")
        lines.append(gen.get("prompt_action_pairs", "N/A"))

        # ── Experience Library ────────────────────────────────
        lines.append(section("EXPERIENCE LIBRARY"))
        lines.append(f"Total experiences in DB: {exp.get('total_experiences', 0)}")
        lines.append(f"Score changed this step: {exp.get('score_changed', False)}")
        if exp.get("score_changed") and exp.get("new_experience_summary"):
            summary = str(exp["new_experience_summary"])
            if not summary.startswith("ERROR"):
                lines.append("\n[NEW EXPERIENCE STORED]")
                lines.append(summary)
        lines.append("\n[RETRIEVED EXPERIENCES]")
        lines.append(str(exp.get("retrieved_experiences", "None")))

        # ── Full Prompt Sent to LLM ───────────────────────────
        lines.append(section("FULL PROMPT TO LLM"))
        lines.append(gen.get("full_prompt", "N/A"))

        # ── LLM Raw Response ──────────────────────────────────
        lines.append(section("LLM RAW RESPONSE"))
        lines.append(str(gen.get("llm_raw_response", "N/A")))

        # ── Reasoning (extracted <rea> tag) ───────────────────
        raw = str(gen.get("llm_raw_response", ""))
        rea_match = re.search(r"<rea>(.*?)</rea>", raw, re.DOTALL)
        if rea_match:
            lines.append(section("REASONING (extracted)"))
            lines.append(rea_match.group(1).strip())

        # ── Prev Action Validation ────────────────────────────
        valid = act_mod.get("prev_action_valid")
        split = act_mod.get("action_split")
        if valid is not None:
            lines.append(section("PREV ACTION VALIDATION"))
            lines.append(f"Valid: {valid}")
            if isinstance(split, dict):
                lines.append(f"Verb: {split.get('verb')}  |  Objects: {split.get('objects')}")

        lines.append("\n" + "─" * W + "\n")
        self._log_file.write("\n".join(lines) + "\n")

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
