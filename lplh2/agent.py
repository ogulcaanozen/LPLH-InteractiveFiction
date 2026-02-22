"""LPLH2 Agent - Enhanced Zero-shot Decision-making.

Extends the original LPLH agent with neutral-state experience storage.

Original LPLH stores experience only on score changes (reward_change != 0).
LPLH2 also stores experience for four neutral-state triggers:
  1. Navigation       — agent enters a previously unvisited location
  2. Narrative        — agent examines/reads/talks and gets meaningful content
  3. Environmental    — an action changes the game world without giving points
  4. Error correction — agent finds a valid command after 2+ consecutive failures
"""

import re
import logging
from .kg_map import KGMap
from .action_space import ActionSpace
from .experience_lib import ExperienceLib
from .llm_client import LLMClient
from .prompts import LPLH_ACTION_GENERATION_PROMPT
from . import config

logger = logging.getLogger(__name__)


class LPLHAgent:
    """LPLH2 Agent for playing Interactive Fiction games."""

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()
        self.kg_map = KGMap()
        self.action_space = ActionSpace()
        self.experience_lib = ExperienceLib()

        # History tracking
        self.history = []          # list of (action, observation) tuples
        self.prev_action = None
        self.prev_score = 0
        self.total_score = 0
        self.step_count = 0

        # Neutral-state tracking
        self.consecutive_failures = 0        # increments on invalid, resets on valid
        self.recent_failed_actions = []      # sliding window of recent invalid commands

        # Per-step detail log for tracking
        self.step_details = []

    def reset(self, keep_experiences: bool = True):
        """Reset the agent for a new epoch."""
        self.kg_map.reset()
        self.action_space.reset()
        if not keep_experiences:
            self.experience_lib.reset()
        self.history = []
        self.prev_action = None
        self.prev_score = 0
        self.total_score = 0
        self.step_count = 0
        self.consecutive_failures = 0
        self.recent_failed_actions = []
        self.step_details = []
        logger.info(f"Agent reset (keep_experiences={keep_experiences})")

    def act(self, observation: str, score: int, done: bool, info: dict) -> str:
        """Decide the next action given the current game state."""
        self.step_count += 1
        reward_change = score - self.prev_score

        # ── History update ────────────────────────────────────
        if self.prev_action is not None:
            self.history.append((self.prev_action, observation))
            if len(self.history) > config.HISTORY_LENGTH:
                self.history = self.history[-config.HISTORY_LENGTH:]

        # Build detailed log for this step
        detail = {
            "step": self.step_count,
            "observation": observation,
            "score": score,
            "reward_change": reward_change,
            "prev_action": self.prev_action,
            "modules": {},
        }

        # ── Snapshot state BEFORE Step 1 (for navigation detection) ──
        visited_rooms_before = set(self.kg_map.visited_rooms)
        prev_location = self.kg_map.current_location

        # ── Step 1: Update KG-map with relation extraction ────
        extracted_triples = []
        if self.prev_action:
            try:
                extracted_triples = self.llm.extract_relations(self.prev_action, observation)
                self.kg_map.update(extracted_triples, self.prev_action)
                logger.debug(f"KG-map updated with {len(extracted_triples)} triples")
            except Exception as e:
                logger.warning(f"Relation extraction failed: {e}")
                extracted_triples = [("ERROR", str(e), "")]

        detail["modules"]["kg_map"] = {
            "extracted_triples": [(s, r, o) for s, r, o in extracted_triples],
            "current_location": self.kg_map.current_location,
            "rooms_visited": list(self.kg_map.visited_rooms),
            "inventory": list(self.kg_map.inventory),
            "room_info": self.kg_map.get_current_room_info(),
        }

        # ── Step 2: Validate previous action & store in action space ──
        action_valid = None
        action_split = None
        if self.prev_action:
            try:
                is_valid = self.llm.validate_action(self.prev_action, observation)
                action_valid = is_valid
                if not is_valid:
                    prev_lower = self.prev_action.lower().strip()
                    if prev_lower in self.kg_map._direction_set():
                        self.kg_map.mark_direction_tried(prev_lower)
                else:
                    prev_lower = self.prev_action.lower().strip()
                    if prev_lower in self.kg_map._direction_set():
                        self.kg_map.mark_direction_tried(prev_lower)
                if is_valid:
                    split = self.llm.split_action(self.prev_action)
                    action_split = split
                    self.action_space.store_action(split["verb"], split["objects"])
                    logger.debug(f"Valid action stored: {split}")

                    prev_lower = self.prev_action.lower().strip()
                    if prev_lower.startswith("take ") or prev_lower.startswith("get "):
                        item = self.prev_action[5:].strip() if prev_lower.startswith("take ") else self.prev_action[4:].strip()
                        self.kg_map.take_item(item)
                    elif prev_lower.startswith("drop "):
                        self.kg_map.drop_item(self.prev_action[5:].strip())
                    elif prev_lower.startswith(("eat ", "drink ", "give ")):
                        item = self.prev_action.split(" ", 1)[1].strip() if " " in self.prev_action else ""
                        if item:
                            self.kg_map.consume_item(item)
            except Exception as e:
                logger.warning(f"Action validation/splitting failed: {e}")
                action_valid = f"ERROR: {e}"

        detail["modules"]["action_space"] = {
            "prev_action_valid": action_valid,
            "action_split": action_split,
            "total_verbs": len(self.action_space.verbs),
            "total_actions_learned": self.action_space.num_actions(),
            "all_verbs": list(self.action_space.verbs.keys()),
        }

        # ── Step 3a: Summarize experience on score change (original) ──
        experience_summary = None
        if reward_change != 0 and self.prev_action:
            try:
                history_text = self._format_history()
                exp_summary = self.llm.summarize_experience(
                    history=history_text,
                    reward_change=reward_change,
                    current_score=score,
                )
                experience_summary = exp_summary
                self.experience_lib.store_experience(
                    experience_text=exp_summary,
                    metadata={
                        "trigger": "score_change",
                        "score_change": reward_change,
                        "current_score": score,
                        "step": self.step_count,
                        "location": self.kg_map.current_location or "unknown",
                    },
                )
                logger.info(f"Experience stored: score change {reward_change:+d}")
            except Exception as e:
                logger.warning(f"Experience summarization failed: {e}")
                experience_summary = f"ERROR: {e}"

        # ── Step 3b: Neutral-state experience storage (LPLH2 enhancement) ──
        neutral_triggers = []
        neutral_summaries = []

        if self.prev_action and reward_change == 0 and not done:
            neutral_triggers = self._detect_neutral_triggers(
                observation=observation,
                action_valid=action_valid,
                visited_rooms_before=visited_rooms_before,
                prev_location=prev_location,
            )

            for trigger_type, trigger_meta in neutral_triggers:
                try:
                    summary = self.llm.summarize_neutral_experience(
                        trigger=trigger_type,
                        action=self.prev_action,
                        observation=observation,
                        location=self.kg_map.current_location or "unknown",
                        prev_location=trigger_meta.get("prev_location"),
                        failed_attempts=trigger_meta.get("failed_attempts"),
                    )
                    if summary:
                        self.experience_lib.store_experience(
                            experience_text=summary,
                            metadata={
                                "trigger": trigger_type,
                                "score_change": 0,
                                "current_score": score,
                                "step": self.step_count,
                                "location": self.kg_map.current_location or "unknown",
                            },
                        )
                        neutral_summaries.append((trigger_type, summary))
                        logger.info(f"Neutral experience stored: {trigger_type}")
                except Exception as e:
                    logger.warning(f"Neutral experience failed ({trigger_type}): {e}")

        # Update consecutive-failure counter AFTER trigger detection
        # (so error_correction check above can still read the current count)
        if action_valid is True:
            self.consecutive_failures = 0
            self.recent_failed_actions = []
        elif action_valid is False:
            self.consecutive_failures += 1
            self.recent_failed_actions.append(self.prev_action)
            if len(self.recent_failed_actions) > 5:
                self.recent_failed_actions = self.recent_failed_actions[-5:]

        # ── Step 4: Retrieve relevant experiences ─────────────
        query = f"Location: {self.kg_map.current_location}. Observation: {observation[:200]}"
        experiences = self.experience_lib.retrieve_relevant(query)

        detail["modules"]["experience_lib"] = {
            "score_changed": reward_change != 0,
            "new_experience_summary": experience_summary,
            "neutral_triggers_fired": [t for t, _ in neutral_triggers],
            "neutral_summaries": neutral_summaries,
            "retrieved_experiences": experiences,
            "total_experiences": self.experience_lib.num_experiences(),
        }

        # ── Step 5: Generate next command ─────────────────────
        room_info = self.kg_map.get_current_room_info()
        current_objects = room_info.get("objects", [])

        prompt = LPLH_ACTION_GENERATION_PROMPT.format(
            kg_map=self.kg_map.to_prompt_string(),
            action_pairs=self.action_space.to_prompt_string(current_objects),
            experiences=experiences,
            history=self._format_history(),
            history_length=config.HISTORY_LENGTH,
            observation=observation,
        )

        raw_llm_response = ""
        try:
            raw_llm_response = self.llm.chat(
                system_prompt="You are an expert player of text-based interactive fiction games.",
                user_prompt=prompt,
                think=True,
            )
            command = self._parse_command(raw_llm_response)
        except Exception as e:
            err_str = str(e).lower()
            if any(x in err_str for x in ["connect", "refused", "unreachable", "failed to connect"]):
                raise RuntimeError(f"Ollama server unreachable: {e}") from e
            logger.error(f"Action generation failed: {e}")
            raw_llm_response = f"ERROR: {e}"
            command = "look"

        detail["modules"]["action_generation"] = {
            "prompt_kg_map": self.kg_map.to_prompt_string(),
            "prompt_action_pairs": self.action_space.to_prompt_string(current_objects),
            "prompt_experiences": experiences,
            "full_prompt": prompt,
            "llm_raw_response": raw_llm_response,
            "parsed_command": command,
        }

        # ── Update state ──────────────────────────────────────
        self.prev_action = command
        self.prev_score = score
        self.total_score = score

        detail["final_command"] = command
        self.step_details.append(detail)

        logger.info(f"Step {self.step_count}: score={score} ({reward_change:+d}) "
                    f"cmd='{command}' loc='{self.kg_map.current_location}'")

        return command

    # ── Neutral-state detection ───────────────────────────────

    def _detect_neutral_triggers(self, observation: str, action_valid,
                                  visited_rooms_before: set,
                                  prev_location: str) -> list:
        """Detect which neutral-state triggers fired this step.

        Returns a list of (trigger_type, metadata_dict) tuples.
        At most one trigger per type fires per step.
        """
        triggers = []

        # 1. Navigation: entered a room not visited before this step
        new_location = self.kg_map.current_location
        if (new_location
                and new_location not in visited_rooms_before
                and new_location != prev_location):
            triggers.append(("navigation", {"prev_location": prev_location}))

        # Only check the remaining triggers if the action was confirmed valid
        if action_valid is not True:
            return triggers

        # 2. Narrative: examine / read / talk with informative result
        if self._is_narrative_action(self.prev_action) and self._is_informative(observation):
            triggers.append(("narrative", {}))

        # 3. Environmental change: non-movement action that altered the world
        #    (skip if we already fired navigation — entering a new room via an
        #     unusual verb like "go through window" is handled by navigation)
        nav_fired = any(t == "navigation" for t, _ in triggers)
        if (not nav_fired
                and not self._is_movement_action(self.prev_action)
                and self._is_environmental_change(observation)):
            triggers.append(("environmental", {}))

        # 4. Error correction: valid action after 2+ consecutive failures
        if self.consecutive_failures >= 2:
            triggers.append(("error_correction", {
                "failed_attempts": list(self.recent_failed_actions),
            }))

        return triggers

    # ── Detection helpers ─────────────────────────────────────

    def _is_narrative_action(self, action: str) -> bool:
        """True if the action is an examine / read / talk type."""
        a = action.lower().strip()
        return a.startswith((
            "examine ", "read ", "look at ", "inspect ", "describe ",
            "ask ", "talk to ", "x ",
        ))

    def _is_informative(self, observation: str) -> bool:
        """True if the observation contains meaningful content (not a noise response)."""
        obs = observation.lower()
        noise = [
            "nothing special", "nothing unusual", "i don't understand",
            "i don't know the word", "you can't see any", "there is nothing",
            "how does one", "you must tell me", "that's not a verb",
            "you can't", "i don't see any", "what do you want to",
        ]
        if any(p in obs for p in noise):
            return False
        return len(observation.strip()) > 50

    def _is_movement_action(self, action: str) -> bool:
        """True if the action is a movement/navigation command."""
        directions = {
            "north", "south", "east", "west",
            "northeast", "northwest", "southeast", "southwest",
            "up", "down", "n", "s", "e", "w",
            "ne", "nw", "se", "sw", "in", "out",
        }
        a = action.lower().strip()
        if a in directions:
            return True
        return a.startswith((
            "go ", "move ", "walk ", "run ",
            "climb up", "climb down", "go through",
            "go up", "go down", "enter ", "exit ",
        ))

    def _is_environmental_change(self, observation: str) -> bool:
        """True if the observation indicates the game world was altered."""
        obs = observation.lower()
        change_phrases = [
            "now open", "now closed", "opens", "opened", "closes", "closed",
            "you open", "you close", "you move", "you push", "you pull",
            "you unlock", "you lock", "click", "you hear a click",
            "a light", "light comes on", "light goes out",
            "reveals", "revealed", "appears", "disappears",
            "the door", "the gate", "the window",
            "passage", "path is now", "way is now",
            "great effort", "with effort",
        ]
        return any(p in obs for p in change_phrases)

    # ── Shared helpers ────────────────────────────────────────

    def _parse_command(self, response: str) -> str:
        """Extract the game command from the LLM response."""
        com_match = re.search(r"<com>\s*(.+?)\s*</com>", response)
        if com_match:
            cmd = self._clean_command(com_match.group(1))
            if self._is_plausible_command(cmd):
                return cmd

        start_match = re.search(r"\|start\|\s*(.+?)\s*\|end\|", response, re.DOTALL)
        if start_match:
            text = start_match.group(1).strip()
            text = re.sub(r"<[^>]+>", "", text).strip()
            cmd = self._clean_command(text.split("\n")[0])
            if self._is_plausible_command(cmd):
                return cmd

        return "look"

    def _clean_command(self, cmd: str) -> str:
        """Strip markdown formatting that the LLM sometimes wraps commands in."""
        cmd = cmd.strip()
        cmd = re.sub(r'^`+|`+$', '', cmd).strip()
        cmd = re.sub(r'^\*+|\*+$', '', cmd).strip()
        cmd = re.sub(r'^\[+|\]+$', '', cmd).strip()
        return cmd

    def _is_plausible_command(self, cmd: str) -> bool:
        """Return False if cmd looks like an assistant response rather than a game command."""
        if not cmd:
            return False
        if len(cmd) > 50:
            return False
        bad_phrases = ["would you", "i can help", "as an ai", "here's", "please ", "i'll "]
        return not any(p in cmd.lower() for p in bad_phrases)

    def _format_history(self) -> str:
        """Format the recent history for prompt inclusion."""
        if not self.history:
            return "No history yet. This is the start of the game."

        output = []
        for i, (action, obs) in enumerate(self.history):
            output.append(f"Turn {i+1}:")
            output.append(f"  Action: {action}")
            obs_short = obs[:300] + "..." if len(obs) > 300 else obs
            output.append(f"  Observation: {obs_short}")
        return "\n".join(output)

    def get_stats(self) -> dict:
        """Get current agent statistics."""
        return {
            "step": self.step_count,
            "score": self.total_score,
            "rooms_visited": self.kg_map.num_rooms(),
            "actions_learned": self.action_space.num_actions(),
            "experiences_stored": self.experience_lib.num_experiences(),
            "current_location": self.kg_map.current_location,
        }

    def get_step_details(self) -> list:
        """Return the detailed per-step log."""
        return self.step_details
