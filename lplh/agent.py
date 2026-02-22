"""LPLH Agent - Zero-shot Decision-making.

The main agent that ties all three modules together:
1. Dynamic KG-Map (spatial reasoning)
2. Action Space Learning (verb-object discovery)
3. Experience Library (reflective learning via RAG)

At each step, it integrates all module outputs with the current
observation to generate the next game command via zero-shot prompting.
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
    """LPLH Agent for playing Interactive Fiction games.

    Implements the full LPLH pipeline from the paper (Section 3.5):
    1. Update KG-map with extracted relations
    2. Validate & store action in action space
    3. Summarize experience on score change
    4. Retrieve relevant experiences
    5. Generate next command via zero-shot LLM
    """

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

        # Per-step detail log for tracking
        self.step_details = []

    def reset(self, keep_experiences: bool = True):
        """Reset the agent for a new epoch.
        
        Args:
            keep_experiences: If True, keep experiences across epochs 
                            (key for learning behavior). If False, full reset.
        """
        self.kg_map.reset()
        self.action_space.reset()
        if not keep_experiences:
            self.experience_lib.reset()
        self.history = []
        self.prev_action = None
        self.prev_score = 0
        self.total_score = 0
        self.step_count = 0
        self.step_details = []
        logger.info(f"Agent reset (keep_experiences={keep_experiences})")

    def act(self, observation: str, score: int, done: bool, info: dict) -> str:
        """Decide the next action given the current game state.
        
        This is the main method called at each game step.
        
        Args:
            observation: Current text observation from the game
            score: Current game score
            done: Whether the game has ended
            info: Additional info from Jericho
            
        Returns:
            The next command string to send to the game
        """
        self.step_count += 1
        reward_change = score - self.prev_score

        # ── History update ────────────────────────────────────
        # observation is the game's response to prev_action, so
        # (prev_action, observation) is a correct completed pair.
        # Must happen before _format_history() is called below.
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
                    # If a movement direction was invalid, remove it from may_direction
                    prev_lower = self.prev_action.lower().strip()
                    if prev_lower in self.kg_map._direction_set():
                        self.kg_map.mark_direction_tried(prev_lower)
                else:
                    # If a movement direction was valid, also remove from may_direction
                    # (the relation extractor may not always extract the direction triple)
                    prev_lower = self.prev_action.lower().strip()
                    if prev_lower in self.kg_map._direction_set():
                        self.kg_map.mark_direction_tried(prev_lower)
                if is_valid:
                    split = self.llm.split_action(self.prev_action)
                    action_split = split
                    self.action_space.store_action(split["verb"], split["objects"])
                    logger.debug(f"Valid action stored: {split}")

                    # Update inventory only for confirmed-valid take/drop actions.
                    # This must happen here (after validation) so that failed
                    # attempts like "take sword" → "The sword is too heavy" do
                    # NOT pollute the inventory. (kg_map.update no longer touches
                    # inventory so it cannot hallucinate items.)
                    prev_lower = self.prev_action.lower().strip()
                    if prev_lower.startswith("take ") or prev_lower.startswith("get "):
                        item = self.prev_action[5:].strip() if prev_lower.startswith("take ") else self.prev_action[4:].strip()
                        self.kg_map.take_item(item)
                    elif prev_lower.startswith("drop "):
                        self.kg_map.drop_item(self.prev_action[5:].strip())
                    elif prev_lower.startswith(("eat ", "drink ", "give ")):
                        # These verbs always consume the item — remove from
                        # inventory without putting it back in the room.
                        # "throw" and "use" are intentionally excluded: outcome
                        # is conditional and can't be determined from verb alone.
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

        # ── Step 3: Summarize experience on score change ──────
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

        # ── Step 4: Retrieve relevant experiences ─────────────
        query = f"Location: {self.kg_map.current_location}. Observation: {observation[:200]}"
        experiences = self.experience_lib.retrieve_relevant(query)

        detail["modules"]["experience_lib"] = {
            "score_changed": reward_change != 0,
            "new_experience_summary": experience_summary,
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
            logger.error(f"Action generation failed: {e}")
            raw_llm_response = f"ERROR: {e}"
            command = "look"  # Safe fallback

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

    def _parse_command(self, response: str) -> str:
        """Extract the game command from the LLM response.

        Looks for |start| <com>[command]</com> ... |end| pattern.
        Falls back to |start| [command] |end| pattern.
        """
        # Try <com>...</com> pattern first (LPLH format)
        com_match = re.search(r"<com>\s*(.+?)\s*</com>", response)
        if com_match:
            cmd = self._clean_command(com_match.group(1))
            if self._is_plausible_command(cmd):
                return cmd

        # Try |start| ... |end| pattern (baseline format)
        start_match = re.search(r"\|start\|\s*(.+?)\s*\|end\|", response, re.DOTALL)
        if start_match:
            text = start_match.group(1).strip()
            text = re.sub(r"<[^>]+>", "", text).strip()
            cmd = self._clean_command(text.split("\n")[0])
            if self._is_plausible_command(cmd):
                return cmd

        return "look"  # fallback — safe and avoids sending garbage to the game

    def _clean_command(self, cmd: str) -> str:
        """Strip markdown formatting that the LLM sometimes wraps commands in."""
        cmd = cmd.strip()
        cmd = re.sub(r'^`+|`+$', '', cmd).strip()   # remove surrounding backticks
        cmd = re.sub(r'^\*+|\*+$', '', cmd).strip()  # remove surrounding asterisks
        return cmd

    def _is_plausible_command(self, cmd: str) -> bool:
        """Return False if cmd looks like an assistant-mode response rather than a game command."""
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
            # Truncate long observations
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
