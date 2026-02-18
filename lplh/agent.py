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

        # ── Step 1: Update KG-map with relation extraction ────
        if self.prev_action:
            try:
                triples = self.llm.extract_relations(self.prev_action, observation)
                self.kg_map.update(triples, self.prev_action)
                logger.debug(f"KG-map updated with {len(triples)} triples")
            except Exception as e:
                logger.warning(f"Relation extraction failed: {e}")

        # ── Step 2: Validate previous action & store in action space ──
        if self.prev_action:
            try:
                is_valid = self.llm.validate_action(self.prev_action, observation)
                if is_valid:
                    split = self.llm.split_action(self.prev_action)
                    self.action_space.store_action(split["verb"], split["objects"])
                    logger.debug(f"Valid action stored: {split}")
            except Exception as e:
                logger.warning(f"Action validation/splitting failed: {e}")

        # ── Step 3: Summarize experience on score change ──────
        if reward_change != 0 and self.prev_action:
            try:
                history_text = self._format_history()
                exp_summary = self.llm.summarize_experience(
                    history=history_text,
                    reward_change=reward_change,
                    current_score=score,
                )
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

        # ── Step 4: Retrieve relevant experiences ─────────────
        query = f"Location: {self.kg_map.current_location}. Observation: {observation[:200]}"
        experiences = self.experience_lib.retrieve_relevant(query)

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

        try:
            response = self.llm.chat(
                system_prompt="You are an expert player of text-based interactive fiction games.",
                user_prompt=prompt,
            )
            command = self._parse_command(response)
        except Exception as e:
            logger.error(f"Action generation failed: {e}")
            command = "look"  # Safe fallback

        # ── Update state ──────────────────────────────────────
        self.history.append((command, observation))
        # Keep only last HISTORY_LENGTH turns
        if len(self.history) > config.HISTORY_LENGTH:
            self.history = self.history[-config.HISTORY_LENGTH:]
        self.prev_action = command
        self.prev_score = score
        self.total_score = score

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
            return com_match.group(1).strip()

        # Try |start| ... |end| pattern (baseline format)
        start_match = re.search(r"\|start\|\s*(.+?)\s*\|end\|", response, re.DOTALL)
        if start_match:
            text = start_match.group(1).strip()
            # Remove any remaining tags
            text = re.sub(r"<[^>]+>", "", text).strip()
            # Take only the first line
            return text.split("\n")[0].strip()

        # Fallback: take the last meaningful line
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        if lines:
            last = lines[-1]
            # Remove common prefixes
            last = re.sub(r"^(Final Command:|Command:)\s*", "", last, flags=re.IGNORECASE)
            return last.strip()

        return "look"  # absolute fallback

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
