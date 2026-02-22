"""LLM Client - Unified wrapper for LLM calls.

Supports Ollama (local, free) and OpenAI (API, paid).
All LPLH modules use this client for their LLM needs.
"""

import re
import json
import logging
from . import config
from .prompts import (
    ACTION_VALIDATION_PROMPT,
    RELATION_EXTRACTION_PROMPT,
    ACTION_SPLITTING_PROMPT,
    EXPERIENCE_SUMMARIZATION_PROMPT,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client that supports multiple providers."""

    def __init__(self, provider=None, model=None, temperature=None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.LLM_MODEL
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE

        if self.provider == "ollama":
            import ollama
            self._ollama = ollama.Client(host=config.OLLAMA_BASE_URL)
        elif self.provider == "openai":
            from openai import OpenAI
            self._openai = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

        logger.info(f"LLM Client initialized: provider={self.provider}, model={self.model}")

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = None,
             think: bool = False) -> str:
        """Send a chat request and return the response text."""
        temp = temperature if temperature is not None else self.temperature

        if self.provider == "ollama":
            return self._chat_ollama(system_prompt, user_prompt, temp, think=think)
        elif self.provider == "openai":
            return self._chat_openai(system_prompt, user_prompt, temp)

    def _chat_ollama(self, system_prompt: str, user_prompt: str, temperature: float,
                     think: bool = False) -> str:
        """Chat via Ollama."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self._ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature},
                think=think,
            )
        except Exception as e:
            if think and "does not support thinking" in str(e):
                # Non-thinking model (e.g. qwen2.5) — retry without think flag
                response = self._ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": temperature},
                )
            else:
                raise
        return response["message"]["content"]

    def _chat_openai(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Chat via OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    # ── Fine-tuned model replacement tasks ─────────────────────

    def validate_action(self, action: str, observation: str) -> bool:
        """Check if the previous action was valid (Table 4 prompt).
        
        Returns True if the action was successful, False otherwise.
        """
        prompt = ACTION_VALIDATION_PROMPT.format(action=action, observation=observation)
        response = self.chat("", prompt, temperature=config.FM_TEMPERATURE)

        # Parse <ais> True </ais> or <ais> False </ais>
        ais_match = re.search(r"<ais>\s*(True|False)\s*</ais>", response, re.IGNORECASE)
        if ais_match:
            return ais_match.group(1).strip().lower() == "true"

        # Fallback: check for common failure phrases in observation
        fail_phrases = ["you can't", "you cannot", "that's not", "i don't understand",
                        "doesn't seem", "nothing happens", "you don't see"]
        obs_lower = observation.lower()
        return not any(phrase in obs_lower for phrase in fail_phrases)

    def extract_relations(self, action: str, observation: str) -> list:
        """Extract (subject, relation, object) triples from observation (Table 5 prompt).
        
        Returns a list of tuples: [(subject, relation, object), ...]
        """
        prompt = RELATION_EXTRACTION_PROMPT.format(action=action, observation=observation)
        response = self.chat("", prompt, temperature=config.FM_TEMPERATURE)

        # Parse |start| ... |end| block
        match = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        if not match:
            return []

        content = match.group(1).strip()
        if content.lower() == "none":
            return []

        # Extract <subject, relation, object> triples
        triples = re.findall(r"<([^>]+)>", content)
        result = []
        for triple in triples:
            parts = [p.strip() for p in triple.split(",")]
            if len(parts) == 3:
                result.append((parts[0], parts[1], parts[2]))

        return result

    def split_action(self, action: str) -> dict:
        """Decompose action into verb + objects (Table 6 prompt).
        
        Returns: {"verb": "take &", "objects": ["sword"]}
        """
        prompt = ACTION_SPLITTING_PROMPT.format(action=action)
        response = self.chat("", prompt, temperature=config.FM_TEMPERATURE)

        # Parse <act> <verb; [objs]> </act>
        act_match = re.search(r"<act>\s*<([^;]+);\s*\[([^\]]*)\]>\s*</act>", response)
        if act_match:
            verb = act_match.group(1).strip()
            objs_str = act_match.group(2).strip()
            objects = [o.strip().strip("'\"") for o in objs_str.split(",") if o.strip()] if objs_str else []
            return {"verb": verb, "objects": objects}

        # Fallback: simple heuristic split
        parts = action.strip().split()
        if len(parts) == 1:
            return {"verb": parts[0], "objects": []}
        else:
            return {"verb": parts[0] + " &", "objects": [" ".join(parts[1:])]}

    def summarize_experience(self, history: str, reward_change: int, 
                              current_score: int) -> str:
        """Generate a structured experience summary (Table 8 prompt).
        
        Returns the raw summarized experience text.
        """
        prompt = EXPERIENCE_SUMMARIZATION_PROMPT.format(
            history=history,
            reward_change=reward_change,
            current_score=current_score,
        )
        response = self.chat("", prompt, temperature=self.temperature)

        # Extract |start| ... |end| block if present
        match = re.search(r"\|start\|\s*(.*?)\s*\|end\|", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()
