"""Module 2: Action Space Learning.

Tracks all validated verb-object pairings discovered during gameplay.
When an action is confirmed valid, it is decomposed into verb + objects
and stored. During decision-making, known verbs are paired with current
location objects to suggest candidate actions.
"""

import logging

logger = logging.getLogger(__name__)


class ActionSpace:
    """Learns and maintains the valid action space.
    
    Actions are decomposed into verb templates (with & placeholders)
    and associated objects. Example:
        "take sword" -> verb="take &", objects=["sword"]
        "put key in box" -> verb="put & in &", objects=["key", "box"]
    """

    def __init__(self):
        # {verb_template: set_of_objects}
        # e.g. {"take &": {"sword", "lamp"}, "open &": {"door", "mailbox"}}
        self.verbs = {}
        self.total_actions_learned = 0

    def reset(self):
        """Reset action space for a new game."""
        self.verbs = {}
        self.total_actions_learned = 0

    def store_action(self, verb: str, objects: list):
        """Store a validated verb-object pairing.
        
        Args:
            verb: The verb template (e.g., "take &", "north")
            objects: List of objects associated with this action
        """
        verb = verb.strip().lower()
        if not verb:
            return

        if verb not in self.verbs:
            self.verbs[verb] = set()

        for obj in objects:
            obj_clean = obj.strip().lower()
            if obj_clean and obj_clean not in self.verbs[verb]:
                self.verbs[verb].add(obj_clean)
                self.total_actions_learned += 1
                logger.debug(f"Learned action: {verb} -> {obj_clean}")

        # Even verbs with no objects should be stored (e.g., "look", "north")
        if not objects:
            self.total_actions_learned += 1

    def get_action_pairs(self, current_objects: list) -> list:
        """Generate possible action-object pairs for the current location.
        
        This is the pairing(objloc, AS) function from the paper (Equation 4).
        Matches current location's objects with known verb templates.
        
        Args:
            current_objects: List of object names in the current location
            
        Returns:
            List of strings like "take sword", "open mailbox", etc.
        """
        pairs = []
        objects_lower = [o.strip().lower() for o in current_objects]

        for verb, known_objs in self.verbs.items():
            if "&" not in verb:
                # No-object verb (directions, look, inventory, etc.)
                continue

            # Check which current objects match this verb's known objects
            for obj in objects_lower:
                if obj in known_objs:
                    # Generate the concrete action
                    concrete = verb.replace("&", obj, 1)
                    pairs.append(concrete)

        return pairs

    def to_prompt_string(self, current_objects: list) -> str:
        """Serialize action pairings for inclusion in the prompt.
        
        Args:
            current_objects: Objects in the current location
        """
        pairs = self.get_action_pairs(current_objects)
        
        if not pairs:
            return "No known actions for objects in this location yet. Try exploring!"

        output = ["Known valid actions for objects here:"]
        for pair in pairs:
            output.append(f"  - {pair}")

        # Also list all known verbs for reference
        output.append("")
        output.append(f"All learned verbs ({len(self.verbs)} total):")
        for verb in sorted(self.verbs.keys()):
            output.append(f"  - {verb}")

        return "\n".join(output)

    def num_actions(self) -> int:
        """Total number of unique verb-object pairs learned."""
        return self.total_actions_learned
