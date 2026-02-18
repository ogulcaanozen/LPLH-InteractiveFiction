"""Module 1: Dynamic Knowledge Graph Map.

Builds and maintains a knowledge graph of the game world,
tracking locations, objects, directions, and relationships.
Updated after every step via LLM-based relation extraction.
"""

import json
import copy
import logging

logger = logging.getLogger(__name__)


class KGMap:
    """Dynamic Knowledge Graph Map for spatial reasoning and memory.
    
    Stores the game world as a graph:
    - Nodes = locations (rooms)
    - Edges = directional connections between rooms  
    - Properties = objects, requirements per location
    """

    def __init__(self):
        self.nodes = {}           # {location_name: {objects, directions, ...}}
        self.current_location = None
        self.visited_rooms = []
        self.inventory = []       # items the player is carrying

    def reset(self):
        """Reset the KG-map for a new game run."""
        self.nodes = {}
        self.current_location = None
        self.visited_rooms = []
        self.inventory = []

    def update(self, triples: list, action: str = ""):
        """Update the knowledge graph with extracted triples.
        
        Args:
            triples: List of (subject, relation, object) tuples from LLM extraction
            action: The action that was just taken (for context)
        """
        if not triples:
            return

        new_location = None

        for subj, rel, obj in triples:
            rel_lower = rel.strip().lower()
            subj_clean = subj.strip()
            obj_clean = obj.strip()

            # Handle location updates: <You, in, Location>
            if subj_clean.lower() == "you" and rel_lower == "in":
                new_location = obj_clean
                self._ensure_node(new_location)

            # Handle objects in location: <Location, have, object>
            elif rel_lower == "have":
                loc = subj_clean if subj_clean != "[Location]" else self.current_location
                if loc:
                    self._ensure_node(loc)
                    if obj_clean not in self.nodes[loc]["have"]:
                        self.nodes[loc]["have"].append(obj_clean)

            # Handle directional connections: <Location, direction, Destination>
            elif rel_lower in self._direction_set():
                loc = subj_clean if subj_clean != "[Location]" else self.current_location
                if loc:
                    self._ensure_node(loc)
                    self.nodes[loc]["direction"][rel_lower] = obj_clean

            # Handle requirements: <Location, need/require, action>
            elif rel_lower in ("need", "require"):
                loc = subj_clean if subj_clean != "[Location]" else self.current_location
                if loc:
                    self._ensure_node(loc)
                    if obj_clean not in self.nodes[loc].get("needs", []):
                        self.nodes[loc].setdefault("needs", []).append(obj_clean)

            # Handle object-to-object relations: <obj1, on/in, obj2>
            elif rel_lower in ("on", "in") and subj_clean.lower() != "you":
                if self.current_location:
                    self._ensure_node(self.current_location)
                    for item in [subj_clean, obj_clean]:
                        if item not in self.nodes[self.current_location]["have"]:
                            self.nodes[self.current_location]["have"].append(item)

        # Update current location if changed
        if new_location:
            self.current_location = new_location
            if new_location not in self.visited_rooms:
                self.visited_rooms.append(new_location)

        # Handle inventory changes from actions
        action_lower = action.lower().strip()
        if action_lower.startswith("take ") or action_lower.startswith("get "):
            item = action[5:].strip() if action_lower.startswith("take ") else action[4:].strip()
            if item and item not in self.inventory:
                self.inventory.append(item)
            # Remove from room objects
            if self.current_location and self.current_location in self.nodes:
                objs = self.nodes[self.current_location]["have"]
                self.nodes[self.current_location]["have"] = [o for o in objs if o.lower() != item.lower()]
        elif action_lower.startswith("drop "):
            item = action[5:].strip()
            self.inventory = [i for i in self.inventory if i.lower() != item.lower()]

    def _ensure_node(self, location: str):
        """Create a node if it doesn't exist."""
        if location not in self.nodes:
            self.nodes[location] = {
                "have": [],           # confirmed objects
                "direction": {},      # confirmed exits {dir: destination}
                "may_have": [],       # uncertain objects
                "may_direction": [],  # uncertain exits
                "needs": [],          # requirements
            }

    def _direction_set(self):
        """All valid direction strings."""
        return {
            "north", "south", "east", "west",
            "northeast", "northwest", "southeast", "southwest",
            "up", "down", "n", "s", "e", "w",
            "ne", "nw", "se", "sw",
        }

    def get_current_room_info(self) -> dict:
        """Get objects and directions for the current location."""
        if not self.current_location or self.current_location not in self.nodes:
            return {"location": self.current_location, "objects": [], "directions": {}}
        
        node = self.nodes[self.current_location]
        return {
            "location": self.current_location,
            "objects": node["have"],
            "directions": node["direction"],
            "may_direction": node.get("may_direction", []),
            "needs": node.get("needs", []),
        }

    def to_prompt_string(self) -> str:
        """Serialize the KG-map for inclusion in the LLM prompt."""
        if not self.nodes:
            return "Map is empty. Start exploring!"

        output = []
        output.append(f"Current Location: {self.current_location or 'Unknown'}")
        output.append(f"Visited Rooms: {', '.join(self.visited_rooms)}")
        output.append(f"Inventory: {', '.join(self.inventory) if self.inventory else 'Empty'}")
        output.append("")

        for loc, data in self.nodes.items():
            marker = " (YOU ARE HERE)" if loc == self.current_location else ""
            output.append(f"[{loc}]{marker}")
            if data["have"]:
                output.append(f"  have: {', '.join(data['have'])}")
            if data["direction"]:
                dirs = [f"{d} -> {dest}" for d, dest in data["direction"].items()]
                output.append(f"  direction: {', '.join(dirs)}")
            if data.get("may_direction"):
                output.append(f"  may_direction: {', '.join(data['may_direction'])}")
            if data.get("needs"):
                output.append(f"  needs: {', '.join(data['needs'])}")
            output.append("")

        return "\n".join(output)

    def to_dict(self) -> dict:
        """Export KG-map as a dictionary (for saving)."""
        return {
            "nodes": copy.deepcopy(self.nodes),
            "current_location": self.current_location,
            "visited_rooms": list(self.visited_rooms),
            "inventory": list(self.inventory),
        }

    def num_rooms(self) -> int:
        """Number of discovered rooms."""
        return len(self.visited_rooms)
