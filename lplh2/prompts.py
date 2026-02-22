"""Prompt templates from the LPLH paper (Tables 4-9).

All prompts are kept as close to the original paper as possible,
with minor formatting adjustments for programmatic use.
"""

# ─────────────────────────────────────────────────────────────
# Table 4: Action Validation
# Determines if the previous action was successful or not
# ─────────────────────────────────────────────────────────────
ACTION_VALIDATION_PROMPT = """You are evaluating the outcome of a text-based game action based on the game's observation (feedback message) after the player's previous action. Your task is to determine if the action was successful or not.

<START OF INSTRUCTIONS>
- You will be given an observation text that follows the player's attempted action.
- If the observation indicates that the action was carried out successfully (e.g., it provides new information, describes the environment, or gives a positive confirmation), respond with:
<ais> True </ais>
- If the observation indicates that the action could not be performed (e.g., includes phrases like "You can't..." or "You cannot..."), respond with:
<ais> False </ais>
Note:
- An unsuccessful action usually explicitly states that the player cannot do something, or that the action fails.
<END OF INSTRUCTIONS>

Previous Action: {action}
Observation: {observation}"""


# ─────────────────────────────────────────────────────────────
# Table 5: Relation Extraction
# Extracts (subject, relation, object) triples from observations
# ─────────────────────────────────────────────────────────────
RELATION_EXTRACTION_PROMPT = """<START OF INSTRUCTIONS>
You're going to extract triples in the format <subject, relation, object> from an input Observation along with previous actions you did, originating from a text-based game. Focus solely on where the character ('You') is located, what objects are in that location, and their immediate properties. The maximum length for any object name in the triples is three words, where length of location name has no limit.

Rules:
1. If the observation doesn't describe an environment or information is insufficient (e.g., "Opened", "Taken"), output |start| none |end| and skip other points.
2. Always use 'in' as the relation to represent the character's location. Convert any spatial descriptions (e.g., 'are facing', 'are standing', 'are behind') to the 'in' relation. If the input begins with a Room name (starts with a capital letter and does not end with a period), use it as the location.
Example:
Input: "Stairwell (First Floor) You're in the north stairwell."
Triple: <You, in, Stairwell (First Floor)>
3. If the observation doesn't include a precise location, do not provide any <You, in, *> triple.
4. Use 'have' as the relation to represent interactive objects present in the location. Focus only on the objects themselves as the 'obj' in the triple. Ignore decorative details unless they indicate an interactive object. Limit object names to a maximum of three words.
Example:
Input: "There is a small mailbox here."
Triple: <[Location], have, mailbox>
5. Do not include additional details or properties of objects. Only extract the objects themselves, ensuring object names are no longer than three words. But if a object have a relation to another object, such as 'in' and 'on', then extract that relation.
Example:
Input: "A buzzing water fountain has been moved."
Triple: "<[Location], have, water fountain>"
Input: "A sock is on the table."
Triple: "<[Location], have, sock>, <[Location], have, table>, <sock, on, table>"
6. If the input specifies a requirement or action needed to continue, use <location/object, need/require, something to action>.
Example:
Input: "Forest. You would need a machete to go further west."
Triple: <Forest, need, machete to go west>
7. For objects or locations mentioned with a direction (e.g., 'to the north', 'up to', 'down'), use <current location, direction, [new location]/to [direction]>.
Example:
Input: "Hall. To the southwest is the entrance to the Computer Site, and you can go east here as well as go up with a stair."
Triples: <Hall, southwest, Computer Site>, <Hall, east, to east>, <Hall, up, to up>
Note: Pay more attention to objects and directions than to objects' states or other decorative details.
Now, extract the relationships for the input step by step and merge all the results into a single output enclosed within |start| * |end|, where * represents the list of extracted triples.
<END OF INSTRUCTIONS>

Previous Action: {action}
Observation: {observation}"""


# ─────────────────────────────────────────────────────────────
# Table 6: Splitting Action into Verb + Objects
# ─────────────────────────────────────────────────────────────
ACTION_SPLITTING_PROMPT = """<START OF INSTRUCTIONS>
You will receive a previous input(step) from a text-based IF game, and please split the input into two parts, action and objs, as "<verb; [objs]>". Please follow these instructions to complete the task step by step.

Use the following rules:
1. If the action is a simple directional command (e.g., "north" or "n"), the object list should be empty.
For example:
Input: "west"
Response: "<act> <west; []> </act>"
2. If the action is "take all" or another "all" command (e.g., "take all"), treat "take all" as the verb and leave the object list empty.
For example:
Input: "drop all"
Response: "<act> <drop all; []> </act>"
3. If there are objects following the Verb (e.g., "eat", "take") or Verb phrase (e.g., "drop down", "go around"), list them. If prepositions (e.g., "on", "at", "with") are present, include them in the verb phrase using "&" as a placeholder, and list each noun object individually.

Final Output:
Use <act> <verb; [objs]> </act> format for final output where: "verb" represents the action phrase with placeholders "&" for objects. objs is a list of object nouns.
<END OF INSTRUCTIONS>

Input: "{action}" """


# ─────────────────────────────────────────────────────────────
# Table 8: Experience Summarization
# Called when score changes (gain or loss/death)
# ─────────────────────────────────────────────────────────────
EXPERIENCE_SUMMARIZATION_PROMPT = """<START OF INSTRUCTIONS>
You are a game engine summarizer. Your task is to read the current log of the game state and produce a concise, cohesive summary of the player's progress so far (This happens every time the player gets a score or loses a score). Do NOT reveal any hidden or undiscovered information. Focus only on details the player already knows or has directly experienced.

A list of "Step" will be provided. Each step includes:
- An observation (what the player sees),
- Info about moves and current score,
- The action taken just before the observation.

**Summary Structure:**
1. "location": where the player is (or what area is described) when the score changes. If the player has died, give the location name before death.
*1.1* - One Location name Only.
*1.2* - Description of situation.
2. "puzzle_status": what puzzles or obstacles have been solved to earn/lose the points.
*2.1* - ONLY related steps to solve the puzzles directly. Any requirement for solving the puzzles, such as 'player need to <step>open door<step> at Room1 to enter <loc>Room2<loc>.
*2.2* - Description of the puzzle.
3. "scoring": how the player earned/lost points for the last step. Any action leads to earning/losing points.
*3.1* - Step done to earn/lose points.
*3.2* - How many points are changed?
4. "important_experience": The experience can be used for the future. Only the most notable and valuable clues or items the player learned about for the global game experience or any warning must be recorded through all previous logs. Only Focus on confirmed information.
*Earn Points* - ONLY when player earn points, then we only need to know what leads to earn points and ignore other unchecked information.
*Lose Points* - ONLY when the player loses points (died usually or lost in the game), you also need to give suggestions for the future.

**Remember**:
- If no related puzzles are encountered, the whole 'puzzle_status' needs to be "No puzzles encountered yet."
- Please focus on how the player scored points with related puzzles and situations that occurred.
- Do not reveal hidden or undiscovered info.
- Keep it concise and factual based on the logs.
- When giving "important_experience", please reflect like an expert player.
- If player has not died, the '*Lose Points*' in 'important_experience' should be 'none'. If player has died, the '*Earn Points*' in 'important_experience' should be 'none'.
- In your reasoning, if you find more than one earning or losing points, please ONLY summarize the last one based on previous steps.

**Final Output Format:**
- In the final output for any 'loc name', please use <loc> loc name <loc> to mark it, as well as 'step did before' by marking in <step> step did <step>, as well as 'interacted obj' by marking in <obj> interacted obj <obj>.
- At the end of the response, please outline TAGs (no more than 4) between <tag> * </tag> that are used for retrieval. Put main location in <room> * </room> as one of the tag.
- After TAGs, please also give the difficulty for current puzzles in between <dif> * </dif>.
- Please think about it first. Then, give your final completed player experience summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Score change: {reward_change}
Current score: {current_score}

Game History:
{history}"""


# ─────────────────────────────────────────────────────────────
# Table 9: LPLH Action Generation
# The main prompt for generating the next game command
# ─────────────────────────────────────────────────────────────
LPLH_ACTION_GENERATION_PROMPT = """<START OF INSTRUCTIONS>
**Instructions for Generating a Next Command in Text-Based Interactive Fiction**
---
**Objective** Craft a single, context-aware **next command** with its motivation that propels the game forward, based on the current map, recent actions, and history of attempts. This command should represent one immediate player action.
---
**Principles for Exploration, Puzzle-Solving, and Earning Points**
1. **Analyze the Current Game State**
- **Room & Map Details**: Assess where you are, noting any exits, known layout, and significant objects.
- **Recent Attempts**: Reflect on the previous actions, the motivation of taking that action and observation after this attempt.
- **Inventory Check**: Identify items on hand (keys, tools, etc.) that might solve current puzzles or overcome obstacles.
- **Objects & Interactions**: Focus on confirmed items or directions. If uncertain leads might advance the game, consider them cautiously.
- **Action Selection**: Only choose to interact with an object (or perform an action) if you're confident it will move the story forward.
2. **Use Retrieved Experiences and Past Attempts**
- **Relevance**: Apply past successes or observed clues that align with the current room or situation.
- **Avoid Repetition**: Do not repeat failing commands indefinitely. If a command fails, adjust strategy.
- **Focus on Gains**: Prioritize moves likely to unlock new paths, uncover essential items, or yield valuable information.
3. **Formulate a Single Effective Command**
- **One Action**: Provide exactly one executable command.
- **Purpose**: Briefly ensure it's the most logical next step, considering both context and success likelihood.
- **Move command**: The full directions are ['north', 'south', 'east', 'west', 'southeast', 'southwest', 'northeast', 'northwest', 'up', 'down']
4. **Output Format**
- Present the final command and a short motivation in the following format without extra commentary:

Your internal reasoning steps Here.
|start|
<com>[command]</com>
<rea>[short motivation for the decision-making reason]</rea>
|end|

---
**Adaptation and Fallback Rules**
1. **Priority Usage**
- **Highest Priority**: Items in 'temp_have'.
- **Next**: Options in 'may_direction' or 'may_have'.
- **Then**: Verified directions ('direction') or items ('have').
2. **Conflict Resolution**
- Disregard prior attempts known to fail at this location or context.
- Validate uncertain ('may_') directions or items before fully committing to them.
- After verify all the exits in one room then you can fully trust the map.
3. **Fallback Strategies**
- If uncertain, explore unvisited areas or re-examine ('look') the current room.
- Look for overlooked clues or alternative ways forward.
4. **Exploratory Commands**
- If tools are available, think of how to use them on obstacles.
- In case an exploration fails, attempt a different angle—return to a previous room, look around again, or try another approach.
- **Explore the world**: It's better to try all directions in each room to identify the exit and update the game map. For 'may_direction', consider testing that path (e.g., "north").
---
**Remember**: You are navigating a text-based world. Combine current observations with past knowledge to decide the best single move.
<END OF INSTRUCTIONS>

=== CURRENT GAME MAP ===
{kg_map}

=== AVAILABLE ACTIONS FOR OBJECTS HERE ===
{action_pairs}

=== RETRIEVED EXPERIENCES ===
{experiences}

=== RECENT HISTORY (last {history_length} turns) ===
{history}

=== CURRENT OBSERVATION ===
{observation}"""


# ─────────────────────────────────────────────────────────────
# LPLH2 Enhancement: Neutral State Experience Prompts
# Four separate prompts, one per neutral trigger type.
# These fire when reward_change == 0 but a meaningful event occurred.
# ─────────────────────────────────────────────────────────────

# Trigger 1: Agent enters a previously unvisited location
NAVIGATION_EXPERIENCE_PROMPT = """<START OF INSTRUCTIONS>
You are summarizing a navigation event in a text-based game. The player has just entered a location they have never visited before. Capture the essential spatial and object information so it can be used as a reference in future playthroughs.

**Summary Structure:**
1. "location": The name of the newly discovered room.
2. "arrived_from": Where the player came from and what action/direction was used to get here.
3. "objects_present": A concise list of objects or features noticed in this room.
4. "exits_known": Any exits or directions mentioned in the observation.
5. "important_experience": Anything notable — locked doors, dangerous areas, useful items, hints — that could matter later.

**Remember:**
- Only record what is directly stated in the observation.
- Do not invent or speculate about content not mentioned.
- Keep it concise and factual.

**Output Format:**
- Mark location names with <loc> loc name <loc>, actions with <step> action <step>, objects with <obj> object <obj>.
- End with retrieval tags (max 4) between <tag> * </tag>. Include the room name as <room> * </room>.
- Think first, then give the final summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Current Location (newly entered): {location}
Previous Location: {prev_location}
Action Taken: {action}
Observation: {observation}"""


# Trigger 2: Agent examines an object, reads a note, or talks to someone
NARRATIVE_EXPERIENCE_PROMPT = """<START OF INSTRUCTIONS>
You are summarizing an information-retrieval event in a text-based game. The player just examined, read, or inspected something that revealed meaningful content. Capture the key information for future reference.

**Summary Structure:**
1. "location": Where this happened.
2. "source": What was examined, read, or interacted with.
3. "content": The key information that was revealed (be concise — only what was actually shown).
4. "implication": What this information suggests about puzzles, progression, or dangers ahead.
5. "important_experience": Any specific clues, warnings, or instructions that could be directly useful in a future playthrough.

**Remember:**
- Only record content explicitly stated in the observation.
- "nothing special" or empty responses are not worth summarising — do not call this for those.
- Focus on actionable information.

**Output Format:**
- Mark location names with <loc> loc name <loc>, actions with <step> action <step>, objects with <obj> object <obj>.
- End with retrieval tags (max 4) between <tag> * </tag>. Include the room name as <room> * </room>.
- Think first, then give the final summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Current Location: {location}
Action Taken: {action}
Observation: {observation}"""


# Trigger 3: An action changes the environment (opens a path, toggles a switch, etc.)
ENVIRONMENTAL_CHANGE_PROMPT = """<START OF INSTRUCTIONS>
You are summarizing an environmental change event in a text-based game. The player just performed an action that altered the game world — such as opening a door, moving an object, or activating a mechanism — without directly earning points. Capture what changed and why it matters.

**Summary Structure:**
1. "location": Where this happened.
2. "trigger_action": The exact action that caused the change.
3. "change_description": What specifically changed in the environment.
4. "new_possibility": What new options, paths, or interactions this change enables.
5. "important_experience": The key lesson — what to remember about this action and in what context it is useful for future playthroughs.

**Remember:**
- Focus on what is now possible that was not possible before.
- Be specific about the action that caused the change.
- Do not speculate beyond what the observation confirms.

**Output Format:**
- Mark location names with <loc> loc name <loc>, actions with <step> action <step>, objects with <obj> object <obj>.
- End with retrieval tags (max 4) between <tag> * </tag>. Include the room name as <room> * </room>.
- Think first, then give the final summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Current Location: {location}
Action Taken: {action}
Observation: {observation}"""


# Trigger 4: Agent finds a valid command after 2+ consecutive failures
ERROR_CORRECTION_PROMPT = """<START OF INSTRUCTIONS>
You are summarising a command-discovery event in a text-based game. After several failed attempts, the player found a command that the game understood and accepted. Capture the correct syntax pattern so it can be reused in similar future situations.

**Summary Structure:**
1. "location": Where this happened.
2. "correct_command": The exact command that succeeded.
3. "failed_attempts": The commands that were tried and rejected before success.
4. "pattern_learned": The general rule or syntax this reveals (e.g., "use 'go through X' not 'enter X'", "use 'climb tree' not 'go up tree'").
5. "important_experience": When and how to apply this command pattern in similar future situations.

**Remember:**
- The goal is to extract a reusable command pattern, not just record the single event.
- Be specific about what was wrong with the failed attempts vs. what made the correct one work.

**Output Format:**
- Mark location names with <loc> loc name <loc>, actions with <step> action <step>, objects with <obj> object <obj>.
- End with retrieval tags (max 4) between <tag> * </tag>. Include the room name as <room> * </room>.
- Think first, then give the final summary between '|start|' and '|end|'.
<END OF INSTRUCTIONS>

Current Location: {location}
Successful Command: {action}
Observation: {observation}
Recent Failed Commands: {failed_attempts}"""


# ─────────────────────────────────────────────────────────────
# Table 7: Baseline Action Generation (for comparison)
# ─────────────────────────────────────────────────────────────
BASELINE_ACTION_GENERATION_PROMPT = """You are playing the classic text-based interactive fiction game. Your goal is to explore, solve puzzles, collect treasures, and reach the winning end state. Throughout the game, you will:
1. Receive a history of the game's the action you performed, the new observation representing what you see or experience after your action.
2. Have access only to the last 10 turns of conversation as your history.
3. Receive current new observation based on the last action and the current game states as input.
4. Produce all responses formatted between "|start|" and "|end|".

**Your Task:**
- At each turn, carefully read the provided new observation and the action you performed.
- Use your internal chain-of-thought to determine the best possible action to advance in the game.
- Once you have reasoned through your options, produce exactly ONE game command.
- Always Format your command as this at the end of your response:
**Final Command:**
|start| [your chosen command] |end|

**Guidelines:**
- Avoid random or nonsensical actions.
- Try to use player (human) logic to guide your decision.
- You can Use 'look' command to examine the current location. And 'inventory' command to examine your inventory.
- Maintain continuity by leveraging the last 10 turns of conversation.
- Always think first, then act.

=== RECENT HISTORY ===
{history}

=== CURRENT OBSERVATION ===
{observation}"""
