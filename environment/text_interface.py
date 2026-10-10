"""Text interface between the blicket environment and LLM agents.

Converts environment states to natural language prompts and parses
LLM text responses back into environment actions.
"""

from __future__ import annotations

import re

from .blicket_env import Action, BlicketEnv, CausalRuleType, EnvState, Observation
from .blicket_env_5obj import Action5, BlicketEnv5, EnvState5
from .blicket_env_6obj import Action6, BlicketEnv6, EnvState6


# Action display names and keywords for parsing
ACTION_INFO = {
    Action.PLACE_A: {"name": "Place object A on detector", "keywords": ["place a", "put a"]},
    Action.PLACE_B: {"name": "Place object B on detector", "keywords": ["place b", "put b"]},
    Action.PLACE_C: {"name": "Place object C on detector", "keywords": ["place c", "put c"]},
    Action.REMOVE_A: {"name": "Remove object A from detector", "keywords": ["remove a", "take a"]},
    Action.REMOVE_B: {"name": "Remove object B from detector", "keywords": ["remove b", "take b"]},
    Action.REMOVE_C: {"name": "Remove object C from detector", "keywords": ["remove c", "take c"]},
    Action.CHECK: {"name": "Submit your answer", "keywords": ["check", "submit", "answer"]},
}


class TextInterface:
    """Renders environment states as text and parses LLM responses into actions."""

    def render_state(self, env: BlicketEnv) -> str:
        """Render the current environment state as text for the LLM."""
        state = env.state
        valid_actions = env.get_valid_actions()

        lines = []
        lines.append("=== Blicket Detector Status ===")

        # Objects on detector
        if state.objects_on_detector:
            on_det = ", ".join(sorted(state.objects_on_detector))
            lines.append(f"Objects on detector: {on_det}")
        else:
            lines.append("Objects on detector: (none)")

        # Detector status
        status = "ACTIVE (glowing)" if state.detector_active else "INACTIVE (not glowing)"
        lines.append(f"Detector: {status}")

        lines.append(f"Steps taken: {state.step_count}/{env.max_steps}")
        lines.append("")

        # Available actions as numbered list
        lines.append("Available actions:")
        for i, action in enumerate(valid_actions, 1):
            lines.append(f"  {i}. {ACTION_INFO[action]['name']}")

        lines.append("")
        lines.append(
            "Choose an action by number or name. "
            "When ready to answer, choose 'Submit your answer'."
        )

        return "\n".join(lines)

    def render_observation(self, obs: Observation) -> str:
        """Render an observation as a natural language description."""
        return obs.message

    def render_task_description(self) -> str:
        """Render the initial task description for the agent."""
        return (
            "You are interacting with a 'blicket detector' — a device that glows "
            "when certain objects (called 'blickets') are placed on it.\n\n"
            "There are 3 objects: A, B, and C. Some may be blickets, some may not.\n\n"
            "The detector follows a hidden rule:\n"
            "- CONJUNCTIVE: ALL blicket objects must be on the detector for it to activate.\n"
            "- DISJUNCTIVE: ANY single blicket object is enough to activate the detector.\n\n"
            "Your goal: Figure out which objects are blickets AND whether the rule is "
            "conjunctive or disjunctive.\n\n"
            "You can place objects, remove objects, and observe whether the detector "
            "activates. When you're confident, submit your answer.\n\n"
            "Respond with:\n"
            "REASONING: <your current thinking>\n"
            "ACTION: <number or action name>"
        )

    def render_answer_prompt(self) -> str:
        """Render the prompt for the agent to submit their answer."""
        return (
            "You chose to submit your answer. Please provide your conclusion.\n\n"
            "Respond with:\n"
            "RULE_TYPE: conjunctive OR disjunctive\n"
            "BLICKETS: <comma-separated list of blicket objects, e.g. A, B>"
        )

    def parse_action(self, response: str, valid_actions: list[Action]) -> Action | None:
        """Parse an LLM response into an Action.

        Uses a cascade parser:
        1. Look for ACTION: N (numbered)
        2. Look for ACTION: keyword
        3. Keyword extraction from full text
        4. Full-text scan for any action keyword

        Returns None on parse failure (triggers retry).
        """
        text = response.strip()

        # Strategy 1: Look for "ACTION: N" pattern
        match = re.search(r"ACTION:\s*(\d+)", text, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(valid_actions):
                return valid_actions[idx]

        # Strategy 2: Look for "ACTION: keyword" pattern
        match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            action_text = match.group(1).strip().lower()
            action = self._match_keywords(action_text, valid_actions)
            if action:
                return action

        # Strategy 3: Check last line for action keywords
        last_line = text.strip().split("\n")[-1].lower()
        action = self._match_keywords(last_line, valid_actions)
        if action:
            return action

        # Strategy 4: Full text scan
        action = self._match_keywords(text.lower(), valid_actions)
        if action:
            return action

        return None

    def _match_keywords(self, text: str, valid_actions: list[Action]) -> Action | None:
        """Match text against action keywords, restricted to valid actions."""
        for action in valid_actions:
            for keyword in ACTION_INFO[action]["keywords"]:
                if keyword in text:
                    return action
        return None

    def parse_answer(self, response: str) -> tuple[CausalRuleType | None, set[str] | None]:
        """Parse the agent's answer from free text.

        Returns:
            Tuple of (rule_type, required_objects) or (None, None) on parse failure.
        """
        text = response.strip()

        # Parse rule type
        rule_type = None
        match = re.search(r"RULE_TYPE:\s*(\w+)", text, re.IGNORECASE)
        if match:
            rt = match.group(1).lower()
            if "conjunct" in rt:
                rule_type = CausalRuleType.CONJUNCTIVE
            elif "disjunct" in rt:
                rule_type = CausalRuleType.DISJUNCTIVE
        else:
            # Fallback: look for the words anywhere
            lower = text.lower()
            if "conjunctive" in lower and "disjunctive" not in lower:
                rule_type = CausalRuleType.CONJUNCTIVE
            elif "disjunctive" in lower and "conjunctive" not in lower:
                rule_type = CausalRuleType.DISJUNCTIVE

        # Parse blickets
        objects = None
        match = re.search(r"BLICKETS?:\s*([A-Za-z,\s]+)", text, re.IGNORECASE)
        if match:
            obj_text = match.group(1).upper()
            objects = {o.strip() for o in obj_text.split(",") if o.strip() in {"A", "B", "C"}}
        else:
            # Fallback: look for "blickets are A and B" patterns
            match = re.search(
                r"blickets?\s+(?:are|is)\s+([A-C](?:\s*(?:,|and)\s*[A-C])*)",
                text,
                re.IGNORECASE,
            )
            if match:
                obj_text = match.group(1).upper()
                objects = {c for c in obj_text if c in {"A", "B", "C"}}

        if rule_type is None or objects is None or len(objects) == 0:
            return None, None

        return rule_type, objects


# --- 5-Object Interface ---

ACTION_INFO_5 = {
    Action5.PLACE_A: {"name": "Place object A on detector", "keywords": ["place a", "put a"]},
    Action5.PLACE_B: {"name": "Place object B on detector", "keywords": ["place b", "put b"]},
    Action5.PLACE_C: {"name": "Place object C on detector", "keywords": ["place c", "put c"]},
    Action5.PLACE_D: {"name": "Place object D on detector", "keywords": ["place d", "put d"]},
    Action5.PLACE_E: {"name": "Place object E on detector", "keywords": ["place e", "put e"]},
    Action5.REMOVE_A: {"name": "Remove object A from detector", "keywords": ["remove a", "take a"]},
    Action5.REMOVE_B: {"name": "Remove object B from detector", "keywords": ["remove b", "take b"]},
    Action5.REMOVE_C: {"name": "Remove object C from detector", "keywords": ["remove c", "take c"]},
    Action5.REMOVE_D: {"name": "Remove object D from detector", "keywords": ["remove d", "take d"]},
    Action5.REMOVE_E: {"name": "Remove object E from detector", "keywords": ["remove e", "take e"]},
    Action5.CHECK: {"name": "Submit your answer", "keywords": ["check", "submit", "answer"]},
}


class TextInterface5:
    """Renders 5-object environment states as text and parses LLM responses."""

    def __init__(
        self,
        include_order_sensitive: bool = False,
        include_order_stochastic: bool = False,
    ) -> None:
        self.include_order_sensitive = include_order_sensitive
        self.include_order_stochastic = include_order_stochastic

    def render_state(self, env: BlicketEnv5) -> str:
        """Render the current environment state as text for the LLM."""
        state = env.state
        valid_actions = env.get_valid_actions()

        lines = []
        lines.append("=== Blicket Detector Status ===")

        if state.objects_on_detector:
            on_det = ", ".join(sorted(state.objects_on_detector))
            lines.append(f"Objects on detector: {on_det}")
        else:
            lines.append("Objects on detector: (none)")

        status = "ACTIVE (glowing)" if state.detector_active else "INACTIVE (not glowing)"
        lines.append(f"Detector: {status}")

        lines.append(f"Steps taken: {state.step_count}/{env.max_steps}")
        lines.append("")

        lines.append("Available actions:")
        for i, action in enumerate(valid_actions, 1):
            lines.append(f"  {i}. {ACTION_INFO_5[action]['name']}")

        lines.append("")
        lines.append(
            "Choose an action by number or name. "
            "When ready to answer, choose 'Submit your answer'."
        )

        return "\n".join(lines)

    def render_observation(self, obs: Observation) -> str:
        """Render an observation as a natural language description."""
        return obs.message

    def render_task_description(self) -> str:
        """Render the initial task description for the 5-object agent."""
        base = (
            "You are interacting with a 'blicket detector' — a device that glows "
            "when certain objects (called 'blickets') are placed on it.\n\n"
            "There are 5 objects: A, B, C, D, and E. Some may be blickets, some may not.\n\n"
            "The detector follows a hidden rule:\n"
            "- CONJUNCTIVE: ALL blicket objects must be on the detector for it to activate.\n"
            "- DISJUNCTIVE: ANY single blicket object is enough to activate the detector.\n"
        )

        if self.include_order_stochastic:
            base += (
                "- ORDER_SENSITIVE: ALL blicket objects must be on the detector AND they "
                "must have been placed in a specific sequence (could be 2, 3, 4, or even "
                "5 objects in a required order).\n"
                "- STOCHASTIC: The correct configuration activates the detector only SOME "
                "of the time (probabilistic activation).\n"
                "- ORDER_SENSITIVE_STOCHASTIC: ALL blicket objects must be in the correct "
                "sequence AND activation is probabilistic (may not activate every time even "
                "with the correct configuration and order).\n"
            )
        elif self.include_order_sensitive:
            base += (
                "- ORDER_SENSITIVE: ALL blicket objects must be on the detector AND they "
                "must have been placed in a specific sequence (could be 2, 3, 4, or even "
                "5 objects in a required order).\n"
            )

        base += (
            "\nYour goal: Figure out which objects are blickets AND whether the rule is "
            "conjunctive"
        )
        if self.include_order_stochastic:
            base += ", disjunctive, order_sensitive, stochastic, or order_sensitive_stochastic"
        elif self.include_order_sensitive:
            base += ", disjunctive, or order_sensitive"
        else:
            base += " or disjunctive"

        base += (
            ".\n\n"
            "You can place objects, remove objects, and observe whether the detector "
            "activates. When you're confident, submit your answer.\n\n"
            "Respond with:\n"
            "REASONING: <your current thinking>\n"
            "ACTION: <number or action name>"
        )

        return base

    def render_answer_prompt(self) -> str:
        """Render the prompt for the agent to submit their answer."""
        if self.include_order_stochastic:
            return (
                "You chose to submit your answer. Please provide your conclusion.\n\n"
                "Respond with:\n"
                "RULE_TYPE: conjunctive OR disjunctive OR order_sensitive OR stochastic OR order_sensitive_stochastic\n"
                "BLICKETS: <comma-separated list of blicket objects, e.g. A, B>\n"
                "ORDER: <comma-separated sequence if order_sensitive or order_sensitive_stochastic, e.g. A, D, C>"
            )
        if self.include_order_sensitive:
            return (
                "You chose to submit your answer. Please provide your conclusion.\n\n"
                "Respond with:\n"
                "RULE_TYPE: conjunctive OR disjunctive OR order_sensitive\n"
                "BLICKETS: <comma-separated list of blicket objects, e.g. A, B>\n"
                "ORDER: <comma-separated sequence if order_sensitive, e.g. A, D, C>"
            )
        return (
            "You chose to submit your answer. Please provide your conclusion.\n\n"
            "Respond with:\n"
            "RULE_TYPE: conjunctive OR disjunctive\n"
            "BLICKETS: <comma-separated list of blicket objects, e.g. A, B>"
        )

    def render_answer_retry_prompt(self) -> str:
        """Render a retry prompt for answer parsing failures."""
        if self.include_order_stochastic:
            return (
                "I couldn't parse your answer. Please respond with exactly:\n"
                "RULE_TYPE: conjunctive OR disjunctive OR order_sensitive OR stochastic OR order_sensitive_stochastic\n"
                "BLICKETS: <comma-separated list, e.g. A, B>\n"
                "ORDER: <comma-separated sequence if order_sensitive or order_sensitive_stochastic, e.g. A, D, C>"
            )
        if self.include_order_sensitive:
            return (
                "I couldn't parse your answer. Please respond with exactly:\n"
                "RULE_TYPE: conjunctive OR disjunctive OR order_sensitive\n"
                "BLICKETS: <comma-separated list, e.g. A, B>\n"
                "ORDER: <comma-separated sequence if order_sensitive, e.g. A, D, C>"
            )
        return (
            "I couldn't parse your answer. Please respond with exactly:\n"
            "RULE_TYPE: conjunctive OR disjunctive\n"
            "BLICKETS: <comma-separated list, e.g. A, B>"
        )

    def parse_action(self, response: str, valid_actions: list[Action5]) -> Action5 | None:
        """Parse an LLM response into an Action5."""
        text = response.strip()

        # Strategy 1: Look for "ACTION: N" pattern
        match = re.search(r"ACTION:\s*(\d+)", text, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(valid_actions):
                return valid_actions[idx]

        # Strategy 2: Look for "ACTION: keyword" pattern
        match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            action_text = match.group(1).strip().lower()
            action = self._match_keywords(action_text, valid_actions)
            if action:
                return action

        # Strategy 3: Check last line for action keywords
        last_line = text.strip().split("\n")[-1].lower()
        action = self._match_keywords(last_line, valid_actions)
        if action:
            return action

        # Strategy 4: Full text scan
        action = self._match_keywords(text.lower(), valid_actions)
        if action:
            return action

        return None

    def _match_keywords(self, text: str, valid_actions: list[Action5]) -> Action5 | None:
        """Match text against action keywords, restricted to valid actions."""
        for action in valid_actions:
            for keyword in ACTION_INFO_5[action]["keywords"]:
                if keyword in text:
                    return action
        return None

    def parse_answer(
        self, response: str
    ) -> tuple[CausalRuleType | None, set[str] | None, list[str] | None]:
        """Parse the agent's answer from free text (5-object version).

        Returns:
            3-tuple of (rule_type, required_objects, order).
            order is None when rule_type is not ORDER_SENSITIVE or ORDER_SENSITIVE_STOCHASTIC.
        """
        text = response.strip()

        # Parse rule type — check longer strings first to avoid ambiguity
        rule_type = None
        match = re.search(r"RULE_TYPE:\s*([\w_]+)", text, re.IGNORECASE)
        if match:
            rt = match.group(1).lower()
            if "order_sensitive_stochastic" in rt or "order_stochastic" in rt:
                rule_type = CausalRuleType.ORDER_SENSITIVE_STOCHASTIC
            elif "order" in rt:
                rule_type = CausalRuleType.ORDER_SENSITIVE
            elif "stochastic" in rt:
                rule_type = CausalRuleType.STOCHASTIC
            if rule_type is None:
                rt = match.group(1).lower()
                if "conjunct" in rt:
                    rule_type = CausalRuleType.CONJUNCTIVE
                elif "disjunct" in rt:
                    rule_type = CausalRuleType.DISJUNCTIVE
        else:
            lower = text.lower()
            if "order_sensitive_stochastic" in lower or "order sensitive stochastic" in lower:
                rule_type = CausalRuleType.ORDER_SENSITIVE_STOCHASTIC
            elif "order_sensitive" in lower or "order sensitive" in lower:
                rule_type = CausalRuleType.ORDER_SENSITIVE
            elif "stochastic" in lower:
                rule_type = CausalRuleType.STOCHASTIC
            elif "conjunctive" in lower and "disjunctive" not in lower:
                rule_type = CausalRuleType.CONJUNCTIVE
            elif "disjunctive" in lower and "conjunctive" not in lower:
                rule_type = CausalRuleType.DISJUNCTIVE

        # Parse blickets (A-E instead of A-C)
        # Use [A-Za-z, ]+ (literal space) to avoid matching across newlines
        objects = None
        match = re.search(r"BLICKETS?:\s*([A-Za-z, ]+)", text, re.IGNORECASE)
        if match:
            obj_text = match.group(1).upper()
            objects = {o.strip() for o in obj_text.split(",") if o.strip() in {"A", "B", "C", "D", "E"}}
        else:
            match = re.search(
                r"blickets?\s+(?:are|is)\s+([A-E](?:\s*(?:,|and)\s*[A-E])*)",
                text,
                re.IGNORECASE,
            )
            if match:
                obj_text = match.group(1).upper()
                objects = {c for c in obj_text if c in {"A", "B", "C", "D", "E"}}

        if rule_type is None or objects is None or len(objects) == 0:
            return None, None, None

        # Parse order (meaningful for ORDER_SENSITIVE and ORDER_SENSITIVE_STOCHASTIC)
        order = None
        if rule_type in (CausalRuleType.ORDER_SENSITIVE, CausalRuleType.ORDER_SENSITIVE_STOCHASTIC):
            match = re.search(r"ORDER:\s*([A-Za-z, ]+)", text, re.IGNORECASE)
            if match:
                order_text = match.group(1).upper()
                order = [o.strip() for o in order_text.split(",") if o.strip() in {"A", "B", "C", "D", "E"}]
                if not order:
                    order = None

        return rule_type, objects, order


# --- 6-Object Interface ---

ACTION_INFO_6 = {
    Action6.PLACE_A: {"name": "Place object A on detector", "keywords": ["place a", "put a"]},
    Action6.PLACE_B: {"name": "Place object B on detector", "keywords": ["place b", "put b"]},
    Action6.PLACE_C: {"name": "Place object C on detector", "keywords": ["place c", "put c"]},
    Action6.PLACE_D: {"name": "Place object D on detector", "keywords": ["place d", "put d"]},
    Action6.PLACE_E: {"name": "Place object E on detector", "keywords": ["place e", "put e"]},
    Action6.PLACE_F: {"name": "Place object F on detector", "keywords": ["place f", "put f"]},
    Action6.REMOVE_A: {"name": "Remove object A from detector", "keywords": ["remove a", "take a"]},
    Action6.REMOVE_B: {"name": "Remove object B from detector", "keywords": ["remove b", "take b"]},
    Action6.REMOVE_C: {"name": "Remove object C from detector", "keywords": ["remove c", "take c"]},
    Action6.REMOVE_D: {"name": "Remove object D from detector", "keywords": ["remove d", "take d"]},
    Action6.REMOVE_E: {"name": "Remove object E from detector", "keywords": ["remove e", "take e"]},
    Action6.REMOVE_F: {"name": "Remove object F from detector", "keywords": ["remove f", "take f"]},
    Action6.CHECK: {"name": "Submit your answer", "keywords": ["check", "submit", "answer"]},
}


class TextInterface6:
    """Renders 6-object environment states as text and parses LLM responses."""

    def __init__(
        self,
        include_order_sensitive: bool = False,
        include_order_stochastic: bool = False,
    ) -> None:
        self.include_order_sensitive = include_order_sensitive
        self.include_order_stochastic = include_order_stochastic

    def render_state(self, env: BlicketEnv6) -> str:
        """Render the current environment state as text for the LLM."""
        state = env.state
        valid_actions = env.get_valid_actions()

        lines = []
        lines.append("=== Blicket Detector Status ===")

        if state.objects_on_detector:
            on_det = ", ".join(sorted(state.objects_on_detector))
            lines.append(f"Objects on detector: {on_det}")
        else:
            lines.append("Objects on detector: (none)")

        status = "ACTIVE (glowing)" if state.detector_active else "INACTIVE (not glowing)"
        lines.append(f"Detector: {status}")

        lines.append(f"Steps taken: {state.step_count}/{env.max_steps}")
        lines.append("")

        lines.append("Available actions:")
        for i, action in enumerate(valid_actions, 1):
            lines.append(f"  {i}. {ACTION_INFO_6[action]['name']}")

        lines.append("")
        lines.append(
            "Choose an action by number or name. "
            "When ready to answer, choose 'Submit your answer'."
        )

        return "\n".join(lines)

    def render_observation(self, obs: Observation) -> str:
        """Render an observation as a natural language description."""
        return obs.message

    def render_task_description(self) -> str:
        """Render the initial task description for the 6-object agent."""
        base = (
            "You are interacting with a 'blicket detector' — a device that glows "
            "when certain objects (called 'blickets') are placed on it.\n\n"
            "There are 6 objects: A, B, C, D, E, and F. Some may be blickets, some may not.\n\n"
            "The detector follows a hidden rule:\n"
            "- CONJUNCTIVE: ALL blicket objects must be on the detector for it to activate.\n"
            "- DISJUNCTIVE: ANY single blicket object is enough to activate the detector.\n"
        )

        if self.include_order_stochastic:
            base += (
                "- ORDER_SENSITIVE: ALL blicket objects must be on the detector AND they "
                "must have been placed in a specific sequence (could be 2, 3, 4, 5, or even "
                "6 objects in a required order).\n"
                "- STOCHASTIC: The correct configuration activates the detector only SOME "
                "of the time (probabilistic activation).\n"
                "- ORDER_SENSITIVE_STOCHASTIC: ALL blicket objects must be in the correct "
                "sequence AND activation is probabilistic (may not activate every time even "
                "with the correct configuration and order).\n"
            )
        elif self.include_order_sensitive:
            base += (
                "- ORDER_SENSITIVE: ALL blicket objects must be on the detector AND they "
                "must have been placed in a specific sequence (could be 2, 3, 4, 5, or even "
                "6 objects in a required order).\n"
            )

        base += (
            "\nYour goal: Figure out which objects are blickets AND whether the rule is "
            "conjunctive"
        )
        if self.include_order_stochastic:
            base += ", disjunctive, order_sensitive, stochastic, or order_sensitive_stochastic"
        elif self.include_order_sensitive:
            base += ", disjunctive, or order_sensitive"
        else:
            base += " or disjunctive"

        base += (
            ".\n\n"
            "You can place objects, remove objects, and observe whether the detector "
            "activates. When you're confident, submit your answer.\n\n"
            "Respond with:\n"
            "REASONING: <your current thinking>\n"
            "ACTION: <number or action name>"
        )

        return base

    def render_answer_prompt(self) -> str:
        """Render the prompt for the agent to submit their answer."""
        if self.include_order_stochastic:
            return (
                "You chose to submit your answer. Please provide your conclusion.\n\n"
                "Respond with:\n"
                "RULE_TYPE: conjunctive OR disjunctive OR order_sensitive OR stochastic OR order_sensitive_stochastic\n"
                "BLICKETS: <comma-separated list of blicket objects, e.g. A, B>\n"
                "ORDER: <comma-separated sequence if order_sensitive or order_sensitive_stochastic, e.g. A, D, C>"
            )
        if self.include_order_sensitive:
            return (
                "You chose to submit your answer. Please provide your conclusion.\n\n"
                "Respond with:\n"
                "RULE_TYPE: conjunctive OR disjunctive OR order_sensitive\n"
                "BLICKETS: <comma-separated list of blicket objects, e.g. A, B>\n"
                "ORDER: <comma-separated sequence if order_sensitive, e.g. A, D, C>"
            )
        return (
            "You chose to submit your answer. Please provide your conclusion.\n\n"
            "Respond with:\n"
            "RULE_TYPE: conjunctive OR disjunctive\n"
            "BLICKETS: <comma-separated list of blicket objects, e.g. A, B>"
        )

    def render_answer_retry_prompt(self) -> str:
        """Render a retry prompt for answer parsing failures."""
        if self.include_order_stochastic:
            return (
                "I couldn't parse your answer. Please respond with exactly:\n"
                "RULE_TYPE: conjunctive OR disjunctive OR order_sensitive OR stochastic OR order_sensitive_stochastic\n"
                "BLICKETS: <comma-separated list, e.g. A, B>\n"
                "ORDER: <comma-separated sequence if order_sensitive or order_sensitive_stochastic, e.g. A, D, C>"
            )
        if self.include_order_sensitive:
            return (
                "I couldn't parse your answer. Please respond with exactly:\n"
                "RULE_TYPE: conjunctive OR disjunctive OR order_sensitive\n"
                "BLICKETS: <comma-separated list, e.g. A, B>\n"
                "ORDER: <comma-separated sequence if order_sensitive, e.g. A, D, C>"
            )
        return (
            "I couldn't parse your answer. Please respond with exactly:\n"
            "RULE_TYPE: conjunctive OR disjunctive\n"
            "BLICKETS: <comma-separated list, e.g. A, B>"
        )

    def parse_action(self, response: str, valid_actions: list[Action6]) -> Action6 | None:
        """Parse an LLM response into an Action6."""
        text = response.strip()

        # Strategy 1: Look for "ACTION: N" pattern
        match = re.search(r"ACTION:\s*(\d+)", text, re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(valid_actions):
                return valid_actions[idx]

        # Strategy 2: Look for "ACTION: keyword" pattern
        match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if match:
            action_text = match.group(1).strip().lower()
            action = self._match_keywords(action_text, valid_actions)
            if action:
                return action

        # Strategy 3: Check last line for action keywords
        last_line = text.strip().split("\n")[-1].lower()
        action = self._match_keywords(last_line, valid_actions)
        if action:
            return action

        # Strategy 4: Full text scan
        action = self._match_keywords(text.lower(), valid_actions)
        if action:
            return action

        return None

    def _match_keywords(self, text: str, valid_actions: list[Action6]) -> Action6 | None:
        """Match text against action keywords, restricted to valid actions."""
        for action in valid_actions:
            for keyword in ACTION_INFO_6[action]["keywords"]:
                if keyword in text:
                    return action
        return None

    def parse_answer(
        self, response: str
    ) -> tuple[CausalRuleType | None, set[str] | None, list[str] | None]:
        """Parse the agent's answer from free text (6-object version).

        Returns:
            3-tuple of (rule_type, required_objects, order).
            order is None when rule_type is not ORDER_SENSITIVE or ORDER_SENSITIVE_STOCHASTIC.
        """
        text = response.strip()

        # Parse rule type — check longer strings first to avoid ambiguity
        rule_type = None
        match = re.search(r"RULE_TYPE:\s*([\w_]+)", text, re.IGNORECASE)
        if match:
            rt = match.group(1).lower()
            if "order_sensitive_stochastic" in rt or "order_stochastic" in rt:
                rule_type = CausalRuleType.ORDER_SENSITIVE_STOCHASTIC
            elif "order" in rt:
                rule_type = CausalRuleType.ORDER_SENSITIVE
            elif "stochastic" in rt:
                rule_type = CausalRuleType.STOCHASTIC
            if rule_type is None:
                rt = match.group(1).lower()
                if "conjunct" in rt:
                    rule_type = CausalRuleType.CONJUNCTIVE
                elif "disjunct" in rt:
                    rule_type = CausalRuleType.DISJUNCTIVE
        else:
            lower = text.lower()
            if "order_sensitive_stochastic" in lower or "order sensitive stochastic" in lower:
                rule_type = CausalRuleType.ORDER_SENSITIVE_STOCHASTIC
            elif "order_sensitive" in lower or "order sensitive" in lower:
                rule_type = CausalRuleType.ORDER_SENSITIVE
            elif "stochastic" in lower:
                rule_type = CausalRuleType.STOCHASTIC
            elif "conjunctive" in lower and "disjunctive" not in lower:
                rule_type = CausalRuleType.CONJUNCTIVE
            elif "disjunctive" in lower and "conjunctive" not in lower:
                rule_type = CausalRuleType.DISJUNCTIVE

        # Parse blickets (A-F)
        objects = None
        match = re.search(r"BLICKETS?:\s*([A-Za-z, ]+)", text, re.IGNORECASE)
        if match:
            obj_text = match.group(1).upper()
            objects = {o.strip() for o in obj_text.split(",") if o.strip() in {"A", "B", "C", "D", "E", "F"}}
        else:
            match = re.search(
                r"blickets?\s+(?:are|is)\s+([A-F](?:\s*(?:,|and)\s*[A-F])*)",
                text,
                re.IGNORECASE,
            )
            if match:
                obj_text = match.group(1).upper()
                objects = {c for c in obj_text if c in {"A", "B", "C", "D", "E", "F"}}

        if rule_type is None or objects is None or len(objects) == 0:
            return None, None, None

        # Parse order (meaningful for ORDER_SENSITIVE and ORDER_SENSITIVE_STOCHASTIC)
        order = None
        if rule_type in (CausalRuleType.ORDER_SENSITIVE, CausalRuleType.ORDER_SENSITIVE_STOCHASTIC):
            match = re.search(r"ORDER:\s*([A-Za-z, ]+)", text, re.IGNORECASE)
            if match:
                order_text = match.group(1).upper()
                order = [o.strip() for o in order_text.split(",") if o.strip() in {"A", "B", "C", "D", "E", "F"}]
                if not order:
                    order = None

        return rule_type, objects, order
