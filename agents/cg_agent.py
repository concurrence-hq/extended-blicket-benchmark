"""Agent 2: Context Graph agent — Sonnet + fixed CG.

The CG provides structured exploration guidance via the system prompt
but cannot be modified at runtime (no Dynamic Behaviors).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from environment.blicket_env import BlicketEnv

from .base_llm_agent import BaseLLMAgent
from .context_graph import ContextGraph, ModificationLogEntry, create_blicket_cg

logger = logging.getLogger(__name__)


class CGAgent(BaseLLMAgent):
    """CG-only agent: system prompt includes Context Graph state each turn.

    The agent can request CG state transitions by including
    TRANSITION: <state_name> in its response. The CG is fixed — no
    new states can be added at runtime.
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "claude-sonnet-4-5-20250929",
        max_retries: int = 3,
        context_graph: ContextGraph | None = None,
        text_interface=None,
        num_objects: int = 3,
        initial_exploration_max_steps: int | None = None,
        cg_version: str = "v1",
        object_list: list[str] | None = None,
        object_screening_max_steps: int | None = None,
    ) -> None:
        super().__init__(llm_client=llm_client, model=model, max_retries=max_retries, text_interface=text_interface)
        self._num_objects = num_objects
        self._initial_exploration_max_steps = initial_exploration_max_steps
        self._cg_version = cg_version
        self._object_list = object_list
        self._object_screening_max_steps = object_screening_max_steps
        self.context_graph = context_graph or create_blicket_cg(
            num_objects=num_objects,
            cg_version=cg_version,
            object_list=object_list,
        )
        if self._initial_exploration_max_steps is not None:
            self.context_graph.initial_exploration_max_steps = self._initial_exploration_max_steps
        if self._object_screening_max_steps is not None:
            self.context_graph.object_screening_max_steps = self._object_screening_max_steps
        self._transition_log: list[dict[str, Any]] = []
        self._pending_desync_correction: str | None = None

    def reset(self) -> None:
        super().reset()
        self.context_graph = create_blicket_cg(
            num_objects=self._num_objects,
            cg_version=self._cg_version,
            object_list=self._object_list,
        )
        if self._initial_exploration_max_steps is not None:
            self.context_graph.initial_exploration_max_steps = self._initial_exploration_max_steps
        if self._object_screening_max_steps is not None:
            self.context_graph.object_screening_max_steps = self._object_screening_max_steps
        self._transition_log = []
        self._pending_desync_correction = None

    def _build_system_prompt(self) -> str:
        """Build system prompt with CG context appended."""
        base = self.text_interface.render_task_description()
        cg_context = self.context_graph.render_for_prompt()

        return (
            f"{base}\n\n"
            f"{cg_context}\n\n"
            f"In your response, include:\n"
            f"CG_STATE: <your current Context Graph state>\n"
            f"TRANSITION: <target state name> (only if you want to change states)\n"
            f"REASONING: <your current thinking>\n"
            f"ACTION: <number or action name>"
        )

    async def get_action(self, env: BlicketEnv):
        """Get action and process any CG transitions."""
        # Inject pending desync correction before backstop/state rendering
        if self._pending_desync_correction is not None:
            self.conversation.append(
                {"role": "user", "content": self._pending_desync_correction}
            )
            self._pending_desync_correction = None

        # Advance step counter and check backstop before rendering state
        backstop_result = self.context_graph.advance_step()
        if backstop_result is not None:
            from_state, backstop_target = backstop_result
            step_num = env.state.step_count
            self.context_graph.modification_log.append(
                ModificationLogEntry(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    step_number=step_num,
                    action="backstop_transition",
                    details={
                        "from": from_state,
                        "to": backstop_target,
                    },
                )
            )
            backstop_msg = (
                f"[SYSTEM] Backstop: forced transition from {from_state} "
                f"to {backstop_target}."
            )
            self.conversation.append({"role": "user", "content": backstop_msg})
            self._transition_log.append({
                "step": step_num,
                "from": from_state,
                "to": backstop_target,
                "success": True,
                "backstop": True,
            })
            logger.info(f"Backstop transition logged at step {step_num}")

        state_text = self.text_interface.render_state(env)
        valid_actions = env.get_valid_actions()

        self.conversation.append({"role": "user", "content": state_text})

        for attempt in range(self.max_retries):
            response_text = await self._call_llm()
            if response_text is None:
                continue

            self._raw_responses.append(response_text)

            # Process CG transitions before parsing action
            self._process_cg_transitions(response_text, env.state.step_count)

            action = self.text_interface.parse_action(response_text, valid_actions)
            if action is not None:
                self.conversation.append(
                    {"role": "assistant", "content": response_text}
                )
                return action

            if attempt < self.max_retries - 1:
                retry_msg = (
                    f"I couldn't parse your action. Please respond with "
                    f"ACTION: <number> where the number corresponds to one of "
                    f"the available actions (1-{len(valid_actions)})."
                )
                self.conversation.append(
                    {"role": "assistant", "content": response_text}
                )
                self.conversation.append({"role": "user", "content": retry_msg})

        logger.warning("Max retries exhausted for action parsing")
        return None

    def _process_cg_transitions(self, response: str, step_number: int) -> None:
        """Parse and execute CG state transitions from the LLM response."""
        match = re.search(r"TRANSITION:\s*(\S+)", response, re.IGNORECASE)
        if not match:
            return

        target = match.group(1).strip().upper()
        previous = self.context_graph.current_state
        success = self.context_graph.transition_to(target)

        self._transition_log.append({
            "step": step_number,
            "from": previous,
            "to": target,
            "success": success,
        })

        if success:
            logger.info(f"CG transition: {previous} -> {target} at step {step_number}")
        else:
            self._pending_desync_correction = (
                f"[SYSTEM] Your requested transition to {target} was not accepted. "
                f"You are still in {previous}. "
                f"Please continue working within {previous} or request a valid transition."
            )
            logger.debug(
                f"Invalid CG transition: {previous} -> {target} at step {step_number}"
            )

    def get_trace_data(self) -> dict[str, Any]:
        data = super().get_trace_data()
        data["agent_type"] = "cg_only"
        data["context_graph_final"] = self.context_graph.model_dump()
        data["transition_log"] = list(self._transition_log)
        return data
