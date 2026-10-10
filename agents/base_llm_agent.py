"""Agent 1: Bare LLM agent — Claude Sonnet with task prompt only.

No Context Graph, no Dynamic Behaviors. Serves as the baseline
for comparison against compositional architectures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from environment.blicket_env import Action, BlicketEnv, CausalRuleType
from environment.text_interface import TextInterface

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_t: int, output_t: int) -> None:
        self.input_tokens += input_t
        self.output_tokens += output_t


class BaseLLMAgent:
    """Bare LLM agent for the blicket task.

    Uses Claude Sonnet with a simple task-description system prompt.
    Conversation history grows throughout an episode.
    """

    def __init__(
        self,
        llm_client: Any,  # anthropic.AsyncAnthropic
        model: str = "claude-sonnet-4-5-20250929",
        max_retries: int = 3,
        text_interface: TextInterface | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.max_retries = max_retries
        self.text_interface = text_interface or TextInterface()
        self.conversation: list[dict[str, str]] = []
        self.token_usage = TokenUsage()
        self._raw_responses: list[str] = []
        self._response_models: list[str] = []
        self._response_ids: list[str] = []

    def reset(self) -> None:
        """Clear conversation history for a new episode."""
        self.conversation = []
        self.token_usage = TokenUsage()
        self._raw_responses = []
        self._response_models = []
        self._response_ids = []

    def _build_system_prompt(self) -> str:
        """Build the system prompt. Override in subclasses to add CG context."""
        return self.text_interface.render_task_description()

    async def get_action(self, env: BlicketEnv) -> Action | None:
        """Get the next action from the LLM.

        Args:
            env: The current environment.

        Returns:
            The parsed action, or None after max retries exhausted.
        """
        state_text = self.text_interface.render_state(env)
        valid_actions = env.get_valid_actions()

        self.conversation.append({"role": "user", "content": state_text})

        for attempt in range(self.max_retries):
            response_text = await self._call_llm()
            if response_text is None:
                continue

            self._raw_responses.append(response_text)

            action = self.text_interface.parse_action(response_text, valid_actions)
            if action is not None:
                self.conversation.append(
                    {"role": "assistant", "content": response_text}
                )
                return action

            # Parse failure — add retry prompt
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

    async def get_answer(
        self,
    ) -> tuple[CausalRuleType | None, set[str] | None, list[str] | None]:
        """Get the agent's final answer after choosing CHECK.

        Returns:
            3-tuple of (rule_type, blickets, order).
            order is None for non-order-sensitive rules.
            Returns (None, None, None) on parse failure.
        """
        answer_prompt = self.text_interface.render_answer_prompt()
        self.conversation.append({"role": "user", "content": answer_prompt})

        for attempt in range(self.max_retries):
            response_text = await self._call_llm()
            if response_text is None:
                continue

            self._raw_responses.append(response_text)

            parsed = self.text_interface.parse_answer(response_text)
            # Normalize: TextInterface returns 2-tuple, TextInterface5 returns 3-tuple
            if len(parsed) == 2:
                rule_type, blickets = parsed
                order = None
            else:
                rule_type, blickets, order = parsed

            if rule_type is not None and blickets is not None:
                self.conversation.append(
                    {"role": "assistant", "content": response_text}
                )
                return rule_type, blickets, order

            # Parse failure — delegate retry prompt to text_interface if available
            if attempt < self.max_retries - 1:
                if hasattr(self.text_interface, "render_answer_retry_prompt"):
                    retry_msg = self.text_interface.render_answer_retry_prompt()
                else:
                    retry_msg = (
                        "I couldn't parse your answer. Please respond with exactly:\n"
                        "RULE_TYPE: conjunctive OR disjunctive\n"
                        "BLICKETS: <comma-separated list, e.g. A, B>"
                    )
                self.conversation.append(
                    {"role": "assistant", "content": response_text}
                )
                self.conversation.append({"role": "user", "content": retry_msg})

        logger.warning("Max retries exhausted for answer parsing")
        return None, None, None

    async def _call_llm(self) -> str | None:
        """Make an API call and return the text response."""
        try:
            response = await self.llm_client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=1.0,
                system=self._build_system_prompt(),
                messages=self.conversation,
            )
            self.token_usage.add(
                response.usage.input_tokens, response.usage.output_tokens
            )
            self._response_models.append(getattr(response, "model", "unknown"))
            self._response_ids.append(getattr(response, "id", "unknown"))
            return response.content[0].text
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None

    def add_observation(self, observation_text: str) -> None:
        """Add an environment observation to the conversation."""
        self.conversation.append({"role": "user", "content": observation_text})

    def get_trace_data(self) -> dict[str, Any]:
        """Return data for episode trace logging."""
        # Check for model mismatches across all API calls
        action_model_mismatch = any(
            m != self.model for m in self._response_models
        )
        return {
            "agent_type": "base_llm",
            "model": self.model,
            "conversation": list(self.conversation),
            "raw_responses": list(self._raw_responses),
            "token_usage": {
                "input_tokens": self.token_usage.input_tokens,
                "output_tokens": self.token_usage.output_tokens,
                "total_tokens": self.token_usage.total,
            },
            "instrumentation": {
                "requested_action_model": self.model,
                "responded_action_model": self._response_models[0] if self._response_models else "no_calls",
                "temperature": 1.0,
                "api_response_id": self._response_ids[0] if self._response_ids else "no_calls",
                "action_model_mismatch": action_model_mismatch,
                "all_responded_action_models": list(set(self._response_models)),
            },
        }
