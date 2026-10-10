"""Agent 3: Full CG+DB agent — Sonnet + Context Graph + Dynamic Behaviors.

The complete compositional architecture. Before each action, Dynamic Behaviors
are evaluated. When triggered, they modify the CG at runtime, expanding the
agent's exploration space — enabling "coordinate system revision."
"""

from __future__ import annotations

import logging
from typing import Any

from environment.blicket_env import BlicketEnv

from .cg_agent import CGAgent
from .context_graph import ContextGraph, create_blicket_cg
from .dynamic_behaviors import DynamicBehavior, DynamicBehaviorEngine, create_blicket_behaviors

logger = logging.getLogger(__name__)


class FullAgent(CGAgent):
    """Full compositional agent: CG + Dynamic Behaviors.

    Before each get_action() call, evaluates DB triggers. When a behavior
    fires, the CG is expanded and a notification is injected into the
    conversation history.
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "claude-sonnet-4-5-20250929",
        trigger_model: str = "claude-haiku-4-5-20251001",
        max_retries: int = 3,
        context_graph: ContextGraph | None = None,
        text_interface=None,
        num_objects: int = 3,
        behaviors: list[DynamicBehavior] | None = None,
        initial_exploration_max_steps: int | None = None,
        cg_version: str = "v1",
        object_list: list[str] | None = None,
        object_screening_max_steps: int | None = None,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            model=model,
            max_retries=max_retries,
            context_graph=context_graph,
            text_interface=text_interface,
            num_objects=num_objects,
            initial_exploration_max_steps=initial_exploration_max_steps,
            cg_version=cg_version,
            object_list=object_list,
            object_screening_max_steps=object_screening_max_steps,
        )
        self.db_engine = DynamicBehaviorEngine(
            behaviors=behaviors or create_blicket_behaviors(),
            llm_client=llm_client,
            trigger_model=trigger_model,
        )
        self._db_activation_log: list[dict[str, Any]] = []
        self._had_unexpected_result = False
        self._activation_history: list[bool] = []
        self._last_step_notifications: list[str] = []

    def reset(self) -> None:
        super().reset()
        self.db_engine.reset()
        self._db_activation_log = []
        self._had_unexpected_result = False
        self._activation_history = []
        self._last_step_notifications = []

    def mark_unexpected_result(self) -> None:
        """Called by the episode runner when the last observation was surprising."""
        self._had_unexpected_result = True

    def observe_activation(self, active: bool) -> None:
        """Record a detector activation observation."""
        self._activation_history.append(active)

    def _build_exploration_summary(self, step: int, max_steps: int) -> str:
        """Build a short exploration summary for DB trigger context."""
        total = len(self._activation_history)
        n_active = sum(self._activation_history)
        cg_state = self.context_graph.current_state
        return f"Step {step}/{max_steps}. Activations: {n_active}/{total}. CG state: {cg_state}."

    async def get_action(self, env: BlicketEnv):
        """Evaluate DB triggers, then get action via CG agent."""
        step = env.state.step_count
        exploration_summary = self._build_exploration_summary(step, env.max_steps)

        # Clear per-step notifications
        self._last_step_notifications = []

        # Evaluate Dynamic Behaviors
        notifications = await self.db_engine.evaluate_and_apply(
            context_graph=self.context_graph,
            conversation_history=self.conversation,
            step_number=step,
            had_unexpected_result=self._had_unexpected_result,
            activation_history=self._activation_history,
            exploration_summary=exploration_summary,
        )

        # Capture notifications for trace logging
        self._last_step_notifications = list(notifications)

        # Inject notifications into conversation
        for notification in notifications:
            self.conversation.append({"role": "user", "content": notification})
            self._db_activation_log.append({
                "step": step,
                "notification": notification,
                "fired_behaviors": list(self.db_engine.fired_behaviors),
            })

        # Reset unexpected flag after DB evaluation
        self._had_unexpected_result = False

        # Delegate to CG agent for action selection
        return await super().get_action(env)

    def get_last_step_notifications(self) -> list[str]:
        """Return DB notifications from the most recent get_action() call."""
        return list(self._last_step_notifications)

    def get_trace_data(self) -> dict[str, Any]:
        data = super().get_trace_data()
        data["agent_type"] = "full_cg_db"
        data["db_activation_log"] = list(self._db_activation_log)
        data["db_fired_behaviors"] = list(self.db_engine.fired_behaviors)
        data["db_trigger_tokens"] = self.db_engine.trigger_tokens_used
        data["db_score_log"] = self.db_engine.score_log
        # Add trigger model instrumentation
        trigger_models = self.db_engine._trigger_response_models
        trigger_model_mismatch = any(
            m != self.db_engine.trigger_model for m in trigger_models
        )
        data["instrumentation"]["requested_trigger_model"] = self.db_engine.trigger_model
        data["instrumentation"]["responded_trigger_model"] = (
            trigger_models[0] if trigger_models else "no_calls"
        )
        data["instrumentation"]["trigger_model_mismatch"] = trigger_model_mismatch
        data["instrumentation"]["all_responded_trigger_models"] = list(set(trigger_models))
        return data
