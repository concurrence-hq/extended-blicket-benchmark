"""Experiment configuration and cost tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class AgentType(Enum):
    BASE_LLM = "base_llm"
    CG_ONLY = "cg_only"
    FULL_CG_DB = "full_cg_db"


class ConditionType(Enum):
    CONJUNCTIVE = "conjunctive"
    DISJUNCTIVE = "disjunctive"
    ORDER_SENSITIVE = "order_sensitive"
    STOCHASTIC = "stochastic"
    HIDDEN_MODERATOR = "hidden_moderator"
    ORDER_STOCHASTIC = "order_stochastic"


# Cost per token estimates (Sonnet 4.5 pricing as of Jan 2025)
SONNET_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000  # $3/M input tokens
SONNET_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15/M output tokens
HAIKU_INPUT_COST_PER_TOKEN = 0.80 / 1_000_000  # $0.80/M input tokens
HAIKU_OUTPUT_COST_PER_TOKEN = 4.0 / 1_000_000  # $4/M output tokens


class ExperimentConfig(BaseModel):
    """Configuration for a full experiment run."""

    # Episode settings
    episodes_per_condition: int = 30
    max_steps_per_episode: int = 50
    max_retries_per_action: int = 3

    # Model settings
    action_model: str = "claude-sonnet-4-5-20250929"
    trigger_model: str = "claude-haiku-4-5-20251001"

    # Cost controls
    max_cost_per_episode: float = 2.0  # USD
    max_total_cost: float = 1000.0  # USD

    # Environment settings — which objects are blickets for each standard condition
    conjunctive_blickets: list[str] = Field(default=["A", "B"])
    disjunctive_blickets: list[str] = Field(default=["A", "B"])

    # Extended condition settings
    order_required_order: list[str] = Field(default=["A", "B"])
    stochastic_probability: float = 0.7
    moderator_switch_after: int = 3

    # Output
    output_dir: Path = Path("data")
    trace_dir: Path = Path("data/traces")
    results_dir: Path = Path("data/results")

    # Reproducibility
    base_seed: int = 42

    def get_episode_seed(self, condition: str, agent: str, episode: int) -> int:
        """Deterministic seed per episode."""
        h = hash((condition, agent, episode, self.base_seed))
        return abs(h) % (2**31)


@dataclass
class PartialFeedbackConfig:
    """Configuration for partial feedback during order-sensitive episodes."""

    enabled: bool = False
    min_steps: int = 5
    min_unique_combos: int = 3
    correct_objects: list[str] = field(default_factory=list)


@dataclass
class GradualFeedbackConfig:
    """Configuration for gradual periodic feedback during order-sensitive episodes.

    Reveals one correct object per checkpoint interval, only if the agent has
    tested it and the detector has never activated. Mutually exclusive with
    PartialFeedbackConfig.
    """

    enabled: bool = False
    checkpoint_interval: int = 5
    correct_objects: list[str] = field(default_factory=list)


class CostTracker:
    """Tracks cumulative API costs across episodes."""

    def __init__(self, max_total: float = 500.0) -> None:
        self.max_total = max_total
        self._total_cost = 0.0
        self._episode_costs: list[dict] = []

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def remaining_budget(self) -> float:
        return self.max_total - self._total_cost

    def add_episode_cost(
        self,
        condition: str,
        agent: str,
        episode: int,
        input_tokens: int,
        output_tokens: int,
        trigger_tokens: int = 0,
    ) -> float:
        """Record cost for one episode. Returns the episode cost."""
        sonnet_cost = (
            input_tokens * SONNET_INPUT_COST_PER_TOKEN
            + output_tokens * SONNET_OUTPUT_COST_PER_TOKEN
        )
        haiku_cost = trigger_tokens * HAIKU_INPUT_COST_PER_TOKEN
        episode_cost = sonnet_cost + haiku_cost

        self._total_cost += episode_cost
        self._episode_costs.append({
            "condition": condition,
            "agent": agent,
            "episode": episode,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "trigger_tokens": trigger_tokens,
            "cost_usd": episode_cost,
            "cumulative_cost_usd": self._total_cost,
        })

        return episode_cost

    def check_budget(self) -> bool:
        """Returns True if budget is still available."""
        return self._total_cost < self.max_total

    def get_summary(self) -> dict:
        return {
            "total_cost_usd": self._total_cost,
            "episodes_run": len(self._episode_costs),
            "avg_cost_per_episode": (
                self._total_cost / len(self._episode_costs)
                if self._episode_costs
                else 0
            ),
            "remaining_budget_usd": self.remaining_budget,
            "episode_costs": self._episode_costs,
        }
