"""Quantitative research, experimentation, and champion-challenger package."""
from ai_crypto_trader.research.champion_challenger import (
    ChampionChallengerManager,
    HeadToHeadEvaluation,
    PromotionCandidate,
)
from ai_crypto_trader.research.experiment_runner import (
    ExperimentResult,
    ExperimentRunner,
)
from ai_crypto_trader.research.hypothesis_generator import (
    Hypothesis,
    HypothesisGenerator,
)
from ai_crypto_trader.research.research_agent import (
    ResearchAgent,
    ResearchCycleResult,
)

__all__ = [
    "ChampionChallengerManager",
    "ExperimentResult",
    "ExperimentRunner",
    "HeadToHeadEvaluation",
    "Hypothesis",
    "HypothesisGenerator",
    "PromotionCandidate",
    "ResearchAgent",
    "ResearchCycleResult",
]
