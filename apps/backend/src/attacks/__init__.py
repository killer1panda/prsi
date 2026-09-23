"""Adversarial Attack Simulator modules."""

from .adversarial_production import (
    AttackResult,
    DoomModelWrapper,
)
from .adversarial_production import ProductionAdversarialGenerator
from .adversarial_production import (
    ProductionAdversarialGenerator as AdversarialGenerator,
)
from .blue_team import BlueTeamOrchestrator, DefenseVerdict
from .doom_discriminator import MultiRewardComposer, WassersteinCritic
from .doom_gan_trainer import DoomGANTrainer, SeedCorpusBuilder, TrainingConfig

# Production GAN stack
from .doom_generator import DoomGenerator, GumbelSoftmaxSampler, ProductionDoomGenerator
from .doom_reward_model import DoomRewardModel
from .purple_team import PurpleTeamOrchestrator

# Red / Blue / Purple team
from .red_team import AggressiveRedTeamOrchestrator, RedTeamOrchestrator, RedTeamResult

__all__ = [
    # Legacy
    "ProductionAdversarialGenerator",
    "AdversarialGenerator",
    "AttackResult",
    "DoomModelWrapper",
    # Teams
    "RedTeamOrchestrator",
    "AggressiveRedTeamOrchestrator",
    "RedTeamResult",
    "BlueTeamOrchestrator",
    "DefenseVerdict",
    "PurpleTeamOrchestrator",
    # GAN
    "ProductionDoomGenerator",
    "DoomGenerator",
    "GumbelSoftmaxSampler",
    "WassersteinCritic",
    "MultiRewardComposer",
    "DoomRewardModel",
    "DoomGANTrainer",
    "TrainingConfig",
    "SeedCorpusBuilder",
]
