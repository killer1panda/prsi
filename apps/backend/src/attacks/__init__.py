"""Adversarial Attack Simulator modules."""

from .adversarial_production import (
    AttackResult,
    DoomModelWrapper,
)
from .adversarial_production import ProductionAdversarialGenerator
from .adversarial_production import (
    ProductionAdversarialGenerator as AdversarialGenerator,
)

# Red / Blue / Purple team
from .red_team import RedTeamOrchestrator, AggressiveRedTeamOrchestrator, RedTeamResult
from .blue_team import BlueTeamOrchestrator, DefenseVerdict
from .purple_team import PurpleTeamOrchestrator

# Production GAN stack
from .doom_generator import ProductionDoomGenerator, DoomGenerator, GumbelSoftmaxSampler
from .doom_discriminator import WassersteinCritic, MultiRewardComposer
from .doom_reward_model import DoomRewardModel
from .doom_gan_trainer import DoomGANTrainer, TrainingConfig, SeedCorpusBuilder

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
