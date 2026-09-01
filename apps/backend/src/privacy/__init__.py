"""Privacy modules (differential privacy, federated learning)."""

from .dp_trainer import DPDoomTrainer, add_gaussian_noise
from .fl_simulator import (
    DoomClient,
    FederatedSimulator,
    FLSimulator,
    federated_averaging,
)

__all__ = [
    "DPDoomTrainer",
    "add_gaussian_noise",
    "FLSimulator",
    "FederatedSimulator",
    "DoomClient",
    "federated_averaging",
]
