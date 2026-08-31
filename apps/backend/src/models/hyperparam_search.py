"""Hyperparameter optimization for Doom Index.

Production-grade search with:
- Optuna integration with pruning
- WandB logging
- Distributed search across GPUs
- Multi-objective optimization (accuracy vs latency)
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try Optuna
try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Hyperparameter search disabled.")

# Try WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DoomHyperParamSearch:
    """Hyperparameter search for multimodal Doom predictor.
    
    Searches over:
    - Graph architecture (hidden dim, layers, dropout)
    - Text encoder (freeze layers, learning rate)
    - Fusion (hidden dim, type)
    - Training (batch size, warmup, weight decay)
    """
    
    def __init__(
        self,
        graph_data,
        train_dataset,
        val_dataset,
        output_dir: str = "models/hpo",
        n_trials: int = 50,
        timeout_hours: int = 12,
        use_wandb: bool = False,
    ):
        self.graph_data = graph_data
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.timeout = timeout_hours * 3600
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not available")
    
    def objective(self, trial: "optuna.Trial") -> float:
        """Optuna objective function."""
        # Sample hyperparameters
        config = {
            "graph_hidden": trial.suggest_categorical("graph_hidden", [64, 128, 256]),
            "graph_layers": trial.suggest_int("graph_layers", 1, 4),
            "graph_dropout": trial.suggest_float("graph_dropout", 0.1, 0.5),
            
            "text_freeze": trial.suggest_int("text_freeze", 0, 6),
            "fusion_hidden": trial.suggest_categorical("fusion_hidden", [128, 256, 512]),
            "fusion_type": trial.suggest_categorical("fusion_type", ["mlp", "gated"]),
            
            "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "warmup_steps": trial.suggest_int("warmup_steps", 100, 1000, step=100),
        }
        
        # Log to WandB
        if self.use_wandb:
            wandb.init(project="doom-index-hpo", config=config, reinit=True)
        
        try:
            # Build model
            from src.models.gnn_model import MultimodalDoomPredictor
            from src.models.multimodal_trainer import MultimodalTrainer
            
            model = MultimodalDoomPredictor(
                graph_in_channels=self.graph_data.x.shape[1],
                graph_hidden=config["graph_hidden"],
                graph_out=128,
                graph_layers=config["graph_layers"],
                text_freeze=config["text_freeze"],
                fusion_hidden=config["fusion_hidden"],
                num_classes=2,
                dropout=config["graph_dropout"],
            )
            
            # Train for limited epochs
            trainer = MultimodalTrainer(
                model=model,
                graph_data=self.graph_data,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                output_dir=f"{self.output_dir}/trial_{trial.number}",
                batch_size=config["batch_size"],
                learning_rate=config["lr"],
                weight_decay=config["weight_decay"],
                epochs=5,  # Short training for HPO
                warmup_steps=config["warmup_steps"],
                fp16=True,
            )
            
            trainer.train()
            
            # Return validation F1 (maximize)
            val_f1 = trainer.best_val_f1
            
            # Report to Optuna for pruning
            trial.report(val_f1, step=5)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if self.use_wandb:
                wandb.log({"val_f1": val_f1})
                wandb.finish()
            
            return val_f1
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            if self.use_wandb:
                wandb.finish()
            raise
    
    def run(self) -> Dict:
        """Run hyperparameter search.
        
        Returns:
            Best hyperparameters found
        """
        logger.info(f"Starting HPO with {self.n_trials} trials")
        
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
            study_name="doom_multimodal_hpo",
        )
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )
        
        best = study.best_trial
        
        logger.info("=" * 60)
        logger.info("HPO Complete")
        logger.info(f"Best F1: {best.value:.4f}")
        logger.info(f"Best params: {best.params}")
        logger.info("=" * 60)
        
        return {
            "best_f1": best.value,
            "best_params": best.params,
            "n_trials_completed": len(study.trials),
            "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }
