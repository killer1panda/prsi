#!/usr/bin/env python3
"""Full TextAttack Integration with Advanced Constraints.

This module provides production-grade adversarial attack generation using
the complete TextAttack library with custom constraints for doom index manipulation.

Features:
- Full TextAttack recipe integration (TextFooler, BAEGarg, PWWS, DeepWordBug)
- Custom toxicity budget constraints (stay under moderation threshold)
- Semantic similarity constraints (USE encoder)
- Grammaticality constraints (language model perplexity)
- Multi-strategy ensemble attacks
- Batch processing for efficiency
- Attack success rate tracking
- Adversarial example caching

Usage:
    # Basic attack
    python src/attacks/textattack_full.py \
        --input "I love this product" \
        --target-doom 85 \
        --output attacked_examples.json
    
    # Batch attack
    python src/attacks/textattack_full.py \
        --input-file data/test_posts.csv \
        --batch-size 100 \
        --attack textfooler
    
    # Ensemble attack (multiple strategies)
    python src/attacks/textattack_full.py \
        --input "Controversial statement here" \
        --ensemble \
        --num-strategies 4
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# TextAttack imports
try:
    from textattack.attack_recipes import (
        TextFoolerJin2019,
        BAEGarg2019,
        PWWSRen2019,
        DeepWordBugGao2018,
        InputReductionFeng2018,
        Pruthi2019,
        MorpheusTan2020,
    )
    from textattack.constraints.pre_transformation import (
        StopwordModification,
        MaxWordIndexModification,
    )
    from textattack.constraints.post_transformation import (
        RepeatModification,
        StopwordInsertion,
    )
    from textattack.constraints.semantics.sentence_encoders import (
        UniversalSentenceEncoder,
        SentenceBERT,
    )
    from textattack.constraints.semantics import WordEmbeddingDistance
    from textattack.constraints.grammaticality import (
        PartOfSpeech,
        LanguageTool,
        GPT2Perplexity,
    )
    from textattack.transformations import (
        WordSwapEmbedding,
        WordSwapWordNet,
        WordSwapMaskedLM,
        WordSwapRandomCharacterDeletion,
        WordSwapRandomCharacterInsertion,
        WordSwapRandomCharacterSubstitution,
        WordSwapChangeCase,
    )
    from textattack.goal_functions import UntargetedClassification, TargetedClassification
    from textattack.search_methods import (
        GreedyWordSwapWIR,
        BeamSearch,
        ParticleSwarmOptimization,
        GeneticAlgorithm,
    )
    from textattack import Attack, Attacker
    from textattack.models.wrappers import ModelWrapper, HuggingFaceModelWrapper
    from textattack.datasets import Dataset
    TEXTATTACK_AVAILABLE = True
except ImportError as e:
    TEXTATTACK_AVAILABLE = False
    print(f"⚠️  TextAttack not fully installed: {e}")
    print("Install with: pip install textattack[transformers]")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    """Configuration for TextAttack attack."""
    attack_type: str = "textfooler"
    max_attacks_per_sample: int = 1
    perturbation_budget: float = 0.3  # Max 30% words changed
    min_semantic_similarity: float = 0.7  # USE similarity threshold
    max_toxicity_score: float = 0.7  # Stay under moderation threshold
    target_doom_increase: float = 20.0  # Minimum doom score increase
    use_ensemble: bool = False
    num_strategies: int = 3
    batch_size: int = 32
    timeout_per_sample: int = 60  # seconds


@dataclass
class AttackResult:
    """Result of a single adversarial attack."""
    original_text: str
    attacked_text: str
    original_doom_score: float
    attacked_doom_score: float
    doom_uplift: float
    attack_strategy: str
    words_changed: int
    percent_changed: float
    semantic_similarity: float
    toxicity_score: float
    passes_moderation: bool
    perplexity_score: float
    attack_duration_seconds: float
    attack_success: bool
    metadata: Optional[Dict] = None


class DoomIndexModelWrapper(ModelWrapper):
    """TextAttack-compatible wrapper for Doom Index predictor."""
    
    def __init__(self, predictor, tokenizer=None, max_length=256):
        self.predictor = predictor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model = None  # Not needed for black-box attacks
    
    def __call__(self, text_input_list: List[str]) -> np.ndarray:
        """Predict doom scores for batch of texts.
        
        TextAttack expects: [negative_prob, positive_prob] format
        We map doom_score to [1 - doom/100, doom/100]
        """
        results = []
        
        for text in text_input_list:
            try:
                # Call the doom predictor
                prediction = self.predictor.predict(text, author_id="attack_target")
                doom_score = prediction.get('probability', 0.5) * 100
                
                # Convert to TextAttack format [P(negative), P(positive)]
                # High doom = high probability of "cancellation" (positive class)
                prob_negative = 1.0 - (doom_score / 100.0)
                prob_positive = doom_score / 100.0
                
                results.append([prob_negative, prob_positive])
                
            except Exception as e:
                logger.warning(f"Prediction failed for text: {text[:50]}... Error: {e}")
                # Return neutral prediction on error
                results.append([0.5, 0.5])
        
        return np.array(results)


class ToxicityConstraint:
    """Custom constraint to keep toxicity under moderation threshold."""
    
    def __init__(self, max_toxicity: float = 0.7, toxicity_model=None):
        self.max_toxicity = max_toxicity
        self.toxicity_model = toxicity_model or self._default_toxicity_proxy
    
    def _default_toxicity_proxy(self, text: str) -> float:
        """Simple heuristic toxicity proxy."""
        # In production, use Perspective API or Detoxify
        toxic_words = {'hate', 'kill', 'die', 'stupid', 'idiot', 'cancel', 'boycott'}
        words = text.lower().split()
        toxic_count = sum(1 for w in words if w in toxic_words)
        return min(1.0, toxic_count / max(1, len(words)) * 10)
    
    def __call__(self, current_text: str, **kwargs) -> bool:
        """Check if text passes toxicity constraint."""
        toxicity = self.toxicity_model(current_text)
        return toxicity <= self.max_toxicity


class TextAttackFullIntegrator:
    """Full TextAttack integration with advanced constraints."""
    
    def __init__(
        self,
        predictor,
        config: AttackConfig = None,
        device: str = "cuda",
    ):
        """Initialize TextAttack integrator.
        
        Args:
            predictor: Doom index predictor with .predict(text, author_id) method
            config: Attack configuration
            device: Device for model inference ('cuda' or 'cpu')
        """
        if not TEXTATTACK_AVAILABLE:
            raise ImportError("TextAttack is not installed. Install with: pip install textattack[transformers]")
        
        self.predictor = predictor
        self.config = config or AttackConfig()
        self.device = device
        
        # Create model wrapper
        self.model_wrapper = DoomIndexModelWrapper(predictor)
        
        # Initialize attack recipes
        self.attacks = self._build_attacks()
        
        # Toxicity constraint
        self.toxicity_constraint = ToxicityConstraint(
            max_toxicity=self.config.max_toxicity_score
        )
        
        # Cache for adversarial examples
        self.cache: Dict[str, AttackResult] = {}
    
    def _build_attacks(self) -> Dict[str, Attack]:
        """Build attack recipes based on config."""
        attacks = {}
        
        # Semantic similarity encoder
        use_encoder = UniversalSentenceEncoder()
        
        # Common transformations
        embedding_swap = WordSwapEmbedding(max_candidates=50)
        wordnet_swap = WordSwapWordNet()
        masked_lm_swap = WordSwapMaskedLM()
        
        # Search methods
        greedy_wir = GreedyWordSwapWIR(wir_method="delete")
        beam_search = BeamSearch(beam_width=1)
        genetic_algo = GeneticAlgorithm(population_size=60, mutation_rate=0.3)
        
        # Build TextFooler
        if "textfooler" in self.config.attack_type or self.config.use_ensemble:
            attacks["textfooler"] = TextFoolerJin2019.build(self.model_wrapper)
        
        # Build BAE
        if "bae" in self.config.attack_type or self.config.use_ensemble:
            attacks["bae"] = BAEGarg2019.build(self.model_wrapper)
        
        # Build PWWS
        if "pwws" in self.config.attack_type or self.config.use_ensemble:
            attacks["pwws"] = PWWSRen2019.build(self.model_wrapper)
        
        # Build DeepWordBug
        if "deepwordbug" in self.config.attack_type or self.config.use_ensemble:
            attacks["deepwordbug"] = DeepWordBugGao2018.build(self.model_wrapper)
        
        # Build custom attack with all constraints
        if "custom" in self.config.attack_type or self.config.use_ensemble:
            custom_attack = Attack(
                goal_function=UntargetedClassification(self.model_wrapper),
                constraints=[
                    StopwordModification(),
                    RepeatModification(),
                    use_encoder,
                    WordEmbeddingDistance(min_cos_sim=0.8),
                    PartOfSpeech(modifier="adj"),
                    self.toxicity_constraint,
                ],
                transformation=masked_lm_swap,
                search_method=genetic_algo,
            )
            attacks["custom"] = custom_attack
        
        return attacks
    
    def attack_single(
        self,
        text: str,
        target_doom: float = None,
        attack_strategy: str = None,
    ) -> AttackResult:
        """Attack a single text sample.
        
        Args:
            text: Original text to attack
            target_doom: Target doom score (optional)
            attack_strategy: Specific attack to use (or auto-select)
            
        Returns:
            AttackResult with original and attacked information
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{text}_{attack_strategy}"
        if cache_key in self.cache:
            logger.info("Returning cached attack result")
            return self.cache[cache_key]
        
        # Get original doom score
        original_pred = self.predictor.predict(text, author_id="attack_target")
        original_doom = original_pred.get('probability', 0.5) * 100
        
        logger.info(f"Original doom score: {original_doom:.1f}")
        
        # Select attack strategy
        if attack_strategy and attack_strategy in self.attacks:
            attack = self.attacks[attack_strategy]
        else:
            # Auto-select best attack
            attack = list(self.attacks.values())[0]
            attack_strategy = list(self.attacks.keys())[0]
        
        # Create dataset sample
        dataset = Dataset([(text, 1)])  # Label=1 for targeted attack
        
        # Run attack
        try:
            attacker = Attacker(
                attack=attack,
                dataset=dataset,
                shuffle=False,
            )
            
            results = attacker.attack_dataset()
            
            if not results or len(results) == 0:
                # Attack failed
                return AttackResult(
                    original_text=text,
                    attacked_text=text,
                    original_doom_score=original_doom,
                    attacked_doom_score=original_doom,
                    doom_uplift=0.0,
                    attack_strategy=attack_strategy,
                    words_changed=0,
                    percent_changed=0.0,
                    semantic_similarity=1.0,
                    toxicity_score=0.0,
                    passes_moderation=True,
                    perplexity_score=0.0,
                    attack_duration_seconds=time.time() - start_time,
                    attack_success=False,
                    metadata={"error": "Attack failed to generate adversarial example"},
                )
            
            # Extract attacked text
            attacked_text = results[0].perturbed_text()
            
        except Exception as e:
            logger.error(f"Attack execution failed: {e}")
            return AttackResult(
                original_text=text,
                attacked_text=text,
                original_doom_score=original_doom,
                attacked_doom_score=original_doom,
                doom_uplift=0.0,
                attack_strategy=attack_strategy,
                words_changed=0,
                percent_changed=0.0,
                semantic_similarity=1.0,
                toxicity_score=0.0,
                passes_moderation=True,
                perplexity_score=0.0,
                attack_duration_seconds=time.time() - start_time,
                attack_success=False,
                metadata={"error": str(e)},
            )
        
        # Get attacked doom score
        attacked_pred = self.predictor.predict(attacked_text, author_id="attack_target")
        attacked_doom = attacked_pred.get('probability', 0.5) * 100
        
        # Calculate metrics
        doom_uplift = attacked_doom - original_doom
        words_original = text.split()
        words_attacked = attacked_text.split()
        words_changed = sum(1 for w1, w2 in zip(words_original, words_attacked) if w1 != w2)
        percent_changed = words_changed / max(1, len(words_original))
        
        # Estimate semantic similarity (simplified)
        semantic_sim = 1.0 - percent_changed  # Rough approximation
        
        # Check toxicity
        toxicity = self.toxicity_constraint.toxicity_proxy(attacked_text)
        passes_moderation = toxicity <= self.config.max_toxicity_score
        
        result = AttackResult(
            original_text=text,
            attacked_text=attacked_text,
            original_doom_score=original_doom,
            attacked_doom_score=attacked_doom,
            doom_uplift=doom_uplift,
            attack_strategy=attack_strategy,
            words_changed=words_changed,
            percent_changed=percent_changed,
            semantic_similarity=semantic_sim,
            toxicity_score=toxicity,
            passes_moderation=passes_moderation,
            perplexity_score=0.0,  # Would need GPT2 to compute
            attack_duration_seconds=time.time() - start_time,
            attack_success=doom_uplift >= self.config.target_doom_increase,
            metadata={
                "target_doom": target_doom,
                "perturbation_budget": self.config.perturbation_budget,
            }
        )
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    def attack_batch(
        self,
        texts: List[str],
        target_dooms: List[float] = None,
        attack_strategy: str = None,
    ) -> List[AttackResult]:
        """Attack a batch of texts.
        
        Args:
            texts: List of texts to attack
            target_dooms: Optional list of target doom scores
            attack_strategy: Attack strategy to use
            
        Returns:
            List of AttackResults
        """
        if target_dooms is None:
            target_dooms = [None] * len(texts)
        
        results = []
        
        for text, target in tqdm(zip(texts, target_dooms), total=len(texts), desc="Attacking"):
            result = self.attack_single(text, target_doom=target, attack_strategy=attack_strategy)
            results.append(result)
        
        return results
    
    def ensemble_attack(
        self,
        text: str,
        num_strategies: int = None,
    ) -> List[AttackResult]:
        """Run multiple attack strategies and return best result.
        
        Args:
            text: Text to attack
            num_strategies: Number of strategies to try
            
        Returns:
            List of results from all strategies, sorted by doom_uplift
        """
        if num_strategies is None:
            num_strategies = self.config.num_strategies
        
        strategies = list(self.attacks.keys())[:num_strategies]
        results = []
        
        for strategy in strategies:
            logger.info(f"Trying attack strategy: {strategy}")
            result = self.attack_single(text, attack_strategy=strategy)
            results.append(result)
        
        # Sort by doom uplift (descending)
        results.sort(key=lambda r: r.doom_uplift, reverse=True)
        
        logger.info(f"Best strategy: {results[0].attack_strategy} (uplift: {results[0].doom_uplift:.1f})")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get attack statistics."""
        if not self.cache:
            return {"total_attacks": 0}
        
        results = list(self.cache.values())
        
        return {
            "total_attacks": len(results),
            "successful_attacks": sum(1 for r in results if r.attack_success),
            "success_rate": sum(1 for r in results if r.attack_success) / len(results),
            "avg_doom_uplift": np.mean([r.doom_uplift for r in results]),
            "max_doom_uplift": max(r.doom_uplift for r in results),
            "avg_words_changed": np.mean([r.words_changed for r in results]),
            "avg_attack_duration": np.mean([r.attack_duration_seconds for r in results]),
            "moderation_pass_rate": sum(1 for r in results if r.passes_moderation) / len(results),
        }
    
    def save_cache(self, output_path: str):
        """Save attack cache to file."""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": asdict(self.config),
            "statistics": self.get_statistics(),
            "examples": [asdict(r) for r in self.cache.values()]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(self.cache)} attack examples to {output_path}")
    
    def load_cache(self, input_path: str):
        """Load attack cache from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        for example in data["examples"]:
            # Convert dict back to AttackResult
            result = AttackResult(**example)
            key = f"{result.original_text}_{result.attack_strategy}"
            self.cache[key] = result
        
        logger.info(f"Loaded {len(self.cache)} attack examples from {input_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TextAttack Full Integration")
    parser.add_argument("--input", type=str, help="Single input text to attack")
    parser.add_argument("--input-file", type=str, help="CSV file with texts to attack")
    parser.add_argument("--target-doom", type=float, default=80.0, help="Target doom score")
    parser.add_argument("--attack", type=str, default="textfooler", 
                       choices=["textfooler", "bae", "pwws", "deepwordbug", "custom", "all"],
                       help="Attack strategy")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble of attacks")
    parser.add_argument("--num-strategies", type=int, default=3, help="Number of strategies for ensemble")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--output", type=str, default="attack_results.json", help="Output file")
    parser.add_argument("--max-toxicity", type=float, default=0.7, help="Max toxicity threshold")
    parser.add_argument("--min-similarity", type=float, default=0.7, help="Min semantic similarity")
    
    args = parser.parse_args()
    
    if not TEXTATTACK_AVAILABLE:
        logger.error("TextAttack is not installed. Exiting.")
        sys.exit(1)
    
    # Create dummy predictor for demo (replace with real model)
    class DummyPredictor:
        def predict(self, text, author_id=None):
            # Simple heuristic: longer text = higher doom
            base_doom = 0.3 + 0.01 * len(text.split())
            # Add randomness
            import random
            doom = base_doom + random.uniform(-0.1, 0.1)
            return {"probability": min(1.0, max(0.0, doom))}
    
    predictor = DummyPredictor()
    
    # Create config
    config = AttackConfig(
        attack_type=args.attack if not args.ensemble else "all",
        use_ensemble=args.ensemble,
        num_strategies=args.num_strategies,
        max_toxicity_score=args.max_toxicity,
        min_semantic_similarity=args.min_similarity,
        target_doom_increase=20.0,
        batch_size=args.batch_size,
    )
    
    # Initialize integrator
    integrator = TextAttackFullIntegrator(predictor, config)
    
    # Process input
    if args.input:
        # Single text
        if args.ensemble:
            results = integrator.ensemble_attack(args.input, args.num_strategies)
            best_result = results[0]
        else:
            best_result = integrator.attack_single(args.input, target_doom=args.target_doom)
        
        print("\n" + "=" * 60)
        print("ATTACK RESULT")
        print("=" * 60)
        print(f"Original: {best_result.original_text}")
        print(f"Attacked: {best_result.attacked_text}")
        print(f"Doom: {best_result.original_doom_score:.1f} → {best_result.attacked_doom_score:.1f} (+{best_result.doom_uplift:.1f})")
        print(f"Strategy: {best_result.attack_strategy}")
        print(f"Success: {best_result.attack_success}")
        print("=" * 60)
        
        integrator.save_cache(args.output)
        
    elif args.input_file:
        # Batch processing
        df = pd.read_csv(args.input_file)
        texts = df['text'].tolist() if 'text' in df.columns else df.iloc[:, 0].tolist()
        
        if args.ensemble:
            logger.warning("Ensemble mode disabled for batch processing (use single strategy)")
        
        results = integrator.attack_batch(texts, attack_strategy=args.attack if not args.ensemble else None)
        
        # Save results
        output_df = pd.DataFrame([asdict(r) for r in results])
        output_df.to_csv(args.output.replace('.json', '.csv'), index=False)
        integrator.save_cache(args.output)
        
        print(f"\nProcessed {len(results)} samples")
        print(f"Success rate: {integrator.get_statistics()['success_rate']:.1%}")
    
    else:
        # Demo mode
        demo_texts = [
            "I think this celebrity is overrated and should be cancelled.",
            "The politician's recent statement was completely unacceptable.",
            "This company's practices are harmful and need to stop.",
        ]
        
        logger.info("Running demo attacks...")
        for text in demo_texts:
            result = integrator.attack_single(text)
            print(f"\nOriginal: {text}")
            print(f"Attacked: {result.attacked_text}")
            print(f"Doom uplift: +{result.doom_uplift:.1f}")
    
    # Print statistics
    stats = integrator.get_statistics()
    print("\n" + "=" * 60)
    print("ATTACK STATISTICS")
    print("=" * 60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
