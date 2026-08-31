"""Adversarial Attack Generator for Doom Index.

Generates text variants that maximize predicted doom score while
staying under moderation toxicity thresholds.

Supports:
- TextAttack integration (WordSwapEmbedding, WordSwapWordNet)
- Custom mutation strategies (emoji, punctuation, framing)
- Genetic algorithm optimization
- Perspective API moderation proxy
"""

import logging
import random
import re
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy

import numpy as np

logger = logging.getLogger(__name__)

# Try to import TextAttack
try:
    from textattack.attack_recipes import TextFoolerJin2019
    from textattack.constraints.pre_transformation import StopwordModification
    from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
    from textattack.transformations import WordSwapEmbedding, WordSwapWordNet
    from textattack.augmentation import EasyDataAugmenter
    TEXTATTACK_AVAILABLE = True
except ImportError:
    TEXTATTACK_AVAILABLE = False
    logger.warning("TextAttack not installed. Using fallback strategies only.")


@dataclass
class AttackResult:
    """Result of a single adversarial attack."""
    variant_text: str
    original_doom: float
    attacked_doom: float
    doom_uplift: float
    toxicity_score: float
    strategy: str
    semantic_similarity: float = 1.0
    passes_moderation: bool = True


class AdversarialGenerator:
    """Generate adversarial text variants for doom score maximization."""

    def __init__(
        self,
        predictor,  # IntegratedDoomPredictor instance
        toxicity_proxy: Optional[Callable] = None,
        max_iterations: int = 50,
        population_size: int = 20,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        elite_size: int = 3,
    ):
        self.predictor = predictor
        self.toxicity_proxy = toxicity_proxy or self._default_toxicity_proxy
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

        # Strategy pool
        self.strategies = self._build_strategy_pool()

        logger.info(f"AdversarialGenerator initialized with {len(self.strategies)} strategies")

    def _build_strategy_pool(self) -> Dict[str, Callable]:
        """Build pool of mutation strategies."""
        strategies = {
            'emoji_injection': self._emoji_injection,
            'outrage_punctuation': self._outrage_punctuation,
            'rhetorical_question': self._rhetorical_question,
            'exaggeration': self._exaggeration,
            'controversy_frame': self._controversy_frame,
            'call_to_action': self._call_to_action,
            'authority_challenge': self._authority_challenge,
            'passive_active_voice': self._passive_active_voice,
            'loaded_language': self._loaded_language,
            'us_vs_them': self._us_vs_them,
        }

        # Add TextAttack strategies if available
        if TEXTATTACK_AVAILABLE:
            strategies['textattack_wordnet'] = self._textattack_wordnet
            strategies['textattack_embedding'] = self._textattack_embedding

        return strategies

    # ── Individual Mutation Strategies ──────────────────────────────────────

    def _emoji_injection(self, text: str, intensity: float = 1.0) -> str:
        """Add engagement-boosting emojis without increasing toxicity."""
        emojis = ["😤", "💀", "🔥", "😡", "🤬", "👀", "😠", "⚠️", "🚨", "❗"]
        n = max(1, int(intensity * random.randint(1, 3)))
        selected = random.sample(emojis, min(n, len(emojis)))
        return text + " " + " ".join(selected)

    def _outrage_punctuation(self, text: str, intensity: float = 1.0) -> str:
        """Amplify punctuation for urgency cues."""
        text = re.sub(r'\.+', '!!!', text)
        text = re.sub(r'!+', '!!!', text)
        if '!' not in text:
            text = text.rstrip('.') + "!!!"
        return text

    def _rhetorical_question(self, text: str, intensity: float = 1.0) -> str:
        """Convert statements to rhetorical questions."""
        if text.endswith('?'):
            return text + " Don't you see the problem?"
        return text + " Isn't it obvious?"

    def _exaggeration(self, text: str, intensity: float = 1.0) -> str:
        """Amplify intensity words."""
        replacements = {
            r'\bvery\b': 'extremely',
            r'\bsome\b': 'countless',
            r'\bmany\b': 'overwhelming',
            r'\ba few\b': 'massive numbers of',
            r'\bimportant\b': 'critical',
            r'\bbad\b': 'disastrous',
            r'\bwrong\b': 'catastrophically wrong',
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _controversy_frame(self, text: str, intensity: float = 1.0) -> str:
        """Add controversy framing prefixes."""
        prefixes = [
            "BREAKING:",
            "EXPOSED:",
            "SHOCKING:",
            "You won't believe this —",
            "The truth about",
        ]
        prefix = random.choice(prefixes)
        if not any(text.startswith(p) for p in prefixes):
            text = f"{prefix} {text}"
        return text

    def _call_to_action(self, text: str, intensity: float = 1.0) -> str:
        """Add engagement-boosting CTAs."""
        ctas = [
            " Retweet if you agree.",
            " Share if this matters to you.",
            " Like if you're outraged.",
            " Comment your thoughts below.",
        ]
        return text + random.choice(ctas)

    def _authority_challenge(self, text: str, intensity: float = 1.0) -> str:
        """Frame against authority to trigger anti-establishment response."""
        frames = [
            " The establishment doesn't want you to know this.",
            " They're trying to silence this.",
            " Mainstream media won't report this.",
        ]
        return text + random.choice(frames)

    def _passive_active_voice(self, text: str, intensity: float = 1.0) -> str:
        """Convert passive to active voice where possible."""
        # Simple heuristic replacements
        text = re.sub(r'was attacked by', 'faces backlash from', text, flags=re.IGNORECASE)
        text = re.sub(r'is being criticized', 'faces mounting criticism', text, flags=re.IGNORECASE)
        text = re.sub(r'has been accused', 'stands accused', text, flags=re.IGNORECASE)
        return text

    def _loaded_language(self, text: str, intensity: float = 1.0) -> str:
        """Replace neutral words with emotionally loaded alternatives."""
        replacements = {
            r'\bsaid\b': 'claimed',
            r'\bclaimed\b': 'insisted',
            r'\bdefended\b': 'desperately defended',
            r'\bresponded\b': 'lashed out',
            r'\bapologized\b': 'was forced to apologize',
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _us_vs_them(self, text: str, intensity: float = 1.0) -> str:
        """Create in-group/out-group framing."""
        frames = [
            " Real people know the truth.",
            " The silent majority agrees.",
            " They think we're stupid.",
        ]
        return text + random.choice(frames)

    def _textattack_wordnet(self, text: str, intensity: float = 1.0) -> str:
        """Use TextAttack WordNet synonym swap."""
        if not TEXTATTACK_AVAILABLE:
            return text

        augmenter = EasyDataAugmenter(
            transformations_per_example=1,
            pct_words_to_swap=0.2 * intensity,
        )
        try:
            results = augmenter.augment(text)
            return results[0] if results else text
        except Exception:
            return text

    def _textattack_embedding(self, text: str, intensity: float = 1.0) -> str:
        """Use TextAttack embedding-based word swap."""
        if not TEXTATTACK_AVAILABLE:
            return text

        # Simplified — full TextAttack integration would use Attack class
        return self._textattack_wordnet(text, intensity)

    # ── Core Generation Logic ───────────────────────────────────────────────

    def generate_variants(
        self,
        text: str,
        author_id: str = "anonymous",
        max_variants: int = 5,
        toxicity_budget: float = 0.7,
        use_genetic: bool = False,
    ) -> List[AttackResult]:
        """Generate adversarial variants.

        Args:
            text: Original text
            author_id: Author identifier
            max_variants: Max number of variants to return
            toxicity_budget: Max allowed toxicity score
            use_genetic: Use genetic algorithm (slower but better)

        Returns:
            List of AttackResult, sorted by doom_uplift descending
        """
        # Get baseline
        baseline = self.predictor.predict(text, author_id)
        original_doom = baseline['probability']

        logger.info(f"Original doom score: {original_doom:.4f}")

        if use_genetic:
            variants = self._genetic_optimize(
                text, author_id, original_doom, 
                max_variants, toxicity_budget
            )
        else:
            variants = self._greedy_optimize(
                text, author_id, original_doom,
                max_variants, toxicity_budget
            )

        # Sort by doom uplift
        variants.sort(key=lambda v: v.doom_uplift, reverse=True)

        return variants[:max_variants]

    def _greedy_optimize(
        self,
        text: str,
        author_id: str,
        original_doom: float,
        max_variants: int,
        toxicity_budget: float,
    ) -> List[AttackResult]:
        """Greedy strategy: apply each strategy, keep best."""
        variants = []
        used_strategies = set()

        for strategy_name, strategy_fn in self.strategies.items():
            if len(variants) >= max_variants * 2:  # Generate extra, filter later
                break

            try:
                # Try different intensities
                for intensity in [0.5, 1.0, 1.5]:
                    variant_text = strategy_fn(text, intensity)

                    if variant_text == text:
                        continue

                    # Evaluate
                    result = self.predictor.predict(variant_text, author_id)
                    toxicity = self.toxicity_proxy(variant_text)

                    if toxicity > toxicity_budget:
                        continue

                    # Semantic similarity (simple Jaccard as proxy)
                    similarity = self._semantic_similarity(text, variant_text)

                    variants.append(AttackResult(
                        variant_text=variant_text,
                        original_doom=original_doom,
                        attacked_doom=result['probability'],
                        doom_uplift=result['probability'] - original_doom,
                        toxicity_score=toxicity,
                        strategy=strategy_name,
                        semantic_similarity=similarity,
                        passes_moderation=toxicity <= toxicity_budget,
                    ))

                    used_strategies.add(strategy_name)

            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed: {e}")
                continue

        return variants

    def _genetic_optimize(
        self,
        text: str,
        author_id: str,
        original_doom: float,
        max_variants: int,
        toxicity_budget: float,
    ) -> List[AttackResult]:
        """Genetic algorithm for adversarial optimization."""

        # Initialize population with strategy-applied texts
        population = []
        for _ in range(self.population_size):
            strategy = random.choice(list(self.strategies.values()))
            individual = strategy(text, random.uniform(0.5, 2.0))
            population.append(individual)

        best_individuals = []

        for generation in range(self.max_iterations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                try:
                    result = self.predictor.predict(individual, author_id)
                    toxicity = self.toxicity_proxy(individual)

                    if toxicity > toxicity_budget:
                        fitness = -1.0  # Invalid
                    else:
                        # Fitness = doom uplift - lambda * (1 - similarity)
                        similarity = self._semantic_similarity(text, individual)
                        fitness = (result['probability'] - original_doom) - 0.1 * (1 - similarity)

                    fitness_scores.append(fitness)
                except Exception:
                    fitness_scores.append(-1.0)

            # Track best
            for i, (ind, fit) in enumerate(zip(population, fitness_scores)):
                if fit > 0:
                    result = self.predictor.predict(ind, author_id)
                    toxicity = self.toxicity_proxy(ind)
                    best_individuals.append(AttackResult(
                        variant_text=ind,
                        original_doom=original_doom,
                        attacked_doom=result['probability'],
                        doom_uplift=result['probability'] - original_doom,
                        toxicity_score=toxicity,
                        strategy=f"genetic_gen{generation}",
                        semantic_similarity=self._semantic_similarity(text, ind),
                    ))

            # Selection (tournament)
            selected = self._tournament_select(population, fitness_scores)

            # Crossover
            offspring = self._crossover(selected, text)

            # Mutation
            offspring = self._mutate(offspring)

            # Elitism
            sorted_indices = np.argsort(fitness_scores)[::-1]
            elites = [population[i] for i in sorted_indices[:self.elite_size]]

            population = elites + offspring[:self.population_size - self.elite_size]

        return best_individuals

    def _tournament_select(self, population: List[str], fitness: List[float], k: int = 3) -> List[str]:
        """Tournament selection."""
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(list(zip(population, fitness)), min(k, len(population)))
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected

    def _crossover(self, parents: List[str], original: str) -> List[str]:
        """Crossover by combining sentences from parents."""
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i], parents[i + 1]

            # Simple sentence-level crossover
            s1 = p1.split('. ')
            s2 = p2.split('. ')

            if len(s1) > 1 and len(s2) > 1:
                split = random.randint(1, min(len(s1), len(s2)) - 1)
                child = '. '.join(s1[:split] + s2[split:])
            else:
                child = p1 if random.random() > 0.5 else p2

            offspring.append(child)

        return offspring

    def _mutate(self, population: List[str]) -> List[str]:
        """Apply random mutations."""
        mutated = []
        for individual in population:
            if random.random() < self.mutation_rate:
                strategy = random.choice(list(self.strategies.values()))
                individual = strategy(individual, random.uniform(0.3, 1.5))
            mutated.append(individual)
        return mutated

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity as proxy for semantic similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _default_toxicity_proxy(self, text: str) -> float:
        """Default toxicity proxy using simple heuristics.

        In production, replace with Perspective API call.
        """
        score = 0.0
        text_lower = text.lower()

        # Profanity list (simplified)
        profanity = ['damn', 'hell', 'stupid', 'idiot', 'moron', 'hate', 'kill']
        score += sum(0.1 for word in profanity if word in text_lower)

        # All caps ratio
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        score += caps_ratio * 0.3

        # Exclamation density
        excl_ratio = text.count('!') / max(len(text.split()), 1)
        score += excl_ratio * 0.2

        return min(score, 1.0)


# ── Convenience wrapper ─────────────────────────────────────────────────────

def generate_attacks(
    text: str,
    predictor,
    author_id: str = "anonymous",
    max_variants: int = 5,
    toxicity_budget: float = 0.7,
    use_genetic: bool = False,
) -> List[AttackResult]:
    """Convenience function to generate attacks."""
    generator = AdversarialGenerator(predictor)
    return generator.generate_variants(
        text=text,
        author_id=author_id,
        max_variants=max_variants,
        toxicity_budget=toxicity_budget,
        use_genetic=use_genetic,
    )


if __name__ == "__main__":
    print("Adversarial Generator module. Import and use with predictor.")
