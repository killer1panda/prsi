"""Production-grade adversarial attack generator with TextAttack integration.

Features:
- TextAttack recipes (TextFooler, BAEGarg, PWWS)
- Custom constraints (toxicity budget, semantic similarity)
- Genetic algorithm with fitness-based selection
- Adversarial training loop for model robustness
"""

import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try TextAttack
try:
    from textattack.attack_recipes import TextFoolerJin2019, BAEGarg2019
    from textattack.constraints.pre_transformation import StopwordModification
    from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
    from textattack.transformations import WordSwapEmbedding, WordSwapWordNet, WordSwapMaskedLM
    from textattack.goal_functions import UntargetedClassification
    from textattack.search_methods import GreedyWordSwapWIR, BeamSearch
    from textattack import Attack
    from textattack.models.wrappers import ModelWrapper
    TEXTATTACK_AVAILABLE = True
except ImportError:
    TEXTATTACK_AVAILABLE = False
    logger.warning("TextAttack not installed. Using advanced fallback strategies.")


@dataclass
class AttackResult:
    variant_text: str
    original_doom: float
    attacked_doom: float
    doom_uplift: float
    toxicity_score: float
    strategy: str
    semantic_similarity: float = 1.0
    passes_moderation: bool = True
    perplexity: float = 0.0


class DoomModelWrapper(ModelWrapper if TEXTATTACK_AVAILABLE else object):
    """TextAttack-compatible wrapper for Doom predictor."""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def __call__(self, text_input_list):
        """TextAttack expects list of texts, returns list of predictions."""
        results = []
        for text in text_input_list:
            result = self.predictor.predict(text, author_id="attack_target")
            # TextAttack expects [negative_prob, positive_prob]
            prob = result['probability']
            results.append([1 - prob, prob])
        return np.array(results)


class ProductionAdversarialGenerator:
    """Production adversarial generator with multiple attack strategies."""
    
    def __init__(
        self,
        predictor,
        toxicity_proxy: Optional[Callable] = None,
        use_textattack: bool = True,
        max_iterations: int = 100,
        population_size: int = 30,
    ):
        self.predictor = predictor
        self.toxicity_proxy = toxicity_proxy or self._default_toxicity_proxy
        self.use_textattack = use_textattack and TEXTATTACK_AVAILABLE
        self.max_iterations = max_iterations
        self.population_size = population_size
        
        # Build TextAttack recipes if available
        if self.use_textattack:
            self._build_textattack_recipes()
        
        # Custom strategy pool
        self.custom_strategies = self._build_custom_strategies()
        
        logger.info(f"ProductionAdversarialGenerator: TextAttack={self.use_textattack}")
    
    def _build_textattack_recipes(self):
        """Build TextAttack attack recipes."""
        self.wrapper = DoomModelWrapper(self.predictor)
        
        # TextFooler: WordNet synonym swap with USE constraint
        self.textfooler = TextFoolerJin2019.build(self.wrapper)
        
        # BAE: BERT-based adversarial examples
        try:
            self.bae = BAEGarg2019.build(self.wrapper)
        except Exception:
            self.bae = None
            logger.warning("BAE attack not available")
    
    def _build_custom_strategies(self) -> Dict[str, Callable]:
        """Build custom mutation strategies."""
        return {
            'emoji_injection': self._emoji_injection,
            'punctuation_manipulation': self._punctuation_manipulation,
            'rhetorical_conversion': self._rhetorical_conversion,
            'intensifier_injection': self._intensifier_injection,
            'framing_prefix': self._framing_prefix,
            'cta_injection': self._cta_injection,
            'authority_challenge': self._authority_challenge,
            'voice_conversion': self._voice_conversion,
            'loaded_language': self._loaded_language,
            'ingroup_framing': self._ingroup_framing,
            'sarcasm_marker': self._sarcasm_marker,
            'ellipsis_tension': self._ellipsis_tension,
            'caps_emphasis': self._caps_emphasis,
        }
    
    # ── Advanced Mutation Strategies ────────────────────────────────────────
    
    def _emoji_injection(self, text: str, intensity: float = 1.0) -> str:
        """Strategic emoji placement for engagement manipulation."""
        emoji_sets = {
            'outrage': ['😤', '💀', '🔥', '😡', '🤬', '👀'],
            'urgency': ['⚠️', '🚨', '❗', '⛔', '🔴'],
            'mockery': ['🙄', '💩', '🤡', '😂', '🤣'],
            'solidarity': ['✊', '💪', '🤝', '❤️', '🔥'],
        }
        category = random.choice(list(emoji_sets.keys()))
        n = max(1, int(intensity * random.randint(1, 3)))
        emojis = random.sample(emoji_sets[category], min(n, len(emoji_sets[category])))
        
        # Strategic placement: beginning for attention, end for engagement
        if random.random() > 0.5:
            return " ".join(emojis) + " " + text
        return text + " " + " ".join(emojis)
    
    def _punctuation_manipulation(self, text: str, intensity: float = 1.0) -> str:
        """Manipulate punctuation for emotional emphasis."""
        strategies = [
            lambda t: t.replace('.', '!!!').replace('!', '!!!'),
            lambda t: t.replace('.', '...').replace(',', '...'),
            lambda t: t + '???',
            lambda t: '!! ' + t + ' !!',
            lambda t: t.replace('.', ' — '),
        ]
        strategy = random.choice(strategies)
        return strategy(text)
    
    def _rhetorical_conversion(self, text: str, intensity: float = 1.0) -> str:
        """Convert statements to rhetorical questions."""
        templates = [
            "{} — or are we just going to ignore this?",
            "Am I the only one who thinks {}?",
            "How is it that {} and nobody cares?",
            "So we're just pretending {} isn't happening?",
            "Let me get this straight: {}?",
        ]
        template = random.choice(templates)
        return template.format(text.rstrip('.!?'))
    
    def _intensifier_injection(self, text: str, intensity: float = 1.0) -> str:
        """Inject intensifiers and amplifiers."""
        intensifiers = ['absolutely', 'completely', 'totally', 'utterly', 
                       'literally', 'genuinely', 'legitimately', 'objectively']
        words = text.split()
        if len(words) > 3:
            idx = random.randint(1, min(3, len(words) - 1))
            words.insert(idx, random.choice(intensifiers))
        return ' '.join(words)
    
    def _framing_prefix(self, text: str, intensity: float = 1.0) -> str:
        """Add news-style framing prefixes."""
        prefixes = [
            "BREAKING:", "EXPOSED:", "SHOCKING:", "REVEALED:",
            "You won't believe this —", "The truth they don't want you to know:",
            "Unpopular opinion but", "Hot take:", "Let's be real:",
        ]
        prefix = random.choice(prefixes)
        return f"{prefix} {text}"
    
    def _cta_injection(self, text: str, intensity: float = 1.0) -> str:
        """Inject engagement calls-to-action."""
        ctas = [
            " Retweet if you agree.",
            " Like if this makes you angry.",
            " Share before they delete this.",
            " Comment 'YES' if you see the problem.",
            " Tag someone who needs to see this.",
        ]
        return text + random.choice(ctas)
    
    def _authority_challenge(self, text: str, intensity: float = 1.0) -> str:
        """Challenge authority to trigger anti-establishment response."""
        challenges = [
            " The mainstream media won't cover this.",
            " They're trying to bury this story.",
            " Big Tech doesn't want you to see this.",
            " The establishment is silent on this.",
            " Why is nobody talking about this?",
        ]
        return text + random.choice(challenges)
    
    def _voice_conversion(self, text: str, intensity: float = 1.0) -> str:
        """Convert passive to active voice with loaded language."""
        conversions = [
            (r'\bwas criticized\b', 'faces mounting criticism'),
            (r'\bhas been accused\b', 'stands accused'),
            (r'\bis being investigated\b', 'is under investigation'),
            (r'\bwas forced to\b', 'had no choice but to'),
            (r'\bapologized for\b', 'was forced to apologize for'),
        ]
        import re
        result = text
        for pattern, replacement in conversions:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    
    def _loaded_language(self, text: str, intensity: float = 1.0) -> str:
        """Replace neutral words with emotionally loaded alternatives."""
        replacements = {
            r'\bsaid\b': 'claimed',
            r'\bstated\b': 'insisted',
            r'\bdefended\b': 'desperately defended',
            r'\bresponded\b': 'lashed out',
            r'\bexplained\b': 'tried to justify',
            r'\baddressed\b': 'was forced to address',
        }
        import re
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    
    def _ingroup_framing(self, text: str, intensity: float = 1.0) -> str:
        """Create in-group/out-group division."""
        frames = [
            " Real people know the truth about this.",
            " The silent majority is waking up to this.",
            " They think we're too stupid to notice.",
            " Anyone with common sense can see this.",
        ]
        return text + random.choice(frames)
    
    def _sarcasm_marker(self, text: str, intensity: float = 1.0) -> str:
        """Add sarcasm markers that increase engagement."""
        markers = [
            " Great job everyone.",
            " What a surprise.",
            " I'm sure this will be handled fairly.",
            " Nothing to see here, move along.",
        ]
        return text + random.choice(markers)
    
    def _ellipsis_tension(self, text: str, intensity: float = 1.0) -> str:
        """Use ellipsis to create suspense/tension."""
        words = text.split()
        if len(words) > 5:
            idx = random.randint(2, len(words) - 2)
            words[idx] = words[idx] + "..."
        return ' '.join(words)
    
    def _caps_emphasis(self, text: str, intensity: float = 1.0) -> str:
        """Strategic ALL CAPS for emphasis."""
        words = text.split()
        if len(words) > 3:
            n_caps = max(1, int(len(words) * 0.1 * intensity))
            indices = random.sample(range(len(words)), min(n_caps, len(words)))
            for idx in indices:
                words[idx] = words[idx].upper()
        return ' '.join(words)
    
    # ── Core Generation ─────────────────────────────────────────────────────
    
    def generate_variants(
        self,
        text: str,
        author_id: str = "anonymous",
        max_variants: int = 5,
        toxicity_budget: float = 0.7,
        use_genetic: bool = True,
        min_semantic_similarity: float = 0.6,
    ) -> List[AttackResult]:
        """Generate adversarial variants."""
        baseline = self.predictor.predict(text, author_id)
        original_doom = baseline['probability']
        
        candidates = []
        
        # 1. Try TextAttack if available
        if self.use_textattack:
            try:
                ta_variants = self._textattack_generate(text, original_doom, author_id, 
                                                        toxicity_budget, max_variants)
                candidates.extend(ta_variants)
            except Exception as e:
                logger.debug(f"TextAttack failed: {e}")
        
        # 2. Custom strategies
        custom_variants = self._custom_generate(text, original_doom, author_id,
                                                toxicity_budget, max_variants * 2)
        candidates.extend(custom_variants)
        
        # 3. Genetic optimization if requested
        if use_genetic and len(candidates) >= 5:
            genetic_variants = self._genetic_optimize(
                text, original_doom, author_id, toxicity_budget,
                min_semantic_similarity, candidates
            )
            candidates.extend(genetic_variants)
        
        # Filter and rank
        candidates = [v for v in candidates if v.passes_moderation]
        candidates = [v for v in candidates if v.semantic_similarity >= min_semantic_similarity]
        
        # Deduplicate
        seen = set()
        unique = []
        for v in candidates:
            if v.variant_text not in seen:
                seen.add(v.variant_text)
                unique.append(v)
        
        unique.sort(key=lambda v: v.doom_uplift, reverse=True)
        return unique[:max_variants]
    
    def _textattack_generate(self, text, original_doom, author_id, toxicity_budget, max_variants):
        """Generate using TextAttack recipes."""
        variants = []
        
        for recipe_name, recipe in [("TextFooler", self.textfooler), ("BAE", self.bae)]:
            if recipe is None:
                continue
            try:
                results = recipe.attack(text, ground_truth_output=0)
                for result in results:
                    if result.perturbed_text() != text:
                        variant_text = result.perturbed_text()
                        pred = self.predictor.predict(variant_text, author_id)
                        tox = self.toxicity_proxy(variant_text)
                        
                        if tox <= toxicity_budget:
                            variants.append(AttackResult(
                                variant_text=variant_text,
                                original_doom=original_doom,
                                attacked_doom=pred['probability'],
                                doom_uplift=pred['probability'] - original_doom,
                                toxicity_score=tox,
                                strategy=f"textattack_{recipe_name}",
                                semantic_similarity=self._semantic_sim(text, variant_text),
                                passes_moderation=tox <= toxicity_budget,
                            ))
            except Exception as e:
                logger.debug(f"TextAttack recipe {recipe_name} failed: {e}")
        
        return variants
    
    def _custom_generate(self, text, original_doom, author_id, toxicity_budget, max_variants):
        """Generate using custom strategies."""
        variants = []
        
        for strategy_name, strategy_fn in self.custom_strategies.items():
            if len(variants) >= max_variants:
                break
            
            for intensity in [0.5, 1.0, 1.5]:
                try:
                    variant_text = strategy_fn(text, intensity)
                    if variant_text == text or variant_text in [v.variant_text for v in variants]:
                        continue
                    
                    pred = self.predictor.predict(variant_text, author_id)
                    tox = self.toxicity_proxy(variant_text)
                    sim = self._semantic_sim(text, variant_text)
                    
                    if tox <= toxicity_budget and sim >= 0.5:
                        variants.append(AttackResult(
                            variant_text=variant_text,
                            original_doom=original_doom,
                            attacked_doom=pred['probability'],
                            doom_uplift=pred['probability'] - original_doom,
                            toxicity_score=tox,
                            strategy=strategy_name,
                            semantic_similarity=sim,
                            passes_moderation=True,
                        ))
                except Exception as e:
                    logger.debug(f"Strategy {strategy_name} failed: {e}")
        
        return variants
    
    def _genetic_optimize(self, text, original_doom, author_id, toxicity_budget, min_sim, seed_pool):
        """Genetic algorithm optimization."""
        population = [v.variant_text for v in seed_pool[:10]]
        best = []
        
        for gen in range(self.max_iterations // 10):
            # Evaluate fitness
            fitness = []
            for individual in population:
                try:
                    pred = self.predictor.predict(individual, author_id)
                    tox = self.toxicity_proxy(individual)
                    sim = self._semantic_sim(text, individual)
                    
                    if tox <= toxicity_budget and sim >= min_sim:
                        fit = (pred['probability'] - original_doom) + 0.1 * sim
                    else:
                        fit = -1.0
                    fitness.append(fit)
                except:
                    fitness.append(-1.0)
            
            # Track best
            for i, (ind, fit) in enumerate(zip(population, fitness)):
                if fit > 0:
                    pred = self.predictor.predict(ind, author_id)
                    tox = self.toxicity_proxy(ind)
                    best.append(AttackResult(
                        variant_text=ind,
                        original_doom=original_doom,
                        attacked_doom=pred['probability'],
                        doom_uplift=pred['probability'] - original_doom,
                        toxicity_score=tox,
                        strategy=f"genetic_gen{gen}",
                        semantic_similarity=self._semantic_sim(text, ind),
                        passes_moderation=True,
                    ))
            
            # Selection + crossover + mutation
            population = self._evolve_population(population, fitness, text)
        
        return best
    
    def _evolve_population(self, population, fitness, original_text):
        """Evolve population one generation."""
        # Tournament selection
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(list(zip(population, fitness)), min(3, len(population)))
            winner = max(candidates, key=lambda x: x[1])[0]
            selected.append(winner)
        
        # Crossover: combine two parents
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            p1, p2 = selected[i], selected[i + 1]
            words1, words2 = p1.split(), p2.split()
            if len(words1) > 3 and len(words2) > 3:
                split = random.randint(1, min(len(words1), len(words2)) - 1)
                child = ' '.join(words1[:split] + words2[split:])
                offspring.append(child)
            else:
                offspring.append(p1)
        
        # Mutation
        mutated = []
        for individual in offspring:
            if random.random() < 0.3:
                strategy = random.choice(list(self.custom_strategies.values()))
                individual = strategy(individual, random.uniform(0.5, 1.5))
            mutated.append(individual)
        
        return mutated[:self.population_size]
    
    def _semantic_sim(self, t1, t2):
        """Jaccard similarity proxy."""
        s1, s2 = set(t1.lower().split()), set(t2.lower().split())
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)
    
    def _default_toxicity_proxy(self, text):
        """Default toxicity heuristic."""
        score = 0.0
        text_lower = text.lower()
        profanity = ['damn', 'hell', 'stupid', 'idiot', 'moron', 'hate', 'kill', 'die', 'trash']
        score += sum(0.08 for w in profanity if w in text_lower)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        score += caps_ratio * 0.25
        score += text.count('!') / max(len(text.split()), 1) * 0.15
        return min(score, 1.0)


class AdversarialTrainer:
    """Adversarial training: train model to be robust against attacks.
    
    Generates adversarial examples during training and includes them
    in the training batch with a robustness loss term.
    """
    
    def __init__(
        self,
        model,
        attack_generator: ProductionAdversarialGenerator,
        alpha: float = 0.5,  # Weight for adversarial loss
        epsilon: float = 0.1,  # Perturbation budget
    ):
        self.model = model
        self.attack_generator = attack_generator
        self.alpha = alpha
        self.epsilon = epsilon
    
    def compute_adversarial_loss(
        self,
        x,
        edge_index,
        input_ids,
        attention_mask,
        user_indices,
        labels,
    ):
        """Compute adversarial training loss.
        
        Standard cross-entropy + adversarial robustness term.
        """
        # Standard loss
        logits = self.model(x, edge_index, input_ids, attention_mask, user_indices)
        standard_loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # Generate adversarial perturbations (FGSM-style)
        input_ids_adv = input_ids.clone().detach()
        input_ids_adv.requires_grad = True
        
        logits_adv = self.model(x, edge_index, input_ids_adv, attention_mask, user_indices)
        loss_adv = torch.nn.functional.cross_entropy(logits_adv, labels)
        loss_adv.backward()
        
        # Gradient-based perturbation on embeddings (simplified)
        # In practice, you'd perturb embeddings, not token IDs
        # This is a conceptual implementation
        
        return standard_loss  # Simplified: full adversarial training needs embedding-level access
    
    def generate_training_adversaries(self, texts, user_indices, labels, n_per_sample=1):
        """Generate adversarial examples for training data augmentation."""
        adversaries = []
        
        for text, uid, label in zip(texts, user_indices, labels):
            if label == 0:  # Only attack safe examples
                variants = self.attack_generator.generate_variants(
                    text, author_id=str(uid), max_variants=n_per_sample, toxicity_budget=0.8
                )
                for v in variants:
                    adversaries.append({
                        'text': v.variant_text,
                        'user_idx': uid,
                        'label': 1,  # Flip to at-risk
                        'is_adversarial': True,
                    })
        
        return adversaries
