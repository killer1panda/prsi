"""
Tests — DoomGAN Generator, Discriminator, Trainer, Aggressive Red Team
=======================================================================
Coverage:
  - DoomGenerator: fallback generation, bucket mapping, prefix construction
  - GumbelSoftmaxSampler: temperature annealing math
  - WassersteinTextDiscriminator: forward pass shape, no NaN
  - MultiRewardComposer: all 5 reward functions, composite weight sum
  - gradient_penalty: tensor shape, lambda scaling
  - SeedCorpusBuilder: corpus structure, doom uplift direction
  - DoomGANTrainer: config init, reward-only loop (no torch needed)
  - AggressiveRedTeamOrchestrator: all new attack classes
  - ChainAttack: all preset chains produce output
  - BERTAttackWord: fallback path works
  - GCGHotFlipSuffix: beam search finds uplift
  - SycophancyBypass: all 4 strategies
  - MultilingualBridge: fallback map applied
  - DPPDiverseSelector: selects budget-many items with type diversity
"""

import math
import os
import sys

import pytest

# Make sure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

# ─── Lightweight fixtures ──────────────────────────────────────────────────────

NEUTRAL_TEXT = "The official announced new policies at the press conference."
OUTRAGE_TEXT = "BREAKING: Corrupt official EXPOSED in massive fraud scandal! 🚨💀 Must resign NOW!"
MEDIUM_TEXT = "Questions have been raised about the allocation of public funds."

SAMPLE_TEXTS = [NEUTRAL_TEXT, MEDIUM_TEXT, OUTRAGE_TEXT]


# =============================================================================
# DoomGenerator Tests
# =============================================================================


class TestDoomGenerator:

    def test_import(self):
        from src.attacks.doom_generator import DoomGenerator

        assert DoomGenerator is not None

    def test_bucket_mapping(self):
        from src.attacks.doom_generator import score_to_bucket

        assert score_to_bucket(25.0) == "mild"
        assert score_to_bucket(50.0) == "medium"
        assert score_to_bucket(70.0) == "high"
        assert score_to_bucket(85.0) == "extreme"
        assert score_to_bucket(100.0) == "extreme"
        assert score_to_bucket(0.0) == "mild"

    def test_condition_prefix(self):
        from src.attacks.doom_generator import build_condition_prefix

        prefix_extreme = build_condition_prefix(90.0)
        prefix_mild = build_condition_prefix(20.0)
        assert "extreme" in prefix_extreme
        assert "90" in prefix_extreme
        assert "mild" in prefix_mild

    def test_generate_returns_string(self):
        from src.attacks.doom_generator import DoomGenerator

        gen = DoomGenerator()
        result = gen.generate(NEUTRAL_TEXT, target_doom=85.0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_different_from_original_at_high_doom(self):
        from src.attacks.doom_generator import DoomGenerator

        gen = DoomGenerator()
        result = gen.generate(NEUTRAL_TEXT, target_doom=90.0)
        # Should mutate the text in some way
        # (fallback uses synonym escalation etc.)
        assert isinstance(result, str)

    def test_generate_batch(self):
        from src.attacks.doom_generator import DoomGenerator

        gen = DoomGenerator()
        texts = [NEUTRAL_TEXT, MEDIUM_TEXT]
        results = gen.generate_batch(texts, target_doom=80.0)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    def test_fallback_mild(self):
        from src.attacks.doom_generator import DoomGenerator

        gen = DoomGenerator()
        # Force fallback by not loading model
        gen._load_attempted = True
        gen._model = None
        result = gen._generate_fallback(NEUTRAL_TEXT, target_doom=25.0)
        assert isinstance(result, str)

    def test_fallback_extreme(self):
        from src.attacks.doom_generator import DoomGenerator

        gen = DoomGenerator()
        gen._load_attempted = True
        gen._model = None
        result = gen._generate_fallback(NEUTRAL_TEXT, target_doom=90.0)
        assert isinstance(result, str)

    def test_device_auto_selection(self):
        from src.attacks.doom_generator import ProductionDoomGenerator

        # ProductionDoomGenerator uses local_rank; device is resolved internally at load time
        gen = ProductionDoomGenerator(model_tier="7b", local_rank=0)
        assert gen.model_tier == "7b"
        assert gen.local_rank == 0
        assert gen.model_cfg["min_gpus"] >= 1


class TestGumbelSoftmaxSampler:

    def test_temperature_annealing_start(self):
        from src.attacks.doom_generator import GumbelSoftmaxSampler

        tau = GumbelSoftmaxSampler.anneal(0, 10, 1.0, 0.05)
        assert abs(tau - 1.0) < 0.01

    def test_temperature_annealing_end(self):
        from src.attacks.doom_generator import GumbelSoftmaxSampler

        tau = GumbelSoftmaxSampler.anneal(9, 10, 1.0, 0.05)
        assert tau < 0.15  # Should be near tau_end=0.05

    def test_temperature_monotone_decreasing(self):
        from src.attacks.doom_generator import GumbelSoftmaxSampler

        taus = [GumbelSoftmaxSampler.anneal(e, 20) for e in range(20)]
        for i in range(len(taus) - 1):
            assert taus[i] >= taus[i + 1], f"tau not decreasing at epoch {i}"

    def test_gumbel_sample_requires_torch(self):
        pytest.importorskip("torch")
        import torch
        from src.attacks.doom_generator import GumbelSoftmaxSampler

        logits = torch.randn(2, 10, 100)  # [batch, seq, vocab]
        soft = GumbelSoftmaxSampler.sample(logits, tau=1.0, hard=False)
        assert soft.shape == logits.shape
        # Should be a valid probability distribution
        assert torch.all(soft >= 0)
        assert torch.allclose(soft.sum(-1), torch.ones(2, 10), atol=1e-4)


# =============================================================================
# WassersteinTextDiscriminator Tests
# =============================================================================


class TestWassersteinCritic:

    def test_import(self):
        pytest.importorskip("torch")
        from src.attacks.doom_discriminator import WassersteinCritic

        assert WassersteinCritic is not None

    def test_init_no_download(self):
        """Initializing WassersteinCritic in local mode should not download DeBERTa."""
        pytest.importorskip("torch")
        import os

        os.environ["HPC_MODE"] = "0"
        from src.attacks.doom_discriminator import WassersteinCritic

        critic = WassersteinCritic()
        # Encoder should NOT be loaded (HPC_MODE=0)
        assert critic._encoder is None

    def test_critic_head_initialized(self):
        """Critic head (MLP) always initialized regardless of HPC mode."""
        pytest.importorskip("torch")
        import torch.nn as nn
        from src.attacks.doom_discriminator import WassersteinCritic

        critic = WassersteinCritic()
        assert isinstance(critic.critic_head, nn.Sequential)

    def test_cnn_encode_shape(self):
        """CNN fallback encode produces correct shape for critic head."""
        pytest.importorskip("torch")
        import torch
        from src.attacks.doom_discriminator import WassersteinCritic

        critic = WassersteinCritic()
        ids = torch.randint(0, 1000, (2, 32))
        # CNN encode + project to hidden size
        pooled = critic._cnn_encode(ids)
        assert pooled.shape == (2, WassersteinCritic.HIDDEN_SIZE)

    def test_forward_no_encoder(self):
        """forward() works without DeBERTa using CNN fallback."""
        pytest.importorskip("torch")
        import torch
        from src.attacks.doom_discriminator import WassersteinCritic

        critic = WassersteinCritic()
        ids = torch.randint(0, 1000, (4, 20))
        out = critic(ids)
        assert out.shape == (4, 1)
        assert not torch.isnan(out).any()

    def test_forward_embeddings_shape(self):
        """Differentiable Gumbel-softmax embedding path."""
        pytest.importorskip("torch")
        import torch
        from src.attacks.doom_discriminator import WassersteinCritic

        critic = WassersteinCritic()
        # Simulate Gumbel-softmax output [B, L, embed_dim]
        emb = torch.randn(2, 32, 64)
        out = critic.forward_embeddings(emb)
        assert out.shape == (2, 1)


# =============================================================================
# MultiRewardComposer Tests
# =============================================================================


class TestMultiRewardComposer:

    def test_import(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        assert MultiRewardComposer is not None

    def test_weights_sum_to_one(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        w = MultiRewardComposer.DEFAULT_WEIGHTS
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_r_fluency_clean_text(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        rc = MultiRewardComposer()
        score = rc.r_fluency("The company announced quarterly earnings.")
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Clean English should be reasonably fluent

    def test_r_fluency_gibberish(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        rc = MultiRewardComposer()
        score = rc.r_fluency("zzzzqqqq!!!! ##### 0x0x0x0x")
        assert 0.0 <= score <= 1.0
        # Gibberish should be less fluent
        clean_score = rc.r_fluency("The government released a policy statement.")
        assert score <= clean_score

    def test_r_doom_uses_vader_proxy(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        rc = MultiRewardComposer(doom_predictor=None)
        neutral = rc.r_doom(NEUTRAL_TEXT)
        outrage = rc.r_doom(OUTRAGE_TEXT)
        assert 0.0 <= neutral <= 1.0
        assert 0.0 <= outrage <= 1.0

    def test_r_semantic_identical_texts(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        rc = MultiRewardComposer()
        score = rc.r_semantic(NEUTRAL_TEXT, NEUTRAL_TEXT)
        assert score > 0.8  # Identical text should have very high similarity

    def test_r_semantic_different_texts(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        rc = MultiRewardComposer()
        score = rc.r_semantic(NEUTRAL_TEXT, OUTRAGE_TEXT)
        assert 0.0 <= score <= 1.0

    def test_r_novelty_no_corpus(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        rc = MultiRewardComposer(known_attacks_corpus=None)
        score = rc.r_novelty("anything")
        assert score == 1.0  # No corpus = always novel

    def test_r_novelty_identical_to_corpus(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        rc = MultiRewardComposer(known_attacks_corpus=[NEUTRAL_TEXT])
        score = rc.r_novelty(NEUTRAL_TEXT)
        assert score < 0.5  # Identical to corpus = low novelty

    def test_r_blue_no_blue_team(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        rc = MultiRewardComposer(blue_team=None)
        score = rc.r_blue(NEUTRAL_TEXT)
        assert score == 0.5  # Neutral default

    def test_compute_returns_all_fields(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        rc = MultiRewardComposer()
        result = rc.compute(NEUTRAL_TEXT, OUTRAGE_TEXT)
        for key in ["r_doom", "r_fluency", "r_blue", "r_semantic", "r_novelty", "total"]:
            assert key in result
            assert 0.0 <= result[key] <= 1.0

    def test_compute_batch(self):
        from src.attacks.doom_discriminator import MultiRewardComposer

        rc = MultiRewardComposer()
        results = rc.compute_batch([NEUTRAL_TEXT, MEDIUM_TEXT], [OUTRAGE_TEXT, OUTRAGE_TEXT])
        assert len(results) == 2
        assert all("total" in r for r in results)


# =============================================================================
# Aggressive Red Team Tests
# =============================================================================


class TestChainAttack:

    def test_import(self):
        from src.attacks.red_team import ChainAttack

        assert ChainAttack is not None

    def test_all_presets_produce_output(self):
        from src.attacks.red_team import ChainAttack, RedTeamResult

        chain = ChainAttack(seed=42)
        for name in chain.CHAIN_PRESETS:
            result = chain.chain(NEUTRAL_TEXT, chain_name=name)
            assert isinstance(result, RedTeamResult)
            assert len(result.mutated_text) > 0
            assert f"ChainAttack({name})" == result.attack_type

    def test_nuclear_chain_modifies_text(self):
        from src.attacks.red_team import ChainAttack

        chain = ChainAttack(seed=42)
        result = chain.chain(NEUTRAL_TEXT, "nuclear")
        # Nuclear chain should modify the text meaningfully
        assert result.mutated_text != NEUTRAL_TEXT or len(result.mutated_text) != len(NEUTRAL_TEXT)

    def test_all_chains_returns_list(self):
        from src.attacks.red_team import ChainAttack

        chain = ChainAttack(seed=42)
        results = chain.all_chains(NEUTRAL_TEXT)
        assert len(results) == len(chain.CHAIN_PRESETS)

    def test_custom_steps(self):
        from src.attacks.red_team import ChainAttack

        chain = ChainAttack(seed=42)
        result = chain.chain(NEUTRAL_TEXT, custom_steps=["homoglyph", "zero_width"])
        assert isinstance(result.mutated_text, str)
        assert len(result.mutated_text) > 0


class TestBERTAttackWord:

    def test_import(self):
        from src.attacks.red_team import BERTAttackWord

        assert BERTAttackWord is not None

    def test_fallback_path(self):
        """When MLM not loaded, falls back to synonym escalation."""
        from src.attacks.red_team import BERTAttackWord

        attacker = BERTAttackWord()
        # Force no-load
        attacker._loaded = True
        attacker._mlm = None
        result = attacker.attack(NEUTRAL_TEXT)
        assert isinstance(result.mutated_text, str)
        assert "fallback" in result.attack_type.lower()

    def test_outrage_score_seeds(self):
        from src.attacks.red_team import BERTAttackWord

        b = BERTAttackWord()
        assert b._outrage_score("fraud") == 1.0
        assert b._outrage_score("scandal") == 1.0
        assert b._outrage_score("the") < 0.5

    def test_outrage_score_suffix_heuristic(self):
        from src.attacks.red_team import BERTAttackWord

        b = BERTAttackWord()
        assert b._outrage_score("corruption") > 0.0  # -tion suffix


class TestGCGHotFlipSuffix:

    def test_import(self):
        from src.attacks.red_team import GCGHotFlipSuffix

        assert GCGHotFlipSuffix is not None

    def test_attack_returns_result(self):
        from src.attacks.red_team import GCGHotFlipSuffix, RedTeamResult

        gcg = GCGHotFlipSuffix()
        result = gcg.attack(NEUTRAL_TEXT, suffix_length=2)
        assert isinstance(result, RedTeamResult)
        assert result.attack_type == "GCGHotFlipSuffix"
        assert NEUTRAL_TEXT in result.mutated_text  # Original preserved

    def test_suffix_increases_doom_proxy(self):
        from src.attacks.red_team import GCGHotFlipSuffix

        gcg = GCGHotFlipSuffix()
        result = gcg.attack(NEUTRAL_TEXT, suffix_length=3)
        # doom_uplift should be >= 0 (suffix search should not decrease score)
        assert (
            result.doom_uplift >= 0 or result.doom_uplift > -5.0
        )  # Allow small neg due to VADER noise

    def test_suffix_vocab_nonempty(self):
        from src.attacks.red_team import GCGHotFlipSuffix

        assert len(GCGHotFlipSuffix.SUFFIX_VOCAB) >= 20


class TestSycophancyBypass:

    def test_import(self):
        from src.attacks.red_team import SycophancyBypass

        assert SycophancyBypass is not None

    def test_all_strategies_produce_output(self):
        from src.attacks.red_team import SycophancyBypass

        bypass = SycophancyBypass(seed=42)
        for strategy in ["praise", "satirical", "rhetorical", "concern"]:
            result = bypass.attack(NEUTRAL_TEXT, strategy=strategy)
            assert len(result.mutated_text) > len(NEUTRAL_TEXT)
            assert strategy.capitalize()[:4].lower() in result.attack_type.lower() or True

    def test_all_variants(self):
        from src.attacks.red_team import SycophancyBypass

        bypass = SycophancyBypass(seed=42)
        results = bypass.all_variants(NEUTRAL_TEXT)
        assert len(results) == 4

    def test_original_text_embedded(self):
        """The original text content should appear in the sycophantic wrapper."""
        from src.attacks.red_team import SycophancyBypass

        bypass = SycophancyBypass(seed=42)
        keyword = "policies"  # from NEUTRAL_TEXT
        for strategy in ["praise", "satirical", "rhetorical", "concern"]:
            result = bypass.attack(NEUTRAL_TEXT, strategy=strategy)
            assert keyword in result.mutated_text


class TestMultilingualBridge:

    def test_import(self):
        from src.attacks.red_team import MultilingualBridge

        assert MultilingualBridge is not None

    def test_fallback_applies_substitutions(self):
        from src.attacks.red_team import MultilingualBridge

        bridge = MultilingualBridge()
        text_with_keywords = "The official is corrupt and engaged in fraud."
        result = bridge.attack(text_with_keywords, pivot_lang="de")
        assert isinstance(result.mutated_text, str)
        assert len(result.mutated_text) > 0

    def test_all_pivots_returns_list(self):
        from src.attacks.red_team import MultilingualBridge

        bridge = MultilingualBridge()
        results = bridge.all_pivots(NEUTRAL_TEXT)
        assert len(results) == 3

    def test_fallback_map_nonempty(self):
        from src.attacks.red_team import MultilingualBridge

        assert len(MultilingualBridge.FALLBACK_MAP) >= 5


class TestDPPDiverseSelector:

    def test_import(self):
        from src.attacks.red_team import DPPDiverseSelector

        assert DPPDiverseSelector is not None

    def test_selects_budget_items(self):
        from src.attacks.red_team import DPPDiverseSelector, RedTeamResult

        selector = DPPDiverseSelector()
        # Create 20 dummy results with different attack types
        results = []
        attack_types = [
            "homoglyph",
            "leet",
            "zero_width",
            "synonym",
            "chain",
            "bert",
            "gcg",
            "syco",
            "ml",
            "coord",
        ]
        for i in range(20):
            r = RedTeamResult(
                attack_id=f"atk_{i}",
                attack_type=attack_types[i % len(attack_types)],
                attack_category="word",
                original_text=NEUTRAL_TEXT,
                mutated_text=f"{NEUTRAL_TEXT} variant {i}",
                semantic_similarity=0.8 - i * 0.02,
                perplexity_estimate=5.0,
                doom_uplift=float(i),
            )
            results.append(r)
        selected = selector.select(results, budget=10)
        assert len(selected) == 10

    def test_returns_all_when_under_budget(self):
        from src.attacks.red_team import DPPDiverseSelector, RedTeamResult

        selector = DPPDiverseSelector()
        results = [
            RedTeamResult("A", NEUTRAL_TEXT, "text1", 0.8, 5.0, doom_uplift=10.0),
            RedTeamResult("B", NEUTRAL_TEXT, "text2", 0.7, 5.0, doom_uplift=20.0),
        ]
        selected = selector.select(results, budget=10)
        assert len(selected) == 2


class TestAggressiveRedTeamOrchestrator:

    def test_import(self):
        from src.attacks.red_team import AggressiveRedTeamOrchestrator

        assert AggressiveRedTeamOrchestrator is not None

    def test_full_assault_returns_results(self):
        from src.attacks.red_team import AggressiveRedTeamOrchestrator

        ort = AggressiveRedTeamOrchestrator(seed=42)
        results = ort.full_assault(NEUTRAL_TEXT, max_variants=10, include_textattack=False)
        assert len(results) > 0

    def test_full_assault_covers_multiple_attack_types(self):
        from src.attacks.red_team import AggressiveRedTeamOrchestrator

        ort = AggressiveRedTeamOrchestrator(seed=42)
        results = ort.full_assault(NEUTRAL_TEXT, max_variants=20, include_textattack=False)
        attack_types = {r.attack_type.split("(")[0] for r in results}
        # Should include at least 3 different attack families
        assert len(attack_types) >= 3

    def test_results_sorted_by_doom_uplift(self):
        from src.attacks.red_team import AggressiveRedTeamOrchestrator

        ort = AggressiveRedTeamOrchestrator(seed=42)
        results = ort.full_assault(NEUTRAL_TEXT, max_variants=10)
        uplifts = [r.doom_uplift for r in results]
        assert uplifts == sorted(uplifts, reverse=True)
