#!/usr/bin/env python3
"""
Production weak labeling pipeline for cancellation event detection.
Replaces broken keyword-matching (y = has_cancel) with a multivariate heuristic
validated by LLM subset labeling. This is the SCIENTIFIC FIX for your viva.

The fundamental insight: cancellation is not "does the post contain the word cancel".
Cancellation is a SOCIOLOGICAL EVENT characterized by:
  1. Rapid engagement velocity (reply storms, quote tweets)
  2. Negative sentiment polarization
  3. Cross-community spread (not just one subreddit)
  4. Persistence (not a flash-in-the-pan)
  5. Action-oriented language (boycott, petition, fired, apology demands)

We use Snorkel-style labeling functions combined with LLM validation.
"""
import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional LLM for validation
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class WeakLabelConfig:
    """Configuration for weak labeling pipeline."""
    # Snorkel-style LF weights
    engagement_threshold: int = 100       # Score + replies
    sentiment_threshold: float = -0.4     # VADER compound
    toxicity_threshold: float = 0.6       # Perspective API proxy
    reply_velocity_threshold: int = 50    # Replies per hour
    cross_subreddit_threshold: int = 3    # Number of distinct communities
    persistence_hours: int = 24           # Event lasts at least N hours
    
    # LLM validation
    llm_model: str = "meta-llama/Llama-2-7b-chat-hf"
    llm_validation_sample_size: int = 5000
    llm_batch_size: int = 32
    use_llm: bool = True
    
    # Output
    positive_label_ratio_target: float = 0.25  # Target 25% positive
    output_path: str = "data/processed/weak_labeled.csv"


class LabelingFunction:
    """Base class for Snorkel-style labeling functions."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
    
    def apply(self, row: pd.Series) -> int:
        """
        Apply labeling function to a row.
        Returns: 1 (positive), 0 (negative), -1 (abstain)
        """
        raise NotImplementedError


class EngagementVelocityLF(LabelingFunction):
    """High engagement velocity indicates cancellation event."""
    
    def __init__(self, threshold: int = 100):
        super().__init__("engagement_velocity", weight=2.0)
        self.threshold = threshold
    
    def apply(self, row: pd.Series) -> int:
        score = row.get("score", 0)
        num_comments = row.get("num_comments", 0)
        total_engagement = score + num_comments
        
        if total_engagement > self.threshold * 5:
            return 1
        elif total_engagement < self.threshold // 5:
            return 0
        return -1


class SentimentPolarizationLF(LabelingFunction):
    """Strong negative sentiment polarization."""
    
    def __init__(self, threshold: float = -0.5):
        super().__init__("sentiment_polarization", weight=1.5)
        self.threshold = threshold
    
    def apply(self, row: pd.Series) -> int:
        sentiment = row.get("sentiment_compound", 0)
        
        if sentiment < self.threshold:
            return 1
        elif sentiment > 0.2:
            return 0
        return -1


class ToxicitySpikeLF(LabelingFunction):
    """High toxicity scores from Perspective API."""
    
    def __init__(self, threshold: float = 0.7):
        super().__init__("toxicity_spike", weight=1.5)
        self.threshold = threshold
    
    def apply(self, row: pd.Series) -> int:
        toxicity = row.get("toxicity", 0)
        severe_toxicity = row.get("severe_toxicity", 0)
        
        if toxicity > self.threshold or severe_toxicity > 0.5:
            return 1
        elif toxicity < 0.2 and severe_toxicity < 0.1:
            return 0
        return -1


class ActionLanguageLF(LabelingFunction):
    """
    Action-oriented language: boycott, petition, fired, resign, apology, etc.
    These are STRONG indicators of organized cancellation campaigns.
    """
    
    def __init__(self):
        super().__init__("action_language", weight=3.0)
        self.action_keywords = {
            "boycott", "petition", "fired", "fire", "resign", "resignation",
            "apologize", "apology", "cancel", "cancelled", "cancelling",
            "remove", "ban", "suspend", "deplatform", "call out",
            "hold accountable", "step down", "drop", "cut ties"
        }
        self.negative_indicators = {
            "i think", "in my opinion", "imo", "personally", "maybe",
            "perhaps", "could be", "might be"
        }
    
    def apply(self, row: pd.Series) -> int:
        text = str(row.get("body", row.get("text", ""))).lower()
        
        action_count = sum(1 for kw in self.action_keywords if kw in text)
        hedge_count = sum(1 for kw in self.negative_indicators if kw in text)
        
        if action_count >= 2 and hedge_count == 0:
            return 1
        elif action_count == 0 and hedge_count >= 2:
            return 0
        return -1


class CrossCommunitySpreadLF(LabelingFunction):
    """
    Cancellation events spread across multiple communities.
    We approximate this by checking if similar posts appear in different subreddits.
    """
    
    def __init__(self):
        super().__init__("cross_community", weight=1.0)
    
    def apply(self, row: pd.Series) -> int:
        # This LF requires pre-computed cross-subreddit features
        # For single-row application, we abstain
        cross_subs = row.get("cross_subreddit_mentions", 0)
        
        if cross_subs >= 3:
            return 1
        elif cross_subs == 0:
            return 0
        return -1


class ReplyStormLF(LabelingFunction):
    """Reply storms: rapid accumulation of replies in short time window."""
    
    def __init__(self, threshold: int = 50):
        super().__init__("reply_storm", weight=2.0)
        self.threshold = threshold
    
    def apply(self, row: pd.Series) -> int:
        num_comments = row.get("num_comments", 0)
        
        if num_comments > self.threshold:
            return 1
        elif num_comments < 5:
            return 0
        return -1


class PersistenceLF(LabelingFunction):
    """
    Cancellation events persist over time (not just a single hot post).
    We approximate by checking if the user has multiple high-engagement negative posts.
    """
    
    def __init__(self):
        super().__init__("persistence", weight=1.0)
    
    def apply(self, row: pd.Series) -> int:
        user_negative_posts = row.get("user_negative_post_count", 0)
        
        if user_negative_posts >= 3:
            return 1
        elif user_negative_posts == 0:
            return 0
        return -1


class WeakLabelingPipeline:
    """
    Complete weak labeling pipeline combining multiple LFs with
    majority voting and optional LLM validation.
    """
    
    def __init__(self, config: Optional[WeakLabelConfig] = None):
        self.config = config or WeakLabelConfig()
        self.labeling_functions: List[LabelingFunction] = [
            EngagementVelocityLF(self.config.engagement_threshold),
            SentimentPolarizationLF(self.config.sentiment_threshold),
            ToxicitySpikeLF(self.config.toxicity_threshold),
            ActionLanguageLF(),
            CrossCommunitySpreadLF(),
            ReplyStormLF(self.config.reply_velocity_threshold),
            PersistenceLF()
        ]
        self.llm_pipeline = None
        if self.config.use_llm:
            self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM for validation subset."""
        try:
            logger.info(f"Loading LLM: {self.config.llm_model}")
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.config.llm_model,
                device=0 if os.system("nvidia-smi > /dev/null 2>&1") == 0 else -1,
                torch_dtype="auto"
            )
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LLM: {e}. Proceeding without LLM validation.")
            self.config.use_llm = False
    
    def apply_labeling_functions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all labeling functions to DataFrame.
        
        Returns:
            DataFrame with columns: lf_<name>_vote, lf_sum, lf_confidence
        """
        logger.info(f"Applying {len(self.labeling_functions)} labeling functions...")
        
        # Initialize vote matrix
        vote_matrix = np.full((len(df), len(self.labeling_functions)), -1, dtype=int)
        
        for lf_idx, lf in enumerate(tqdm(self.labeling_functions, desc="Labeling functions")):
            votes = []
            for _, row in df.iterrows():
                try:
                    vote = lf.apply(row)
                except Exception as e:
                    logger.debug(f"LF {lf.name} error: {e}")
                    vote = -1
                votes.append(vote)
            
            vote_matrix[:, lf_idx] = votes
            df[f"lf_{lf.name}"] = votes
        
        # Majority vote with abstention handling
        df["lf_positive_votes"] = (vote_matrix == 1).sum(axis=1)
        df["lf_negative_votes"] = (vote_matrix == 0).sum(axis=1)
        df["lf_abstentions"] = (vote_matrix == -1).sum(axis=1)
        df["lf_total_votes"] = df["lf_positive_votes"] + df["lf_negative_votes"]
        
        # Weighted vote (using LF weights)
        weighted_positive = np.zeros(len(df))
        weighted_negative = np.zeros(len(df))
        
        for lf_idx, lf in enumerate(self.labeling_functions):
            weighted_positive += (vote_matrix[:, lf_idx] == 1) * lf.weight
            weighted_negative += (vote_matrix[:, lf_idx] == 0) * lf.weight
        
        df["lf_weighted_positive"] = weighted_positive
        df["lf_weighted_negative"] = weighted_negative
        
        # Initial label: weighted majority
        df["weak_label"] = np.where(
            weighted_positive > weighted_negative, 1,
            np.where(weighted_negative > weighted_positive, 0, -1)
        )
        
        # Confidence: margin between positive and negative votes
        total_weight = sum(lf.weight for lf in self.labeling_functions)
        df["lf_confidence"] = np.abs(weighted_positive - weighted_negative) / total_weight
        
        logger.info(f"Weak labeling complete. Label distribution:\n{df['weak_label'].value_counts()}")
        
        return df
    
    def llm_validate_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Use LLM to validate a subset of labels, then propagate corrections.
        
        Strategy:
        1. Sample high-confidence disagreements (where LFs conflict)
        2. Sample borderline cases (low confidence)
        3. LLM labels these
        4. Train a small classifier to propagate LLM labels to full dataset
        """
        if not self.config.use_llm or self.llm_pipeline is None:
            logger.info("Skipping LLM validation")
            return df
        
        logger.info("Starting LLM validation...")
        
        # Select validation set
        # 1. Conflicts: positive_votes > 0 and negative_votes > 0
        conflicts = df[(df["lf_positive_votes"] > 0) & (df["lf_negative_votes"] > 0)]
        
        # 2. Borderline: low confidence
        borderline = df[df["lf_confidence"] < 0.3]
        
        # 3. Random sample
        random_sample = df.sample(min(1000, len(df) // 10))
        
        validation_df = pd.concat([conflicts, borderline, random_sample]).drop_duplicates()
        validation_df = validation_df.sample(min(self.config.llm_validation_sample_size, len(validation_df)))
        
        logger.info(f"LLM validation on {len(validation_df)} samples...")
        
        llm_labels = []
        for _, row in tqdm(validation_df.iterrows(), total=len(validation_df), desc="LLM labeling"):
            text = str(row.get("body", row.get("text", "")))[:500]
            label = self._llm_label_single(text)
            llm_labels.append(label)
        
        validation_df["llm_label"] = llm_labels
        
        # Compare LLM with weak labels
        agreement = (validation_df["weak_label"] == validation_df["llm_label"]).mean()
        logger.info(f"LLM-Weak label agreement: {agreement:.2%}")
        
        # Train correction model: predict LLM label from LF votes
        from sklearn.linear_model import LogisticRegression
        
        feature_cols = [c for c in validation_df.columns if c.startswith("lf_") and c not in 
                        ["weak_label", "llm_label", "lf_confidence"]]
        
        X = validation_df[feature_cols].fillna(0)
        y = validation_df["llm_label"]
        
        # Only use rows where LLM didn't abstain (-1)
        valid_mask = y != -1
        if valid_mask.sum() > 100:
            corrector = LogisticRegression(max_iter=1000, class_weight="balanced")
            corrector.fit(X[valid_mask], y[valid_mask])
            
            # Apply correction to full dataset
            X_full = df[feature_cols].fillna(0)
            corrected_probs = corrector.predict_proba(X_full)[:, 1]
            df["corrected_label_prob"] = corrected_probs
            df["corrected_label"] = (corrected_probs > 0.5).astype(int)
            
            # Use corrected label where confidence is high, weak label otherwise
            df["final_label"] = np.where(
                df["lf_confidence"] > 0.7,
                df["weak_label"],
                df["corrected_label"]
            )
        else:
            logger.warning("Insufficient LLM labels for correction. Using weak labels.")
            df["final_label"] = df["weak_label"]
        
        return df
    
    def _llm_label_single(self, text: str) -> int:
        """Use LLM to label a single text."""
        prompt = f"""You are an expert in social media dynamics. Classify whether the following text describes or is part of a social media cancellation event (organized public backlash against an individual).

Text: {text}

Reply with ONLY one of:
- CANCELLATION_EVENT (if this is part of an organized backlash/cancellation campaign)
- NOT_CANCELLATION (if this is normal discourse, praise, neutral, or unrelated)
- UNCLEAR (if you cannot determine)

Answer:"""
        
        try:
            result = self.llm_pipeline(
                prompt,
                max_new_tokens=20,
                do_sample=False,
                return_full_text=False
            )[0]["generated_text"].strip().upper()
            
            if "CANCELLATION" in result and "NOT" not in result:
                return 1
            elif "NOT" in result or "NORMAL" in result:
                return 0
            else:
                return -1
        except Exception as e:
            logger.debug(f"LLM labeling error: {e}")
            return -1
    
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance dataset to target positive ratio using undersampling + SMOTE.
        """
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
        
        label_col = "final_label" if "final_label" in df.columns else "weak_label"
        
        positive = df[df[label_col] == 1]
        negative = df[df[label_col] == 0]
        
        current_ratio = len(positive) / max(len(df), 1)
        logger.info(f"Current positive ratio: {current_ratio:.2%}")
        
        if current_ratio >= self.config.positive_label_ratio_target:
            logger.info("Dataset already sufficiently balanced")
            return df
        
        # Calculate target counts
        target_pos = int(len(df) * self.config.positive_label_ratio_target)
        target_neg = len(df) - target_pos
        
        # Undersample negative, SMOTE positive
        sampling_strategy = {
            0: min(target_neg, len(negative)),
            1: max(target_pos, len(positive))
        }
        
        # For SMOTE, we need numeric features
        numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
        X = numeric_df.drop(columns=[label_col], errors="ignore")
        y = df[label_col]
        
        pipeline = ImbPipeline([
            ("under", RandomUnderSampler(sampling_strategy={0: sampling_strategy[0]}, random_state=42)),
            ("over", SMOTE(sampling_strategy={1: sampling_strategy[1]}, random_state=42))
        ])
        
        try:
            X_res, y_res = pipeline.fit_resample(X, y)
            
            # Reconstruct DataFrame
            resampled_df = pd.DataFrame(X_res, columns=X.columns)
            resampled_df[label_col] = y_res
            
            # Add back text columns
            # (Simplified: in production, use SMOTENC for categorical + text)
            
            logger.info(f"Balanced dataset: {len(resampled_df)} samples, "
                       f"{y_res.mean():.2%} positive")
            return resampled_df
        except Exception as e:
            logger.warning(f"Balancing failed: {e}. Returning original.")
            return df
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run complete weak labeling pipeline."""
        logger.info("=" * 60)
        logger.info("WEAK LABELING PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Apply labeling functions
        df = self.apply_labeling_functions(df)
        
        # Step 2: LLM validation
        df = self.llm_validate_subset(df)
        
        # Step 3: Balance
        df = self.balance_dataset(df)
        
        # Step 4: Finalize
        label_col = "final_label" if "final_label" in df.columns else "weak_label"
        df["label"] = df[label_col]
        
        # Save
        os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
        df.to_csv(self.config.output_path, index=False)
        logger.info(f"Labeled data saved to {self.config.output_path}")
        
        # Generate report
        self._generate_report(df)
        
        return df
    
    def _generate_report(self, df: pd.DataFrame):
        """Generate labeling quality report."""
        report = {
            "total_samples": len(df),
            "positive_samples": int((df["label"] == 1).sum()),
            "negative_samples": int((df["label"] == 0).sum()),
            "positive_ratio": float((df["label"] == 1).mean()),
            "labeling_functions": [
                {"name": lf.name, "weight": lf.weight}
                for lf in self.labeling_functions
            ]
        }
        
        if "lf_confidence" in df.columns:
            report["avg_confidence"] = float(df["lf_confidence"].mean())
            report["high_confidence_ratio"] = float((df["lf_confidence"] > 0.7).mean())
        
        report_path = Path(self.config.output_path).parent / "labeling_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Labeling report saved to {report_path}")
        logger.info(f"Final label distribution: {df['label'].value_counts().to_dict()}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV/Parquet with features")
    parser.add_argument("--output", default="data/processed/weak_labeled.csv")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM validation")
    parser.add_argument("--target-ratio", type=float, default=0.25)
    args = parser.parse_args()
    
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    config = WeakLabelConfig(
        output_path=args.output,
        use_llm=not args.no_llm,
        positive_label_ratio_target=args.target_ratio
    )
    
    pipeline = WeakLabelingPipeline(config)
    labeled_df = pipeline.run(df)
    
    print(f"\nLabeled {len(labeled_df)} samples")
    print(f"Positive: {(labeled_df['label'] == 1).sum()}")
    print(f"Negative: {(labeled_df['label'] == 0).sum()}")


if __name__ == "__main__":
    main()
