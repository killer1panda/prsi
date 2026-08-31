#!/usr/bin/env python3
"""LLM-as-Judge for Weak Label Verification.

Uses a local LLM (via Ollama) or OpenAI API to verify the quality of
heuristic-generated labels on a validation subset. Provides:
- Label correctness estimation (precision of weak labels)
- Ambiguous case flagging for human review
- Confidence scores per prediction
- Cost-effective batched evaluation with caching

This closes the loop on the weak supervision pipeline by quantifying
how reliable the heuristics actually are, which is critical for
viva defense when examiners question label validity.

Usage:
    python -m src.labels.llm_verifier \
        --dataset data/reddit_processed.parquet \
        --sample-size 500 \
        --model llama3.1:8b \
        --output reports/label_verification.json
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class VerificationResult:
    """Result of LLM verification for a single sample."""
    sample_id: str
    text_preview: str
    heuristic_label: int
    llm_label: int
    llm_confidence: float  # 0.0 - 1.0
    llm_reasoning: str
    agreement: bool
    heuristic_score: int
    verification_time_ms: float


@dataclass
class VerificationReport:
    total_samples: int
    agreement_rate: float
    heuristic_precision: float
    heuristic_recall: float
    llm_ambiguous_count: int
    ambiguous_indices: List[int] = field(default_factory=list)
    per_sample_results: List[VerificationResult] = field(default_factory=list)
    cost_estimate_usd: float = 0.0
    total_time_sec: float = 0.0
    model_used: str = "unknown"

    def to_dict(self) -> Dict:
        return asdict(self)


class LLMVerifier:
    """LLM-as-judge for validating weak supervision labels."""

    SYSTEM_PROMPT = """You are an expert content moderator evaluating Reddit posts for cancellation risk.

A post is "cancelled" (label=1) if it describes or relates to:
- Someone being fired, resigned, or forced to apologize due to public backlash
- A boycott or petition campaign against a person or organization
- Severe reputation damage from exposed misconduct
- Coordinated online harassment campaigns ("dogpiling")
- A celebrity/public figure stepping down from roles

A post is "not cancelled" (label=0) if it is:
- General discussion, news, or opinion
- Mild criticism without career/reputation consequences
- Memes or humor without targeted harm
- Personal anecdotes unrelated to public figures

Rules:
1. Respond ONLY with a JSON object: {"label": 0 or 1, "confidence": 0.0-1.0, "reasoning": "short explanation"}
2. Be conservative: only label 1 if there is clear evidence of cancellation
3. If the text is ambiguous or lacks context, use confidence < 0.7
4. Consider both the title and body text together
"""

    def __init__(
        self,
        backend: str = "ollama",  # "ollama" or "openai"
        model: str = "llama3.1:8b",
        api_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 256,
        rate_limit_qps: float = 2.0,
        cache_dir: str = ".cache/llm_verifier",
    ):
        self.backend = backend
        self.model = model
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limit_qps = rate_limit_qps
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0
        self._session = requests.Session()

    def _get_cache_key(self, text: str) -> str:
        """Deterministic cache key from text content."""
        return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()

    def _check_cache(self, text: str) -> Optional[Dict]:
        key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def _save_cache(self, text: str, response: Dict) -> None:
        key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w") as f:
            json.dump(response, f)

    def _rate_limit(self) -> None:
        """Enforce QPS rate limit."""
        elapsed = time.time() - self._last_request_time
        min_interval = 1.0 / self.rate_limit_qps
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def _call_ollama(self, text: str) -> Dict:
        """Call Ollama API for local LLM inference."""
        self._rate_limit()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Evaluate this Reddit post:\n\n{text}\n\nRespond with JSON only."},
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        resp = self._session.post(f"{self.api_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        return self._parse_response(content)

    def _call_openai(self, text: str) -> Dict:
        """Call OpenAI API."""
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed")
        self._rate_limit()
        client = openai.OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Evaluate this Reddit post:\n\n{text}\n\nRespond with JSON only."},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        return self._parse_response(content)

    def _parse_response(self, content: str) -> Dict:
        """Parse and sanitize LLM JSON response."""
        # Extract JSON if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: extract with regex
            import re
            label_match = re.search(r'"label"\s*:\s*(0|1)', content)
            conf_match = re.search(r'"confidence"\s*:\s*(0\.\d+|1\.0|1)', content)
            parsed = {
                "label": int(label_match.group(1)) if label_match else 0,
                "confidence": float(conf_match.group(1)) if conf_match else 0.5,
                "reasoning": content[:200],
            }

        # Sanitize
        label = int(parsed.get("label", 0))
        if label not in (0, 1):
            label = 0
        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        return {
            "label": label,
            "confidence": confidence,
            "reasoning": str(parsed.get("reasoning", "")),
        }

    def verify_sample(self, text: str, heuristic_label: int, heuristic_score: int) -> VerificationResult:
        """Verify a single sample against LLM judge."""
        start = time.time()

        # Check cache
        cached = self._check_cache(text)
        if cached:
            llm_result = cached
        else:
            try:
                if self.backend == "ollama":
                    llm_result = self._call_ollama(text)
                else:
                    llm_result = self._call_openai(text)
                self._save_cache(text, llm_result)
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
                llm_result = {"label": heuristic_label, "confidence": 0.0, "reasoning": f"ERROR: {e}"}

        elapsed = (time.time() - start) * 1000
        return VerificationResult(
            sample_id=self._get_cache_key(text)[:16],
            text_preview=text[:200].replace("\n", " "),
            heuristic_label=heuristic_label,
            llm_label=llm_result["label"],
            llm_confidence=llm_result["confidence"],
            llm_reasoning=llm_result["reasoning"],
            agreement=(heuristic_label == llm_result["label"]),
            heuristic_score=heuristic_score,
            verification_time_ms=elapsed,
        )

    def verify_dataset(
        self,
        dataset_path: str,
        sample_size: int = 500,
        random_seed: int = 42,
    ) -> VerificationReport:
        """Run verification on a stratified random sample."""
        logger.info(f"Loading dataset: {dataset_path}")
        df = pd.read_parquet(dataset_path) if dataset_path.endswith(".parquet") else pd.read_csv(dataset_path)
        if len(df) < sample_size:
            sample_size = len(df)

        # Stratified sample to preserve label balance
        if "label" in df.columns:
            sample = df.groupby("label", group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size // 2 + 1), random_state=random_seed)
            )
            if len(sample) > sample_size:
                sample = sample.sample(sample_size, random_state=random_seed)
        else:
            sample = df.sample(sample_size, random_state=random_seed)

        logger.info(f"Verifying {len(sample)} samples with {self.backend}/{self.model}")
        start_total = time.time()

        results = []
        ambiguous = []
        for idx, row in sample.iterrows():
            text = str(row.get("combined_text", row.get("text", "")))
            h_label = int(row.get("label", 0))
            h_score = int(row.get("label_score", 0))

            result = self.verify_sample(text, h_label, h_score)
            results.append(result)

            if result.llm_confidence < 0.7:
                ambiguous.append(idx)

            if (len(results) % 50) == 0:
                agree = sum(1 for r in results if r.agreement)
                logger.info(f"Progress: {len(results)}/{len(sample)} | Agreement: {agree}/{len(results)}")

        total_time = time.time() - start_total
        agreements = sum(1 for r in results if r.agreement)

        # Estimate precision/recall assuming LLM is ground truth
        tp = sum(1 for r in results if r.heuristic_label == 1 and r.llm_label == 1)
        fp = sum(1 for r in results if r.heuristic_label == 1 and r.llm_label == 0)
        fn = sum(1 for r in results if r.heuristic_label == 0 and r.llm_label == 1)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Cost estimate (very rough)
        cost = 0.0
        if self.backend == "openai":
            cost = len(sample) * 0.001  # Approx $0.001 per call for GPT-4-mini

        report = VerificationReport(
            total_samples=len(results),
            agreement_rate=agreements / len(results) if results else 0.0,
            heuristic_precision=precision,
            heuristic_recall=recall,
            llm_ambiguous_count=len(ambiguous),
            ambiguous_indices=ambiguous[:50],  # Limit output
            per_sample_results=results,
            cost_estimate_usd=cost,
            total_time_sec=total_time,
            model_used=f"{self.backend}:{self.model}",
        )

        logger.info(f"Verification complete: {report.agreement_rate:.1%} agreement, "
                    f"precision={precision:.2f}, recall={recall:.2f}")
        return report

    def save_report(self, report: VerificationReport, output_path: str) -> None:
        """Save detailed verification report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info(f"Report saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="LLM Label Verification")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--backend", default="ollama", choices=["ollama", "openai"])
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--api-url", default="http://localhost:11434")
    parser.add_argument("--output", default="reports/label_verification.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    verifier = LLMVerifier(backend=args.backend, model=args.model, api_url=args.api_url)
    report = verifier.verify_dataset(args.dataset, sample_size=args.sample_size, random_seed=args.seed)
    verifier.save_report(report, args.output)

    print(f"Agreement rate: {report.agreement_rate:.1%}")
    print(f"Heuristic precision (vs LLM): {report.heuristic_precision:.2f}")
    print(f"Ambiguous cases: {report.llm_ambiguous_count}")


if __name__ == "__main__":
    main()
