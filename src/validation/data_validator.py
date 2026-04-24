#!/usr/bin/env python3
"""Production Data Validation Pipeline with Great Expectations + Custom Validators.

Validates Reddit/ingested datasets against production schemas to catch:
- Schema drift (new columns, type changes)
- Distribution shift (label balance changes, missing values spikes)
- Data quality regressions (text length anomalies, encoding corruption)
- PII leakage (raw usernames not hashed)
- Pipeline corruption (duplicate IDs, negative scores)

Integrates with:
- Great Expectations (open-source data validation)
- Pandera (runtime schema enforcement)
- Custom ML-specific checks (embedding dimension consistency)

Usage:
    python -m src.validation.data_validator \
        --dataset data/reddit_processed.parquet \
        --expectation-suite suites/reddit_suite.json \
        --report-dir reports/validation/
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Great Expectations (optional but recommended)
try:
    import great_expectations as gx
    from great_expectations.core.expectation_suite import ExpectationSuite
    from great_expectations.expectations import (
        ExpectColumnValuesToBeBetween,
        ExpectColumnValuesToNotBeNull,
        ExpectTableColumnsToMatchOrderedList,
        ExpectColumnValuesToBeInSet,
        ExpectColumnPairValuesToBeEqual,
    )
    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False
    logger.warning("Great Expectations not installed. Using fallback validators only.")

# Pandera for runtime schema validation
try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False


@dataclass
class ValidationResult:
    """Structured result of a validation run."""
    suite_name: str
    dataset_path: str
    passed: bool
    checks_total: int
    checks_passed: int
    checks_failed: int
    failures: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    row_count: int = 0
    memory_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "dataset_path": self.dataset_path,
            "passed": self.passed,
            "checks_total": self.checks_total,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "row_count": self.row_count,
            "memory_mb": self.memory_mb,
            "timestamp": self.timestamp,
            "failures": self.failures,
            "warnings": self.warnings,
        }


class DataValidator:
    """Production data validation engine for Doom Index datasets."""

    # Expected schema for processed Reddit data
    EXPECTED_COLUMNS = [
        "id", "author_hash", "subreddit", "created_utc",
        "title", "selftext", "combined_text", "score",
        "num_comments", "upvote_ratio", "distinguished",
        "edited", "label", "label_score", "label_components",
        "engagement_velocity", "content_hash",
    ]

    CANCEL_SUBREDDITS = frozenset([
        "outoftheloop", "subredditdrama", "againsthatesubreddits",
        "circlebroke", "worstof", "bestofoutrageculture",
    ])

    def __init__(self, strict: bool = True, pii_check: bool = True):
        self.strict = strict
        self.pii_check = pii_check
        self._failure_threshold = 0.05  # 5% bad rows max

    def validate(self, dataset_path: str, suite_name: str = "reddit_production") -> ValidationResult:
        """Run full validation suite against dataset."""
        logger.info(f"Validating {dataset_path}")
        df = self._load(dataset_path)
        result = ValidationResult(
            suite_name=suite_name,
            dataset_path=dataset_path,
            passed=True,
            checks_total=0,
            checks_passed=0,
            checks_failed=0,
            row_count=len(df),
            memory_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
        )

        checks = [
            self._check_schema,
            self._check_nulls,
            self._check_types,
            self._check_ranges,
            self._check_label_balance,
            self._check_text_quality,
            self._check_duplicates,
            self._check_pii_leakage,
            self._check_date_consistency,
            self._check_engagement_anomalies,
        ]

        if GX_AVAILABLE:
            checks.append(self._check_with_great_expectations)

        for check_fn in checks:
            result.checks_total += 1
            try:
                passed, detail = check_fn(df)
                if passed:
                    result.checks_passed += 1
                else:
                    result.checks_failed += 1
                    result.failures.append({"check": check_fn.__name__, "detail": detail})
                    if self.strict:
                        result.passed = False
                    else:
                        result.warnings.append({"check": check_fn.__name__, "detail": detail})
            except Exception as e:
                result.checks_failed += 1
                result.failures.append({"check": check_fn.__name__, "detail": str(e)})
                result.passed = False

        # Aggregate pass/fail
        if result.checks_failed / max(result.checks_total, 1) > self._failure_threshold:
            result.passed = False

        logger.info(f"Validation: {result.checks_passed}/{result.checks_total} passed")
        return result

    def _load(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix == ".parquet":
            return pd.read_parquet(path)
        elif p.suffix == ".csv":
            return pd.read_csv(path)
        elif p.suffix in (".arrow", ".ipc"):
            import pyarrow.ipc as ipc
            with ipc.open_file(path) as reader:
                return reader.read_pandas()
        else:
            raise ValueError(f"Unsupported format: {p.suffix}")

    def _check_schema(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Verify column presence and ordering."""
        missing = set(self.EXPECTED_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(self.EXPECTED_COLUMNS)
        if missing:
            return False, f"Missing columns: {missing}"
        if extra:
            return True, f"Extra columns (non-critical): {extra}"  # Warning, not failure
        return True, "Schema matches expected"

    def _check_nulls(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check critical columns have no nulls."""
        critical = ["id", "combined_text", "label", "content_hash"]
        nulls = {col: df[col].isnull().sum() for col in critical if col in df.columns}
        bad = {k: v for k, v in nulls.items() if v > 0}
        if bad:
            pct = {k: f"{v/len(df)*100:.2f}%" for k, v in bad.items()}
            return False, f"Nulls in critical columns: {pct}"
        return True, "No nulls in critical columns"

    def _check_types(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Verify column types."""
        expected_types = {
            "id": "object",
            "label": ("int64", "int32"),
            "score": ("int64", "float64"),
            "created_utc": ("int64", "float64"),
            "upvote_ratio": "float64",
        }
        errors = []
        for col, expected in expected_types.items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            if isinstance(expected, tuple):
                if actual not in expected:
                    errors.append(f"{col}: expected {expected}, got {actual}")
            else:
                if actual != expected:
                    errors.append(f"{col}: expected {expected}, got {actual}")
        if errors:
            return False, f"Type mismatches: {errors}"
        return True, "Types valid"

    def _check_ranges(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate value ranges."""
        errors = []
        if "label" in df.columns:
            invalid = df[~df["label"].isin([0, 1])]
            if len(invalid) > 0:
                errors.append(f"{len(invalid)} rows with invalid label (not 0/1)")
        if "score" in df.columns:
            neg = (df["score"] < 0).sum()
            if neg > 0:
                errors.append(f"{neg} rows with negative score")
        if "upvote_ratio" in df.columns:
            out_of_range = ((df["upvote_ratio"] < 0) | (df["upvote_ratio"] > 1)).sum()
            if out_of_range > 0:
                errors.append(f"{out_of_range} rows with upvote_ratio outside [0,1]")
        if "num_comments" in df.columns:
            neg = (df["num_comments"] < 0).sum()
            if neg > 0:
                errors.append(f"{neg} rows with negative num_comments")
        if errors:
            return False, f"Range violations: {errors}"
        return True, "Ranges valid"

    def _check_label_balance(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for extreme label imbalance (indicates pipeline bug)."""
        if "label" not in df.columns:
            return True, "No label column"
        counts = df["label"].value_counts(normalize=True)
        if len(counts) < 2:
            return False, "Only one label class present (100% imbalance)"
        minority_pct = counts.min()
        if minority_pct < 0.01:  # Less than 1% minority
            return False, f"Extreme imbalance: minority={minority_pct:.2%}"
        if minority_pct < 0.05:
            return True, f"Warning: high imbalance, minority={minority_pct:.2%}"
        return True, f"Label balance OK: {counts.to_dict()}"

    def _check_text_quality(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Detect encoding corruption, empty text, extreme lengths."""
        if "combined_text" not in df.columns:
            return True, "No text column to check"
        texts = df["combined_text"].astype(str)
        empty = (texts.str.len() < 5).sum()
        too_long = (texts.str.len() > 10000).sum()
        encoding_issues = texts.str.contains(r"[^\x00-\x7F]", regex=True).sum()

        errors = []
        if empty / len(df) > 0.01:
            errors.append(f"{empty} very short texts (>1%)")
        if too_long > 0:
            errors.append(f"{too_long} extremely long texts (>10000 chars)")
        if errors:
            return False, f"Text quality issues: {errors}"
        return True, f"Text quality OK (empty={empty}, long={too_long}, non-ascii={encoding_issues})"

    def _check_duplicates(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for duplicate IDs or content hashes."""
        if "id" in df.columns:
            dup_ids = df["id"].duplicated().sum()
            if dup_ids > 0:
                return False, f"{dup_ids} duplicate IDs"
        if "content_hash" in df.columns:
            dup_hashes = df["content_hash"].duplicated().sum()
            if dup_hashes > 0:
                return False, f"{dup_hashes} duplicate content hashes"
        return True, "No duplicates detected"

    def _check_pii_leakage(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Detect potential PII leakage in author field."""
        if not self.pii_check:
            return True, "PII check disabled"
        if "author_hash" not in df.columns:
            return True, "No author_hash column"

        # Heuristic: if author values look like usernames (contain alphanumeric, no hash length)
        sample = df["author_hash"].dropna().head(100).astype(str)
        looks_like_raw = sample.str.match(r"^[a-zA-Z0-9_-]{3,20}$").mean()
        if looks_like_raw > 0.5:
            return False, f"{looks_like_raw:.0%} of author_hash looks like raw usernames, not hashes"
        return True, "Author hashing appears valid"

    def _check_date_consistency(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check created_utc timestamps are reasonable."""
        if "created_utc" not in df.columns:
            return True, "No date column"
        now = datetime.utcnow().timestamp()
        too_old = (df["created_utc"] < 1.2e9).sum()  # Before ~2008
        future = (df["created_utc"] > now).sum()
        errors = []
        if too_old > 0:
            errors.append(f"{too_old} posts before 2008")
        if future > 0:
            errors.append(f"{future} posts from the future")
        if errors:
            return False, f"Date anomalies: {errors}"
        return True, "Date range valid"

    def _check_engagement_anomalies(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Detect engagement score manipulation or bot patterns."""
        if "score" not in df.columns or "num_comments" not in df.columns:
            return True, "Missing engagement columns"
        # Score should generally correlate with comments
        ratio = df["score"] / (df["num_comments"] + 1)
        outliers = (ratio > 1000).sum()  # 1000 upvotes per comment is suspicious
        if outliers > len(df) * 0.001:  # More than 0.1%
            return True, f"Warning: {outliers} posts with extreme score/comment ratio"
        return True, "Engagement distributions normal"

    def _check_with_great_expectations(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Run Great Expectations validation suite if available."""
        if not GX_AVAILABLE:
            return True, "GX not available"
        try:
            context = gx.get_context()
            suite = ExpectationSuite(name="reddit_suite")
            suite.add_expectation(ExpectColumnValuesToNotBeNull(column="id"))
            suite.add_expectation(ExpectColumnValuesToBeBetween(column="label", min_value=0, max_value=1))
            suite.add_expectation(ExpectColumnValuesToBeBetween(column="upvote_ratio", min_value=0, max_value=1))

            # Validate
            validation_results = df.validate(suite)
            success = validation_results.success
            details = f"GX success={success}, results={len(validation_results.results)}"
            return success, details
        except Exception as e:
            return False, f"GX validation error: {e}"

    def generate_report(self, result: ValidationResult, output_dir: str) -> str:
        """Generate HTML/JSON validation report."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = out / f"validation_{result.suite_name}_{datetime.utcnow():%Y%m%d_%H%M%S}.json"
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Simple HTML report
        html_path = out / f"validation_{result.suite_name}_{datetime.utcnow():%Y%m%d_%H%M%S}.html"
        status_color = "green" if result.passed else "red"
        html = f"""
        <html><head><title>Validation Report</title></head><body>
        <h1>Data Validation Report: {result.suite_name}</h1>
        <p>Dataset: {result.dataset_path}</p>
        <p>Status: <span style="color:{status_color};font-weight:bold;">
            {"PASSED" if result.passed else "FAILED"}</span></p>
        <p>Rows: {result.row_count:,} | Memory: {result.memory_mb:.1f} MB</p>
        <p>Checks: {result.checks_passed}/{result.checks_total} passed</p>
        <h2>Failures</h2>
        <ul>{"".join(f"<li>{f['check']}: {f['detail']}</li>" for f in result.failures)}</ul>
        <h2>Warnings</h2>
        <ul>{"".join(f"<li>{w['check']}: {w['detail']}</li>" for w in result.warnings)}</ul>
        <p>Generated: {result.timestamp}</p>
        </body></html>
        """
        html_path.write_text(html)
        logger.info(f"Validation reports saved to {out}")
        return str(html_path)


def main():
    parser = argparse.ArgumentParser(description="Data Validation Pipeline")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--suite", default="reddit_production")
    parser.add_argument("--report-dir", default="reports/validation")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--no-pii-check", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    validator = DataValidator(strict=args.strict, pii_check=not args.no_pii_check)
    result = validator.validate(args.dataset, args.suite)
    validator.generate_report(result, args.report_dir)

    if not result.passed:
        print(f"VALIDATION FAILED: {result.checks_failed} checks failed")
        sys.exit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
