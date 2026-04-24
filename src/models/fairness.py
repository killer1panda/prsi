"""
Fairness analysis for Doom Index: demographic parity, equalized odds,
and disparate impact across language groups, user types, and engagement levels.
"""
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class FairnessConfig:
    protected_attributes: List[str] = None
    fairness_threshold: float = 0.8  # Disparate impact threshold (80% rule)
    epsilon: float = 1e-10


class FairnessAnalyzer:
    """
    Comprehensive fairness auditing for social media risk prediction.
    Detects bias against protected groups (language, region, user type).
    """

    def __init__(self, config: Optional[FairnessConfig] = None):
        self.config = config or FairnessConfig()
        if self.config.protected_attributes is None:
            self.config.protected_attributes = ["language", "user_type", "region"]
        logger.info("FairnessAnalyzer initialized")

    def demographic_parity(self, y_pred: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
        """
        Measure demographic parity: P(Ŷ=1 | G=g) should be equal across groups.

        Returns:
            Dict of group -> positive rate
        """
        results = {}
        unique_groups = np.unique(groups)

        for g in unique_groups:
            mask = groups == g
            rate = np.mean(y_pred[mask])
            results[str(g)] = float(rate)

        # Max difference
        rates = list(results.values())
        results["max_disparity"] = float(max(rates) - min(rates))
        results["parity_violation"] = results["max_disparity"] > 0.1

        return results

    def equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       groups: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Measure equalized odds: TPR and FPR should be equal across groups.

        Returns:
            Dict of group -> {tpr, fpr, tpr_disparity, fpr_disparity}
        """
        results = {}
        unique_groups = np.unique(groups)

        tpr_list = []
        fpr_list = []

        for g in unique_groups:
            mask = groups == g
            yt = y_true[mask]
            yp = y_pred[mask]

            if len(np.unique(yt)) < 2:
                continue

            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

            tpr = tp / max((tp + fn), self.config.epsilon)
            fpr = fp / max((fp + tn), self.config.epsilon)

            tpr_list.append(tpr)
            fpr_list.append(fpr)

            results[str(g)] = {
                "tpr": float(tpr),
                "fpr": float(fpr),
                "sample_size": int(mask.sum())
            }

        if tpr_list and fpr_list:
            results["max_tpr_disparity"] = float(max(tpr_list) - min(tpr_list))
            results["max_fpr_disparity"] = float(max(fpr_list) - min(fpr_list))
            results["equalized_odds_violation"] = (
                results["max_tpr_disparity"] > 0.1 or results["max_fpr_disparity"] > 0.1
            )

        return results

    def disparate_impact(self, y_pred: np.ndarray, groups: np.ndarray) -> Dict[str, any]:
        """
        Measure disparate impact ratio: min(P(Ŷ=1|G=g)) / max(P(Ŷ=1|G=g)).
        Legal threshold is typically 0.8 (80% rule).

        Returns:
            Dict with impact ratios and violation flags
        """
        unique_groups = np.unique(groups)
        rates = {}

        for g in unique_groups:
            mask = groups == g
            rates[str(g)] = float(np.mean(y_pred[mask]))

        rate_values = list(rates.values())
        if max(rate_values) > 0:
            min_rate = min(rate_values)
            max_rate = max(rate_values)
            impact_ratio = min_rate / max_rate
        else:
            impact_ratio = 1.0

        return {
            "group_rates": rates,
            "impact_ratio": float(impact_ratio),
            "threshold": self.config.fairness_threshold,
            "violation": impact_ratio < self.config.fairness_threshold,
            "severity": "critical" if impact_ratio < 0.6 else "high" if impact_ratio < 0.8 else "low"
        }

    def calibration(self, y_true: np.ndarray, y_prob: np.ndarray, 
                    groups: np.ndarray, n_bins: int = 10) -> Dict[str, any]:
        """
        Measure calibration across groups: predicted prob should match actual rate.

        Returns:
            Dict of group -> calibration curve data and ECE
        """
        results = {}
        unique_groups = np.unique(groups)

        for g in unique_groups:
            mask = groups == g
            yt = y_true[mask]
            yp = y_prob[mask]

            if len(yt) < n_bins * 2:
                continue

            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            bin_accs = []
            bin_confs = []
            bin_counts = []

            for i in range(n_bins):
                bin_mask = (yp >= bin_edges[i]) & (yp < bin_edges[i + 1])
                if i == n_bins - 1:
                    bin_mask = (yp >= bin_edges[i]) & (yp <= bin_edges[i + 1])

                if bin_mask.sum() > 0:
                    bin_acc = np.mean(yt[bin_mask])
                    bin_conf = np.mean(yp[bin_mask])
                    bin_count = bin_mask.sum()

                    ece += (bin_count / len(yt)) * abs(bin_acc - bin_conf)
                    bin_accs.append(float(bin_acc))
                    bin_confs.append(float(bin_conf))
                    bin_counts.append(int(bin_count))

            results[str(g)] = {
                "ece": float(ece),
                "bin_accuracies": bin_accs,
                "bin_confidences": bin_confs,
                "bin_counts": bin_counts,
                "well_calibrated": ece < 0.05
            }

        return results

    def full_audit(self, y_true: np.ndarray, y_pred: np.ndarray, 
                   y_prob: np.ndarray, groups: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Run complete fairness audit across all protected attributes.

        Args:
            y_true: Ground truth labels
            y_pred: Binary predictions
            y_prob: Probability scores
            groups: Dict of attribute_name -> group assignments

        Returns:
            Comprehensive fairness report
        """
        report = {
            "summary": {
                "fair": True,
                "violations": [],
                "overall_score": 1.0
            },
            "details": {}
        }

        total_violations = 0
        total_checks = 0

        for attr_name, group_labels in groups.items():
            attr_report = {}

            # Demographic parity
            dp = self.demographic_parity(y_pred, group_labels)
            attr_report["demographic_parity"] = dp
            if dp.get("parity_violation", False):
                total_violations += 1
            total_checks += 1

            # Equalized odds
            eo = self.equalized_odds(y_true, y_pred, group_labels)
            attr_report["equalized_odds"] = eo
            if eo.get("equalized_odds_violation", False):
                total_violations += 1
            total_checks += 1

            # Disparate impact
            di = self.disparate_impact(y_pred, group_labels)
            attr_report["disparate_impact"] = di
            if di.get("violation", False):
                total_violations += 1
            total_checks += 1

            # Calibration
            cal = self.calibration(y_true, y_prob, group_labels)
            attr_report["calibration"] = cal

            report["details"][attr_name] = attr_report

        # Overall assessment
        if total_checks > 0:
            report["summary"]["overall_score"] = 1.0 - (total_violations / total_checks)
            report["summary"]["fair"] = total_violations == 0
            report["summary"]["violation_count"] = total_violations
            report["summary"]["total_checks"] = total_checks

        if total_violations > 0:
            report["summary"]["violations"].append(
                f"Found {total_violations} fairness violations across {len(groups)} protected attributes"
            )

        return report

    def generate_mitigation_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations based on audit results."""
        recommendations = []

        for attr, details in report["details"].items():
            di = details.get("disparate_impact", {})
            if di.get("violation"):
                recommendations.append(
                    f"[{attr}] Disparate impact detected (ratio={di['impact_ratio']:.3f}). "
                    f"Consider reweighting training samples or adding fairness constraints."
                )

            eo = details.get("equalized_odds", {})
            if eo.get("equalized_odds_violation"):
                recommendations.append(
                    f"[{attr}] Equalized odds violated. Consider adversarial debiasing or threshold tuning."
                )

            dp = details.get("demographic_parity", {})
            if dp.get("parity_violation"):
                recommendations.append(
                    f"[{attr}] Demographic parity violated (disparity={dp['max_disparity']:.3f}). "
                    f"Consider stratified sampling or fairness-regularized training."
                )

        if not recommendations:
            recommendations.append("No fairness violations detected. Model passes fairness audit.")

        return recommendations
