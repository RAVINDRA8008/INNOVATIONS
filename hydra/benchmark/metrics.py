"""
CERA Metrics — Compute-Efficient Reasoning Assessment Metrics

Novel metrics that evaluate BOTH accuracy AND computational efficiency:

1. CERA Score = Accuracy × Efficiency
   - Rewards models that solve problems with proportional compute

2. Pathway Alignment Score (PAS)
   - Measures how well routing decisions match target pathways per difficulty
   
3. Compute Efficiency Ratio (CER)
   - actual_compute / target_compute — ideal is 1.0

4. Routing Entropy
   - Measures diversity of routing decisions across the benchmark

5. Difficulty Calibration Score
   - How well the model scales compute with problem difficulty
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class TaskResult:
    """Result of evaluating a single CERA task."""
    task_id: str
    category: str
    difficulty: int
    correct: bool
    predicted: str
    target: str
    routing_decisions: Optional[List[int]] = None  # Per-token pathway indices
    compute_cost: float = 0.0                       # Actual compute cost
    target_compute: float = 0.0                     # Expected compute cost
    confidence: float = 0.0                         # Model confidence
    latency_ms: float = 0.0                         # Wall-clock inference time


class ComputeEfficiencyMetrics:
    """
    Computes all CERA metrics from a list of task results.
    """
    
    PATHWAY_NAMES = ["Stream", "Focus", "Reason"]
    PATHWAY_COSTS = [1.0, 3.0, 10.0]
    
    def __init__(self):
        self.results: List[TaskResult] = []
    
    def add_result(self, result: TaskResult):
        """Add a single task result."""
        self.results.append(result)
    
    def add_results(self, results: List[TaskResult]):
        """Add multiple task results."""
        self.results.extend(results)
    
    def compute_all(self) -> Dict:
        """Compute all CERA metrics."""
        if not self.results:
            return {"error": "No results to evaluate"}
        
        metrics = {}
        
        # 1. Core CERA Score
        metrics["cera_score"] = self._compute_cera_score()
        
        # 2. Accuracy metrics
        metrics["accuracy"] = self._compute_accuracy()
        
        # 3. Pathway Alignment Score
        metrics["pathway_alignment"] = self._compute_pathway_alignment()
        
        # 4. Compute Efficiency Ratio
        metrics["compute_efficiency"] = self._compute_efficiency_ratio()
        
        # 5. Routing Entropy
        metrics["routing_entropy"] = self._compute_routing_entropy()
        
        # 6. Difficulty Calibration
        metrics["difficulty_calibration"] = self._compute_difficulty_calibration()
        
        # 7. Per-category breakdown
        metrics["per_category"] = self._compute_per_category()
        
        # 8. Per-difficulty breakdown
        metrics["per_difficulty"] = self._compute_per_difficulty()
        
        return metrics
    
    def _compute_cera_score(self) -> Dict:
        """
        CERA Score = Accuracy × Efficiency
        
        The flagship metric. A model that's 95% accurate but wastes 5x compute
        scores lower than a model that's 90% accurate with optimal compute.
        """
        accuracy = sum(1 for r in self.results if r.correct) / len(self.results)
        
        # Efficiency: ratio of target to actual compute (capped at 1.0)
        efficiencies = []
        for r in self.results:
            if r.compute_cost > 0:
                eff = min(1.0, r.target_compute / max(r.compute_cost, 0.01))
            else:
                eff = 1.0  # No compute info available
            efficiencies.append(eff)
        
        avg_efficiency = np.mean(efficiencies) if efficiencies else 1.0
        
        cera = accuracy * avg_efficiency
        
        return {
            "score": cera,
            "accuracy_component": accuracy,
            "efficiency_component": avg_efficiency,
            "interpretation": self._interpret_cera(cera),
        }
    
    def _interpret_cera(self, score: float) -> str:
        """Human-readable interpretation of CERA score."""
        if score >= 0.9:
            return "Exceptional — near-optimal accuracy and efficiency"
        elif score >= 0.7:
            return "Strong — good balance of accuracy and efficiency"
        elif score >= 0.5:
            return "Moderate — room for improvement in accuracy or efficiency"
        elif score >= 0.3:
            return "Weak — significant inefficiency or accuracy gaps"
        else:
            return "Poor — major issues with both accuracy and efficiency"
    
    def _compute_accuracy(self) -> Dict:
        """Standard accuracy metrics."""
        correct = sum(1 for r in self.results if r.correct)
        total = len(self.results)
        
        return {
            "overall": correct / total,
            "correct": correct,
            "total": total,
        }
    
    def _compute_pathway_alignment(self) -> Dict:
        """
        Pathway Alignment Score (PAS).
        
        Measures how well routing decisions match the expected pathway
        for each difficulty level.
        
        Perfect alignment: all Level 1 tasks → Stream, Level 5 → Reason
        """
        target_pathway_map = {
            1: 0,  # Stream
            2: 0,  # Stream
            3: 1,  # Focus
            4: 1,  # Focus
            5: 2,  # Reason
        }
        
        aligned = 0
        total_with_routing = 0
        
        per_difficulty_alignment = {i: {"aligned": 0, "total": 0} for i in range(1, 6)}
        
        for r in self.results:
            if r.routing_decisions is not None and len(r.routing_decisions) > 0:
                total_with_routing += 1
                
                # Dominant pathway: most frequently selected
                pathway_counts = [0, 0, 0]
                for decision in r.routing_decisions:
                    if 0 <= decision < 3:
                        pathway_counts[decision] += 1
                
                dominant = pathway_counts.index(max(pathway_counts))
                target = target_pathway_map.get(r.difficulty, 1)
                
                if dominant == target:
                    aligned += 1
                    per_difficulty_alignment[r.difficulty]["aligned"] += 1
                
                per_difficulty_alignment[r.difficulty]["total"] += 1
        
        overall_pas = aligned / max(total_with_routing, 1)
        
        per_diff_scores = {}
        for diff, counts in per_difficulty_alignment.items():
            if counts["total"] > 0:
                per_diff_scores[f"level_{diff}"] = (
                    counts["aligned"] / counts["total"]
                )
            else:
                per_diff_scores[f"level_{diff}"] = None
        
        return {
            "overall_pas": overall_pas,
            "per_difficulty": per_diff_scores,
            "total_evaluated": total_with_routing,
        }
    
    def _compute_efficiency_ratio(self) -> Dict:
        """
        Compute Efficiency Ratio (CER).
        
        CER = target_compute / actual_compute
        
        CER > 1.0: Model uses LESS compute than expected (over-efficient)
        CER = 1.0: Perfect compute allocation
        CER < 1.0: Model uses MORE compute than expected (wasteful)
        """
        ratios = []
        per_difficulty = {i: [] for i in range(1, 6)}
        
        for r in self.results:
            if r.compute_cost > 0:
                ratio = r.target_compute / max(r.compute_cost, 0.01)
                ratios.append(ratio)
                per_difficulty[r.difficulty].append(ratio)
        
        overall_cer = np.mean(ratios) if ratios else 1.0
        
        per_diff_cer = {}
        for diff, rats in per_difficulty.items():
            per_diff_cer[f"level_{diff}"] = np.mean(rats) if rats else None
        
        return {
            "overall_cer": overall_cer,
            "cer_std": np.std(ratios) if ratios else 0,
            "per_difficulty": per_diff_cer,
            "over_efficient_frac": sum(1 for r in ratios if r > 1.0) / max(len(ratios), 1),
            "wasteful_frac": sum(1 for r in ratios if r < 0.5) / max(len(ratios), 1),
        }
    
    def _compute_routing_entropy(self) -> Dict:
        """
        Routing Entropy — measures routing diversity.
        
        Low entropy: Model routes most tokens to the same pathway (bad)
        High entropy: Diverse routing across pathways (good, if aligned)
        """
        all_decisions = []
        for r in self.results:
            if r.routing_decisions is not None:
                all_decisions.extend(r.routing_decisions)
        
        if not all_decisions:
            return {"entropy": None, "distribution": None}
        
        # Count pathway usage
        total = len(all_decisions)
        counts = [0, 0, 0]
        for d in all_decisions:
            if 0 <= d < 3:
                counts[d] += 1
        
        distribution = [c / total for c in counts]
        
        # Shannon entropy
        entropy = 0
        for p in distribution:
            if p > 0:
                entropy -= p * math.log2(p)
        
        max_entropy = math.log2(3)
        normalized_entropy = entropy / max_entropy
        
        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "max_entropy": max_entropy,
            "distribution": {
                name: frac
                for name, frac in zip(self.PATHWAY_NAMES, distribution)
            },
        }
    
    def _compute_difficulty_calibration(self) -> Dict:
        """
        Difficulty Calibration Score.
        
        Measures if compute increases monotonically with difficulty.
        Perfect calibration: Level 1 uses least compute, Level 5 uses most.
        
        Computed as Spearman correlation between difficulty and compute.
        """
        per_difficulty_compute = {i: [] for i in range(1, 6)}
        
        for r in self.results:
            if r.compute_cost > 0:
                per_difficulty_compute[r.difficulty].append(r.compute_cost)
        
        avg_compute = {}
        for diff in range(1, 6):
            costs = per_difficulty_compute[diff]
            if costs:
                avg_compute[diff] = np.mean(costs)
        
        if len(avg_compute) < 3:
            return {
                "calibration_score": None,
                "message": "Insufficient data for calibration",
            }
        
        # Check if compute increases with difficulty
        difficulties = sorted(avg_compute.keys())
        computes = [avg_compute[d] for d in difficulties]
        
        # Spearman-like: fraction of correctly ordered pairs
        correct_pairs = 0
        total_pairs = 0
        for i in range(len(difficulties)):
            for j in range(i + 1, len(difficulties)):
                total_pairs += 1
                if computes[j] > computes[i]:
                    correct_pairs += 1
        
        calibration = correct_pairs / max(total_pairs, 1)
        
        return {
            "calibration_score": calibration,
            "avg_compute_by_difficulty": avg_compute,
            "is_monotonic": all(
                computes[i] <= computes[i + 1]
                for i in range(len(computes) - 1)
            ),
        }
    
    def _compute_per_category(self) -> Dict:
        """Per-category metrics."""
        categories = set(r.category for r in self.results)
        per_cat = {}
        
        for cat in categories:
            cat_results = [r for r in self.results if r.category == cat]
            accuracy = sum(1 for r in cat_results if r.correct) / len(cat_results)
            
            efficiencies = []
            for r in cat_results:
                if r.compute_cost > 0:
                    eff = min(1.0, r.target_compute / max(r.compute_cost, 0.01))
                    efficiencies.append(eff)
            
            avg_eff = np.mean(efficiencies) if efficiencies else 1.0
            
            per_cat[cat] = {
                "accuracy": accuracy,
                "efficiency": avg_eff,
                "cera_score": accuracy * avg_eff,
                "n_tasks": len(cat_results),
            }
        
        return per_cat
    
    def _compute_per_difficulty(self) -> Dict:
        """Per-difficulty metrics."""
        per_diff = {}
        
        for diff in range(1, 6):
            diff_results = [r for r in self.results if r.difficulty == diff]
            if not diff_results:
                continue
            
            accuracy = sum(1 for r in diff_results if r.correct) / len(diff_results)
            
            efficiencies = []
            for r in diff_results:
                if r.compute_cost > 0:
                    eff = min(1.0, r.target_compute / max(r.compute_cost, 0.01))
                    efficiencies.append(eff)
            
            avg_eff = np.mean(efficiencies) if efficiencies else 1.0
            
            per_diff[f"level_{diff}"] = {
                "accuracy": accuracy,
                "efficiency": avg_eff,
                "cera_score": accuracy * avg_eff,
                "n_tasks": len(diff_results),
            }
        
        return per_diff
    
    def generate_report(self) -> str:
        """Generate a human-readable report of all metrics."""
        metrics = self.compute_all()
        
        lines = [
            "=" * 70,
            "CERA — Compute-Efficient Reasoning Assessment Report",
            "=" * 70,
            "",
            f"Overall CERA Score: {metrics['cera_score']['score']:.4f}",
            f"  Accuracy:   {metrics['cera_score']['accuracy_component']:.4f}",
            f"  Efficiency: {metrics['cera_score']['efficiency_component']:.4f}",
            f"  → {metrics['cera_score']['interpretation']}",
            "",
            "─" * 40,
            "Per-Category Breakdown:",
            "─" * 40,
        ]
        
        for cat, data in metrics.get("per_category", {}).items():
            lines.append(
                f"  {cat:15s} | Acc: {data['accuracy']:.3f} | "
                f"Eff: {data['efficiency']:.3f} | "
                f"CERA: {data['cera_score']:.3f} | "
                f"n={data['n_tasks']}"
            )
        
        lines.extend([
            "",
            "─" * 40,
            "Per-Difficulty Breakdown:",
            "─" * 40,
        ])
        
        for level, data in metrics.get("per_difficulty", {}).items():
            lines.append(
                f"  {level:10s} | Acc: {data['accuracy']:.3f} | "
                f"Eff: {data['efficiency']:.3f} | "
                f"CERA: {data['cera_score']:.3f}"
            )
        
        lines.extend([
            "",
            "─" * 40,
            "Compute Efficiency:",
            "─" * 40,
            f"  CER: {metrics['compute_efficiency']['overall_cer']:.3f} "
            f"(ideal: 1.0)",
            f"  Over-efficient: {metrics['compute_efficiency']['over_efficient_frac']:.1%}",
            f"  Wasteful: {metrics['compute_efficiency']['wasteful_frac']:.1%}",
        ])
        
        routing = metrics.get("routing_entropy", {})
        if routing.get("distribution"):
            lines.extend([
                "",
                "─" * 40,
                "Routing Distribution:",
                "─" * 40,
            ])
            for name, frac in routing["distribution"].items():
                bar = "█" * int(frac * 40)
                lines.append(f"  {name:8s} | {bar} {frac:.1%}")
            lines.append(
                f"  Entropy: {routing['normalized_entropy']:.3f} (normalized)"
            )
        
        calibration = metrics.get("difficulty_calibration", {})
        if calibration.get("calibration_score") is not None:
            lines.extend([
                "",
                "─" * 40,
                "Difficulty Calibration:",
                "─" * 40,
                f"  Score: {calibration['calibration_score']:.3f}",
                f"  Monotonic: {calibration['is_monotonic']}",
            ])
        
        lines.extend(["", "=" * 70])
        
        return "\n".join(lines)
    
    def reset(self):
        """Clear all results."""
        self.results = []
