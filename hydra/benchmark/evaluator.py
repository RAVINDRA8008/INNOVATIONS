"""
CERA Evaluator — End-to-End Evaluation Pipeline

Connects the HYDRA model to the CERA benchmark for complete evaluation:
    1. Generates benchmark tasks
    2. Tokenizes and feeds them to the model
    3. Captures routing decisions and compute costs
    4. Computes all CERA metrics
    5. Generates a comprehensive report
"""

import time
from typing import Dict, List, Optional
import logging

import torch

from .cera import CERABenchmark, CERATask
from .metrics import ComputeEfficiencyMetrics, TaskResult

logger = logging.getLogger("hydra.evaluator")


class CERAEvaluator:
    """
    Evaluates a HYDRA model using the CERA benchmark.
    """
    
    PATHWAY_COSTS = [1.0, 3.0, 10.0]
    
    def __init__(
        self,
        model,
        tokenizer=None,
        device: str = "auto",
        max_gen_length: int = 128,
        tasks_per_category_per_level: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_gen_length = max_gen_length
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        
        # Initialize benchmark
        self.benchmark = CERABenchmark(
            tasks_per_category_per_level=tasks_per_category_per_level
        )
        
        # Metrics tracker
        self.metrics = ComputeEfficiencyMetrics()
    
    def evaluate(
        self,
        categories: Optional[List[str]] = None,
        difficulties: Optional[List[int]] = None,
    ) -> Dict:
        """
        Run full CERA evaluation.
        
        Args:
            categories: Optional filter for specific categories
            difficulties: Optional filter for specific difficulty levels
            
        Returns:
            Complete metrics dictionary
        """
        logger.info("Generating CERA benchmark tasks...")
        tasks = self.benchmark.generate()
        
        # Filter if needed
        if categories:
            tasks = [t for t in tasks if t.category in categories]
        if difficulties:
            tasks = [t for t in tasks if t.difficulty in difficulties]
        
        logger.info(f"Evaluating {len(tasks)} tasks...")
        
        self.model.eval()
        self.metrics.reset()
        
        for i, task in enumerate(tasks):
            result = self._evaluate_task(task)
            self.metrics.add_result(result)
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Evaluated {i + 1}/{len(tasks)} tasks")
        
        # Compute all metrics
        all_metrics = self.metrics.compute_all()
        
        # Generate report
        report = self.metrics.generate_report()
        logger.info(f"\n{report}")
        
        return all_metrics
    
    def _evaluate_task(self, task: CERATask) -> TaskResult:
        """Evaluate a single CERA task."""
        start_time = time.time()
        
        # Tokenize input
        if self.tokenizer is not None:
            input_ids = self.tokenizer.encode(
                task.input_text,
                return_tensors="pt",
            ).to(self.device)
        else:
            # Fallback: use hash-based pseudo-tokenization for testing
            input_ids = self._pseudo_tokenize(task.input_text).to(self.device)
        
        # Get model output with routing information
        with torch.no_grad():
            output = self.model(input_ids)
            routing_map = self.model.get_routing_map(input_ids)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Extract routing decisions (average across layers)
        decisions = routing_map["decisions"]  # (n_layers, B, L)
        avg_decisions = decisions.float().mean(dim=0)  # (B, L)
        
        # Per-token dominant pathway
        token_decisions = decisions.mode(dim=0).values.squeeze(0).tolist()  # (L,)
        
        # Compute actual cost
        actual_cost = self._compute_actual_cost(decisions)
        
        # Generate prediction (simplified — in practice would use autoregressive generation)
        predicted = self._generate_prediction(output)
        
        # Check correctness (simplified matching)
        correct = self._check_correct(predicted, task.target_text)
        
        return TaskResult(
            task_id=task.task_id,
            category=task.category,
            difficulty=task.difficulty,
            correct=correct,
            predicted=predicted,
            target=task.target_text,
            routing_decisions=token_decisions,
            compute_cost=actual_cost,
            target_compute=task.target_compute_cost,
            confidence=output["logits"].softmax(dim=-1).max().item(),
            latency_ms=elapsed_ms,
        )
    
    def _compute_actual_cost(self, decisions: torch.Tensor) -> float:
        """
        Compute actual normalized compute cost from routing decisions.
        
        decisions: (n_layers, batch, seq_len) — pathway indices per token per layer
        """
        costs = torch.tensor(self.PATHWAY_COSTS, device=decisions.device)
        
        # Map decisions to costs
        token_costs = costs[decisions.long()]  # Same shape as decisions
        
        # Average cost across all tokens and layers
        avg_cost = token_costs.float().mean().item()
        
        return avg_cost
    
    def _generate_prediction(self, output: Dict) -> str:
        """
        Generate a prediction string from model output.
        
        In a full implementation, this would use autoregressive decoding.
        For the benchmark framework, we use argmax of the last token's logits.
        """
        logits = output["logits"]  # (B, L, vocab_size)
        
        # Take last token prediction
        last_logits = logits[:, -1, :]  # (B, vocab_size)
        predicted_ids = last_logits.argmax(dim=-1)  # (B,)
        
        if self.tokenizer is not None:
            return self.tokenizer.decode(predicted_ids)
        else:
            return str(predicted_ids.item())
    
    def _check_correct(self, predicted: str, target: str) -> bool:
        """
        Check if prediction matches target.
        
        Uses fuzzy matching for robustness:
        - Strips whitespace
        - Case-insensitive
        - Number comparison with tolerance
        """
        pred = predicted.strip().lower()
        tgt = target.strip().lower()
        
        # Exact match
        if pred == tgt:
            return True
        
        # Try numeric comparison
        try:
            pred_num = float(pred)
            tgt_num = float(tgt)
            return abs(pred_num - tgt_num) < 0.01
        except ValueError:
            pass
        
        # Check if target is contained in prediction
        if tgt in pred:
            return True
        
        return False
    
    def _pseudo_tokenize(self, text: str) -> torch.Tensor:
        """
        Pseudo-tokenization for testing without a real tokenizer.
        Maps characters to integer IDs.
        """
        vocab_size = self.model.config.vocab_size
        ids = [hash(c) % vocab_size for c in text[:self.model.config.max_seq_len]]
        return torch.tensor([ids], dtype=torch.long)
    
    def get_routing_analysis(self, text: str) -> Dict:
        """
        Analyze how the model routes a specific input.
        
        Returns detailed per-token, per-layer routing information.
        Useful for qualitative analysis of routing behavior.
        """
        if self.tokenizer is not None:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        else:
            input_ids = self._pseudo_tokenize(text).to(self.device)
            tokens = list(text[:input_ids.shape[1]])
        
        self.model.eval()
        with torch.no_grad():
            routing_map = self.model.get_routing_map(input_ids)
        
        decisions = routing_map["decisions"].squeeze(1)  # (n_layers, L)
        complexity = routing_map["complexity_map"].squeeze(1)  # (n_layers, L)
        
        pathway_names = ["Stream", "Focus", "Reason"]
        
        per_token = []
        for t in range(len(tokens)):
            token_info = {
                "token": tokens[t],
                "dominant_pathway": pathway_names[
                    decisions[:, t].mode().values.item()
                ],
                "complexity_mean": complexity[:, t].mean().item(),
                "per_layer": [
                    pathway_names[decisions[l, t].item()]
                    for l in range(decisions.shape[0])
                ],
            }
            per_token.append(token_info)
        
        return {
            "input": text,
            "n_tokens": len(tokens),
            "n_layers": decisions.shape[0],
            "per_token": per_token,
            "overall_distribution": {
                name: (decisions == i).float().mean().item()
                for i, name in enumerate(pathway_names)
            },
        }
