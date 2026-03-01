"""
Curriculum Complexity Scheduling — Algorithmic Innovation #3

A training strategy that gradually increases input complexity to allow
stable routing emergence:

Phase 1 (Warmup): All tokens routed to Stream (SSM)
    - The model learns basic language modeling with cheap compute
    - Router temperature is high (near-uniform routing)
    - Only Stream pathway receives meaningful gradients

Phase 2 (Emergence): Routing begins to differentiate
    - Temperature anneals, routing becomes sharper
    - Input sequences include increasingly complex patterns
    - Focus pathway starts receiving targeted tokens

Phase 3 (Maturation): Full routing active
    - All three pathways are utilized
    - Router makes confident, cost-aware decisions
    - Overall compute is ~40-60% of a pure transformer

Phase 4 (Refinement): Fine-tuning of routing decisions
    - Lowest temperature, hardest routing
    - Model optimizes the efficiency-accuracy trade-off

Innovation Insight:
    Standard training of mixture/routing architectures is unstable because
    early random routing creates inconsistent gradient signals. Curriculum
    scheduling solves this by ensuring each pathway is trained on 
    appropriate data before the router needs to make decisions.

    This is analogous to how humans learn: master basic operations first,
    then learn when to apply complex reasoning.
"""

import math
from typing import Dict, Optional, Callable

import torch


class CurriculumScheduler:
    """
    Controls curriculum progression and temperature annealing.
    
    Manages:
    1. Temperature annealing for Gumbel-Softmax routing
    2. Complexity scaling for training data
    3. Loss weight scheduling for auxiliary losses
    4. Phase transitions and logging
    """
    
    PHASES = {
        0: "Warmup",
        1: "Emergence",
        2: "Maturation",
        3: "Refinement",
    }
    
    def __init__(self, config):
        self.config = config
        self.current_step = 0
        self.warmup_steps = config.curriculum_warmup_steps
        self.max_steps = config.max_steps
        self.schedule_type = config.curriculum_schedule
        
        # Temperature range
        self.initial_temperature = config.router_temperature
        self.min_temperature = config.router_min_temperature
        
        # Phase boundaries (as fractions of total training)
        self.phase_boundaries = [0.1, 0.4, 0.75, 1.0]
        
        # Complexity progression
        self.max_complexity = config.curriculum_max_complexity
        
        # Tracking
        self._phase_history = []
    
    def step(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Advance the curriculum by one step and update model parameters.
        
        Args:
            model: The HYDRA model whose router temperatures need updating
            
        Returns:
            dict with current curriculum parameters
        """
        self.current_step += 1
        progress = self.current_step / self.max_steps
        
        # Determine current phase
        phase = self._get_phase(progress)
        
        # Compute temperature
        temperature = self._compute_temperature(progress)
        
        # Compute complexity level
        complexity = self._compute_complexity(progress)
        
        # Compute auxiliary loss weights
        aux_weights = self._compute_aux_weights(progress, phase)
        
        # Update model router temperatures
        self._update_model(model, temperature, aux_weights)
        
        info = {
            "step": self.current_step,
            "progress": progress,
            "phase": phase,
            "phase_name": self.PHASES[phase],
            "temperature": temperature,
            "complexity": complexity,
            "aux_loss_weight": aux_weights["aux"],
            "budget_loss_weight": aux_weights["budget"],
            "efficiency_loss_weight": aux_weights["efficiency"],
        }
        
        self._phase_history.append(info)
        
        return info
    
    def _get_phase(self, progress: float) -> int:
        """Determine current training phase."""
        for i, boundary in enumerate(self.phase_boundaries):
            if progress <= boundary:
                return i
        return len(self.phase_boundaries) - 1
    
    def _compute_temperature(self, progress: float) -> float:
        """
        Compute Gumbel-Softmax temperature based on training progress.
        
        Strategies:
        - Linear: straight line from initial to minimum
        - Cosine: smooth cosine curve (gentler at start and end)
        - Step: discrete drops at phase boundaries
        """
        if progress < 0.05:
            # Keep high temperature during very early training
            return self.initial_temperature
        
        if self.schedule_type == "linear":
            t = max(0, min(1, (progress - 0.05) / 0.9))
            temperature = self.initial_temperature + t * (
                self.min_temperature - self.initial_temperature
            )
        
        elif self.schedule_type == "cosine":
            t = max(0, min(1, (progress - 0.05) / 0.9))
            cosine_val = 0.5 * (1 + math.cos(math.pi * t))
            temperature = (
                self.min_temperature
                + (self.initial_temperature - self.min_temperature) * cosine_val
            )
        
        elif self.schedule_type == "step":
            phase = self._get_phase(progress)
            temperatures = [
                self.initial_temperature,
                self.initial_temperature * 0.5,
                self.min_temperature * 2,
                self.min_temperature,
            ]
            temperature = temperatures[phase]
        
        else:
            temperature = self.initial_temperature
        
        return max(temperature, self.min_temperature)
    
    def _compute_complexity(self, progress: float) -> float:
        """
        Compute target complexity level for data generation.
        
        Returns a value in [0, max_complexity] that controls:
        - Sequence length
        - Task difficulty
        - Number of reasoning steps required
        """
        if progress < 0.05:
            return 0.1  # Minimum complexity during warmup
        
        # Gradual ramp-up
        t = max(0, min(1, (progress - 0.05) / 0.85))
        
        if self.schedule_type == "cosine":
            complexity = 0.5 * (1 - math.cos(math.pi * t))
        elif self.schedule_type == "linear":
            complexity = t
        elif self.schedule_type == "step":
            phase = self._get_phase(progress)
            complexities = [0.2, 0.5, 0.8, 1.0]
            complexity = complexities[phase]
        else:
            complexity = t
        
        return complexity * self.max_complexity
    
    def _compute_aux_weights(
        self, progress: float, phase: int
    ) -> Dict[str, float]:
        """
        Schedule auxiliary loss weights across phases.
        
        Phase 0 (Warmup): Low aux weights — let model learn basics
        Phase 1 (Emergence): Ramp up — start shaping routing
        Phase 2 (Maturation): Full weights — optimize routing
        Phase 3 (Refinement): Slightly reduced — don't overfit routing
        """
        base_aux = self.config.router_aux_loss_weight
        base_budget = self.config.router_budget_loss_weight
        base_efficiency = 0.1  # Default efficiency weight
        
        weight_schedules = {
            0: {"aux": 0.2, "budget": 0.1, "efficiency": 0.05},
            1: {"aux": 0.6, "budget": 0.5, "efficiency": 0.3},
            2: {"aux": 1.0, "budget": 1.0, "efficiency": 1.0},
            3: {"aux": 0.8, "budget": 1.2, "efficiency": 0.8},
        }
        
        schedule = weight_schedules[phase]
        
        return {
            "aux": base_aux * schedule["aux"],
            "budget": base_budget * schedule["budget"],
            "efficiency": base_efficiency * schedule["efficiency"],
        }
    
    def _update_model(
        self,
        model: torch.nn.Module,
        temperature: float,
        aux_weights: Dict[str, float],
    ):
        """Update model parameters based on curriculum state."""
        # Update router temperatures across all blocks
        for module in model.modules():
            if hasattr(module, 'set_temperature'):
                module.set_temperature(temperature)
            if hasattr(module, 'aux_loss_weight'):
                module.aux_loss_weight = aux_weights["aux"]
            if hasattr(module, 'budget_loss_weight'):
                module.budget_loss_weight = aux_weights["budget"]
    
    def get_current_info(self) -> Dict:
        """Get current curriculum state."""
        if self._phase_history:
            return self._phase_history[-1]
        
        return {
            "step": 0,
            "progress": 0,
            "phase": 0,
            "phase_name": "Not Started",
            "temperature": self.initial_temperature,
            "complexity": 0.0,
        }
    
    def state_dict(self) -> Dict:
        return {
            "current_step": self.current_step,
        }
    
    def load_state_dict(self, state_dict: Dict):
        self.current_step = state_dict["current_step"]


class TemperatureCallback:
    """
    Callback for monitoring and logging temperature changes.
    
    Can be registered with the trainer to log temperature-related
    metrics at specified intervals.
    """
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.temperature_history = []
        self.routing_history = []
    
    def __call__(
        self,
        step: int,
        curriculum_info: Dict,
        routing_stats: Optional[Dict] = None,
    ):
        self.temperature_history.append({
            "step": step,
            "temperature": curriculum_info["temperature"],
            "phase": curriculum_info["phase_name"],
        })
        
        if routing_stats:
            self.routing_history.append({
                "step": step,
                **routing_stats,
            })
    
    def get_summary(self) -> Dict:
        """Get summary of temperature and routing progression."""
        if not self.temperature_history:
            return {}
        
        return {
            "temperature_range": (
                self.temperature_history[0]["temperature"],
                self.temperature_history[-1]["temperature"],
            ),
            "phases_traversed": list(set(
                t["phase"] for t in self.temperature_history
            )),
            "n_recordings": len(self.temperature_history),
        }
