"""
CERA Benchmark — Compute-Efficient Reasoning Assessment

A novel benchmark that measures not just accuracy, but the EFFICIENCY
of compute allocation during reasoning.

Core Innovation:
    Traditional benchmarks ask: "Can the model solve this?"
    CERA asks: "Can the model solve this with PROPORTIONAL compute?"

    A model that uses O(n^2) attention for "2 + 2 = ?" is wasteful.
    A model that uses O(n) SSM for complex logical chains is underpowered.
    The IDEAL model allocates compute proportionally to task complexity.

CERA Score:
    CERA = Accuracy × Efficiency
    
    Efficiency = target_compute / actual_compute
    (capped at 1.0 — being cheaper than target is fine)
    
    This means a model with 90% accuracy and 80% efficiency
    scores 0.72, beating a model with 95% accuracy but 40% efficiency
    (score 0.38).

Task Categories:
    1. Arithmetic: From simple addition to multi-step calculations
    2. Logical Reasoning: Propositional logic, syllogisms, boolean chains
    3. Pattern Recognition: Sequence completion, transformation rules
    4. Compositional: Tasks requiring multiple reasoning types

Difficulty Levels:
    Level 1 (Trivial): Single-step, should use Stream (SSM)
    Level 2 (Easy): 2-3 steps, Stream or Focus
    Level 3 (Medium): 4-6 steps, Focus recommended
    Level 4 (Hard): 7-10 steps, Focus or Reason
    Level 5 (Expert): 10+ steps, Reason pathway justified
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random

from .generators.arithmetic import ArithmeticGenerator
from .generators.logical import LogicalReasoningGenerator
from .generators.pattern import PatternRecognitionGenerator
from .generators.compositional import CompositionalGenerator


@dataclass
class CERATask:
    """A single CERA benchmark task."""
    task_id: str
    category: str           # "arithmetic", "logical", "pattern", "compositional"
    difficulty: int          # 1-5
    input_text: str          # Input to the model
    target_text: str         # Expected output
    reasoning_steps: int     # Number of reasoning steps required
    target_pathway: str      # Recommended pathway ("stream", "focus", "reason")
    target_compute_cost: float  # Expected normalized compute
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CERABenchmark:
    """
    CERA Benchmark Suite.
    
    Generates and manages benchmark tasks across all categories
    and difficulty levels.
    """
    
    # Mapping from difficulty to recommended pathway
    DIFFICULTY_PATHWAY_MAP = {
        1: ("stream", 1.0),
        2: ("stream", 1.5),
        3: ("focus", 3.0),
        4: ("focus", 5.0),
        5: ("reason", 10.0),
    }
    
    def __init__(
        self,
        seed: int = 42,
        tasks_per_category_per_level: int = 50,
    ):
        self.seed = seed
        self.tasks_per_category_per_level = tasks_per_category_per_level
        
        # Initialize generators
        self.generators = {
            "arithmetic": ArithmeticGenerator(seed=seed),
            "logical": LogicalReasoningGenerator(seed=seed + 1),
            "pattern": PatternRecognitionGenerator(seed=seed + 2),
            "compositional": CompositionalGenerator(seed=seed + 3),
        }
        
        self.tasks: List[CERATask] = []
        self._generated = False
    
    def generate(self) -> List[CERATask]:
        """Generate the full benchmark suite."""
        random.seed(self.seed)
        self.tasks = []
        task_counter = 0
        
        for category, generator in self.generators.items():
            for difficulty in range(1, 6):
                pathway, compute_cost = self.DIFFICULTY_PATHWAY_MAP[difficulty]
                
                for i in range(self.tasks_per_category_per_level):
                    task_data = generator.generate(difficulty=difficulty)
                    
                    task = CERATask(
                        task_id=f"CERA-{category[:3].upper()}-L{difficulty}-{task_counter:04d}",
                        category=category,
                        difficulty=difficulty,
                        input_text=task_data["input"],
                        target_text=task_data["target"],
                        reasoning_steps=task_data["reasoning_steps"],
                        target_pathway=pathway,
                        target_compute_cost=compute_cost,
                        metadata=task_data.get("metadata", {}),
                    )
                    
                    self.tasks.append(task)
                    task_counter += 1
        
        self._generated = True
        return self.tasks
    
    def get_tasks_by_category(self, category: str) -> List[CERATask]:
        """Get all tasks for a specific category."""
        return [t for t in self.tasks if t.category == category]
    
    def get_tasks_by_difficulty(self, difficulty: int) -> List[CERATask]:
        """Get all tasks at a specific difficulty level."""
        return [t for t in self.tasks if t.difficulty == difficulty]
    
    def get_task_distribution(self) -> Dict:
        """Get the distribution of tasks across categories and levels."""
        dist = {}
        for task in self.tasks:
            key = f"{task.category}_L{task.difficulty}"
            dist[key] = dist.get(key, 0) + 1
        return dist
    
    def get_summary(self) -> Dict:
        """Get benchmark summary statistics."""
        if not self.tasks:
            return {"status": "not generated"}
        
        return {
            "total_tasks": len(self.tasks),
            "categories": list(self.generators.keys()),
            "difficulty_levels": 5,
            "tasks_per_cell": self.tasks_per_category_per_level,
            "distribution": self.get_task_distribution(),
            "avg_reasoning_steps": sum(t.reasoning_steps for t in self.tasks) / len(self.tasks),
        }
