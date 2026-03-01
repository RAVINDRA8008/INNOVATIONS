"""
Logical Reasoning Task Generator

Generates logical reasoning problems:
    Level 1: Simple boolean evaluation (True AND False)
    Level 2: Two-step implications (If A then B; A is true; is B true?)
    Level 3: Syllogisms (All X are Y; Z is X; Is Z a Y?)
    Level 4: Multi-step deduction chains
    Level 5: Complex scenarios with negation, quantifiers, and multiple rules
"""

import random
from typing import Dict, List


class LogicalReasoningGenerator:
    """Generates logical reasoning tasks at various difficulty levels."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        
        # Entity pools for generating varied problems
        self.animals = ["cat", "dog", "bird", "fish", "horse", "rabbit", "eagle", "whale"]
        self.colors = ["red", "blue", "green", "yellow", "orange", "purple"]
        self.properties = ["fast", "tall", "strong", "wise", "brave", "gentle"]
        self.names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace"]
    
    def generate(self, difficulty: int) -> Dict:
        generators = {
            1: self._level_1,
            2: self._level_2,
            3: self._level_3,
            4: self._level_4,
            5: self._level_5,
        }
        return generators[difficulty]()
    
    def _level_1(self) -> Dict:
        """Simple boolean evaluation."""
        a = self.rng.choice([True, False])
        b = self.rng.choice([True, False])
        op = self.rng.choice(["AND", "OR", "XOR"])
        
        if op == "AND":
            result = a and b
        elif op == "OR":
            result = a or b
        else:
            result = a ^ b
        
        return {
            "input": f"Evaluate: {a} {op} {b} = ?",
            "target": str(result),
            "reasoning_steps": 1,
            "metadata": {"operation": op, "operands": [a, b]},
        }
    
    def _level_2(self) -> Dict:
        """Simple implications."""
        a_name, b_name = self.rng.sample(self.names, 2)
        prop1 = self.rng.choice(self.properties)
        prop2 = self.rng.choice(self.properties)
        
        a_has_prop = self.rng.choice([True, False])
        
        if a_has_prop:
            problem = (
                f"Rule: If {a_name} is {prop1}, then {b_name} is {prop2}. "
                f"Fact: {a_name} is {prop1}. "
                f"Question: Is {b_name} {prop2}?"
            )
            answer = "Yes"
        else:
            problem = (
                f"Rule: If {a_name} is {prop1}, then {b_name} is {prop2}. "
                f"Fact: {a_name} is NOT {prop1}. "
                f"Question: Can we conclude that {b_name} is {prop2}?"
            )
            answer = "Cannot be determined"
        
        return {
            "input": problem,
            "target": answer,
            "reasoning_steps": 2,
            "metadata": {"type": "modus_ponens"},
        }
    
    def _level_3(self) -> Dict:
        """Syllogisms."""
        category1 = self.rng.choice(self.animals)
        category2 = self.rng.choice(self.animals)
        while category2 == category1:
            category2 = self.rng.choice(self.animals)
        
        prop = self.rng.choice(self.properties)
        name = self.rng.choice(self.names)
        
        valid = self.rng.choice([True, False])
        
        if valid:
            problem = (
                f"Premise 1: All {category1}s are {prop}. "
                f"Premise 2: {name}'s pet is a {category1}. "
                f"Conclusion: {name}'s pet is {prop}. "
                f"Is this conclusion valid?"
            )
            answer = "Valid"
        else:
            problem = (
                f"Premise 1: All {category1}s are {prop}. "
                f"Premise 2: {name}'s pet is {prop}. "
                f"Conclusion: {name}'s pet is a {category1}. "
                f"Is this conclusion valid?"
            )
            answer = "Invalid (affirming the consequent)"
        
        return {
            "input": problem,
            "target": answer,
            "reasoning_steps": 3,
            "metadata": {"type": "syllogism", "valid": valid},
        }
    
    def _level_4(self) -> Dict:
        """Multi-step deduction chains."""
        names = self.rng.sample(self.names, 4)
        props = self.rng.sample(self.properties, 4)
        
        # Create a chain: A→B→C→conclusion
        rules = [
            f"If {names[0]} is {props[0]}, then {names[1]} is {props[1]}.",
            f"If {names[1]} is {props[1]}, then {names[2]} is {props[2]}.",
            f"If {names[2]} is {props[2]}, then {names[3]} is {props[3]}.",
        ]
        
        start_true = self.rng.choice([True, False])
        
        if start_true:
            fact = f"{names[0]} is {props[0]}."
            question = f"Is {names[3]} {props[3]}?"
            answer = "Yes"
            steps = 4
        else:
            fact = f"{names[0]} is NOT {props[0]}."
            question = f"Must {names[3]} be {props[3]}?"
            answer = "Not necessarily"
            steps = 3
        
        problem = (
            "Given the following rules:\n"
            + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(rules))
            + f"\n\nFact: {fact}\nQuestion: {question}"
        )
        
        return {
            "input": problem,
            "target": answer,
            "reasoning_steps": steps,
            "metadata": {"type": "chain_reasoning", "chain_length": 3},
        }
    
    def _level_5(self) -> Dict:
        """Complex multi-rule scenarios."""
        names = self.rng.sample(self.names, 5)
        
        # Create a complex scenario with multiple rules and facts
        rules = [
            f"If {names[0]} goes to the party, then {names[1]} also goes.",
            f"If {names[1]} goes to the party, then {names[2]} does NOT go.",
            f"If {names[2]} does not go, then {names[3]} goes.",
            f"If {names[3]} goes and {names[1]} goes, then {names[4]} brings cake.",
            f"Either {names[0]} or {names[2]} goes (or both).",
        ]
        
        # Scenario: names[0] goes → names[1] goes → names[2] doesn't →
        # names[3] goes → names[1] AND names[3] go → names[4] brings cake
        
        problem = (
            "Given these rules about a party:\n"
            + "\n".join(f"  {i+1}. {r}" for i, r in enumerate(rules))
            + f"\n\nFact: {names[0]} decides to go to the party."
            + f"\n\nQuestion: Does {names[4]} bring cake? "
            + "Explain step by step."
        )
        
        answer = (
            f"Yes. Step 1: {names[0]} goes → {names[1]} goes (Rule 1). "
            f"Step 2: {names[1]} goes → {names[2]} doesn't go (Rule 2). "
            f"Step 3: {names[2]} doesn't go → {names[3]} goes (Rule 3). "
            f"Step 4: {names[3]} goes AND {names[1]} goes → "
            f"{names[4]} brings cake (Rule 4)."
        )
        
        return {
            "input": problem,
            "target": answer,
            "reasoning_steps": 5,
            "metadata": {"type": "multi_rule", "n_rules": 5},
        }
