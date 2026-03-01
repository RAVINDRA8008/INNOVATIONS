"""
Compositional Reasoning Task Generator

Tasks that require combining multiple reasoning types:
    Level 1: Simple lookup + comparison
    Level 2: Arithmetic + logical
    Level 3: Pattern + arithmetic + logic
    Level 4: Multi-step problem solving with constraints
    Level 5: Open-ended multi-domain reasoning

These tasks are the most diagnostic of the routing mechanism because
they contain sub-problems at different complexity levels within the
same sequence — forcing the router to dynamically switch pathways.
"""

import random
from typing import Dict, List


class CompositionalGenerator:
    """Generates compositional reasoning tasks combining multiple types."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]
    
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
        """Simple lookup + comparison."""
        items = {}
        names = self.rng.sample(self.names, 3)
        for name in names:
            items[name] = self.rng.randint(1, 20)
        
        question_type = self.rng.choice(["max", "sum", "diff"])
        
        if question_type == "max":
            max_name = max(items, key=items.get)
            question = (
                f"Scores: {', '.join(f'{n}: {v}' for n, v in items.items())}. "
                f"Who has the highest score?"
            )
            answer = max_name
        elif question_type == "sum":
            total = sum(items.values())
            question = (
                f"Scores: {', '.join(f'{n}: {v}' for n, v in items.items())}. "
                f"What is the total of all scores?"
            )
            answer = str(total)
        else:
            names_list = list(items.keys())
            diff = abs(items[names_list[0]] - items[names_list[1]])
            question = (
                f"Scores: {', '.join(f'{n}: {v}' for n, v in items.items())}. "
                f"What is the difference between {names_list[0]}'s and {names_list[1]}'s score?"
            )
            answer = str(diff)
        
        return {
            "input": question,
            "target": answer,
            "reasoning_steps": 2,
            "metadata": {"type": "lookup_compare", "data": items},
        }
    
    def _level_2(self) -> Dict:
        """Arithmetic + logical reasoning."""
        names = self.rng.sample(self.names, 3)
        ages = {name: self.rng.randint(20, 60) for name in names}
        
        oldest = max(ages, key=ages.get)
        youngest = min(ages, key=ages.get)
        
        question = (
            f"Ages: {', '.join(f'{n} is {a}' for n, a in ages.items())}. "
            f"If the oldest person gives the youngest person half their age "
            f"in dollars, how much money does the youngest person receive?"
        )
        
        answer = str(ages[oldest] // 2)
        
        return {
            "input": question,
            "target": answer,
            "reasoning_steps": 3,
            "metadata": {"type": "arithmetic_logical", "ages": ages},
        }
    
    def _level_3(self) -> Dict:
        """Pattern + arithmetic + logic."""
        # A sequence follows a pattern, need to find next, then apply logic
        start = self.rng.randint(2, 5)
        step = self.rng.randint(2, 4)
        seq = [start + i * step for i in range(5)]
        next_val = start + 5 * step
        
        threshold = self.rng.randint(15, 30)
        
        question = (
            f"A sequence goes: {', '.join(map(str, seq))}...\n"
            f"Find the next number in the sequence. "
            f"Then, determine if that number is greater than {threshold}. "
            f"If yes, multiply it by 2. If no, add 10. "
            f"What is the final result?"
        )
        
        if next_val > threshold:
            final = next_val * 2
        else:
            final = next_val + 10
        
        return {
            "input": question,
            "target": str(final),
            "reasoning_steps": 4,
            "metadata": {
                "type": "pattern_arithmetic_logic",
                "next_val": next_val,
                "threshold": threshold,
            },
        }
    
    def _level_4(self) -> Dict:
        """Multi-step problem solving with constraints."""
        names = self.rng.sample(self.names, 4)
        
        # Resource allocation problem
        total_budget = self.rng.randint(100, 200)
        min_share = self.rng.randint(10, 25)
        
        # Generate constraints
        ratio_a_to_b = self.rng.randint(2, 3)  # A gets ratio_a_to_b times B's share
        
        # Solve: A = ratio * B, C = min_share, D = min_share
        # A + B + C + D = total
        # ratio*B + B + 2*min_share = total
        # B = (total - 2*min_share) / (ratio + 1)
        
        b_share = (total_budget - 2 * min_share) / (ratio_a_to_b + 1)
        a_share = ratio_a_to_b * b_share
        
        question = (
            f"Problem: Divide ${total_budget} among {', '.join(names)}.\n"
            f"Constraints:\n"
            f"  1. {names[0]} gets {ratio_a_to_b} times what {names[1]} gets.\n"
            f"  2. {names[2]} and {names[3]} each get exactly ${min_share}.\n"
            f"  3. All money must be distributed.\n"
            f"How much does {names[0]} get?"
        )
        
        return {
            "input": question,
            "target": f"{a_share:.2f}",
            "reasoning_steps": 5,
            "metadata": {
                "type": "constrained_allocation",
                "total": total_budget,
                "shares": {
                    names[0]: a_share,
                    names[1]: b_share,
                    names[2]: min_share,
                    names[3]: min_share,
                },
            },
        }
    
    def _level_5(self) -> Dict:
        """Complex multi-domain reasoning."""
        names = self.rng.sample(self.names, 4)
        
        # Tournament bracket problem
        skills = {name: self.rng.randint(1, 100) for name in names}
        
        # Round 1
        r1_winner1 = names[0] if skills[names[0]] > skills[names[1]] else names[1]
        r1_winner2 = names[2] if skills[names[2]] > skills[names[3]] else names[3]
        
        # Skill boost: winners get +10, losers are eliminated
        final_skills = {
            r1_winner1: skills[r1_winner1] + 10,
            r1_winner2: skills[r1_winner2] + 10,
        }
        
        # Final
        champion = r1_winner1 if final_skills[r1_winner1] > final_skills[r1_winner2] else r1_winner2
        
        question = (
            f"Tournament Rules:\n"
            f"  - Players: {', '.join(f'{n} (skill: {skills[n]})' for n in names)}\n"
            f"  - Round 1: {names[0]} vs {names[1]}, {names[2]} vs {names[3]}\n"
            f"  - Higher skill wins each match\n"
            f"  - Winners get +10 skill bonus for the final\n"
            f"  - Final: Round 1 winners face off\n\n"
            f"Who wins the tournament? Show the bracket."
        )
        
        answer = (
            f"Round 1: {r1_winner1} beats {'the other' if r1_winner1 == names[0] else names[0]}, "
            f"{r1_winner2} beats {'the other' if r1_winner2 == names[2] else names[2]}. "
            f"Final: {r1_winner1} ({final_skills[r1_winner1]}) vs "
            f"{r1_winner2} ({final_skills[r1_winner2]}). "
            f"Champion: {champion}"
        )
        
        return {
            "input": question,
            "target": champion,
            "reasoning_steps": 7,
            "metadata": {
                "type": "tournament",
                "skills": skills,
                "champion": champion,
                "bracket": {
                    "r1_match1": [names[0], names[1], r1_winner1],
                    "r1_match2": [names[2], names[3], r1_winner2],
                    "final": [r1_winner1, r1_winner2, champion],
                },
            },
        }
