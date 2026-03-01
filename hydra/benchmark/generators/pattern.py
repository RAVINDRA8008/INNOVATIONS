"""
Pattern Recognition Task Generator

Generates sequence/pattern recognition problems:
    Level 1: Simple arithmetic sequences (2, 4, 6, ?)
    Level 2: Geometric or alternating patterns
    Level 3: Multi-rule patterns (Fibonacci-like, interleaved sequences)
    Level 4: Matrix/2D pattern completion
    Level 5: Abstract transformation rules on symbolic sequences
"""

import random
from typing import Dict, List


class PatternRecognitionGenerator:
    """Generates pattern recognition tasks at various difficulty levels."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
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
        """Simple arithmetic sequences."""
        start = self.rng.randint(1, 20)
        step = self.rng.randint(1, 10)
        
        seq = [start + i * step for i in range(6)]
        visible = seq[:5]
        answer = seq[5]
        
        return {
            "input": f"What comes next in the sequence? {', '.join(map(str, visible))}, ?",
            "target": str(answer),
            "reasoning_steps": 1,
            "metadata": {"pattern": "arithmetic", "step": step},
        }
    
    def _level_2(self) -> Dict:
        """Geometric or alternating patterns."""
        pattern_type = self.rng.choice(["geometric", "alternating", "square"])
        
        if pattern_type == "geometric":
            start = self.rng.randint(1, 5)
            ratio = self.rng.choice([2, 3])
            seq = [start * (ratio ** i) for i in range(6)]
        
        elif pattern_type == "alternating":
            a_start = self.rng.randint(1, 10)
            b_start = self.rng.randint(20, 30)
            a_step = self.rng.randint(1, 5)
            b_step = self.rng.randint(1, 5)
            seq = []
            for i in range(3):
                seq.append(a_start + i * a_step)
                seq.append(b_start + i * b_step)
        
        else:  # square
            start = self.rng.randint(1, 5)
            seq = [(start + i) ** 2 for i in range(6)]
        
        visible = seq[:5]
        answer = seq[5]
        
        return {
            "input": f"Find the next number: {', '.join(map(str, visible))}, ?",
            "target": str(answer),
            "reasoning_steps": 2,
            "metadata": {"pattern": pattern_type},
        }
    
    def _level_3(self) -> Dict:
        """Multi-rule patterns."""
        pattern_type = self.rng.choice(["fibonacci_like", "interleaved", "cumulative"])
        
        if pattern_type == "fibonacci_like":
            a, b = self.rng.randint(1, 5), self.rng.randint(1, 5)
            seq = [a, b]
            for i in range(6):
                seq.append(seq[-1] + seq[-2])
            seq = seq[:8]
        
        elif pattern_type == "interleaved":
            # Two independent sequences interleaved
            seq1_start = self.rng.randint(1, 5)
            seq2_start = self.rng.randint(10, 20)
            step1 = self.rng.randint(2, 5)
            step2 = self.rng.randint(-3, -1)
            
            seq = []
            for i in range(4):
                seq.append(seq1_start + i * step1)
                seq.append(seq2_start + i * step2)
        
        else:  # cumulative
            base = [self.rng.randint(1, 5) for _ in range(8)]
            seq = [base[0]]
            for i in range(1, 8):
                seq.append(seq[-1] + base[i])
        
        visible = seq[:7]
        answer = seq[7]
        
        return {
            "input": (
                f"Identify the pattern and find the next number:\n"
                f"{', '.join(map(str, visible))}, ?"
            ),
            "target": str(answer),
            "reasoning_steps": 4,
            "metadata": {"pattern": pattern_type},
        }
    
    def _level_4(self) -> Dict:
        """Matrix/2D pattern completion."""
        # Create a 3x3 grid with a pattern, ask for missing element
        rule = self.rng.choice(["row_sum", "diagonal", "multiply"])
        
        if rule == "row_sum":
            grid = []
            for _ in range(3):
                a = self.rng.randint(1, 10)
                b = self.rng.randint(1, 10)
                grid.append([a, b, a + b])
            
            # Hide one entry
            hide_row = self.rng.randint(0, 2)
            answer = grid[hide_row][2]
            grid[hide_row][2] = "?"
            
            pattern_hint = "Each row follows a consistent rule."
        
        elif rule == "diagonal":
            base = self.rng.randint(1, 5)
            grid = [
                [base, base + 1, base + 2],
                [base + 1, base + 2, base + 3],
                [base + 2, base + 3, "?"],
            ]
            answer = base + 4
            pattern_hint = "Look at the diagonals."
        
        else:  # multiply
            a = self.rng.randint(2, 5)
            b = self.rng.randint(2, 5)
            grid = [
                [a, b, a * b],
                [a + 1, b, (a + 1) * b],
                [a + 2, b, "?"],
            ]
            answer = (a + 2) * b
            pattern_hint = "Look for the relationship between columns."
        
        grid_str = "\n".join(
            "  ".join(str(cell).rjust(4) for cell in row)
            for row in grid
        )
        
        return {
            "input": f"Complete the grid (? = unknown):\n{grid_str}\n\nHint: {pattern_hint}",
            "target": str(answer),
            "reasoning_steps": 5,
            "metadata": {"rule": rule, "grid": grid},
        }
    
    def _level_5(self) -> Dict:
        """Abstract transformation rules on symbolic sequences."""
        # Define a transformation and ask to apply it
        symbols = ['A', 'B', 'C', 'D', 'E']
        
        rule_type = self.rng.choice(["cipher", "reversal_shift", "conditional"])
        
        if rule_type == "cipher":
            # Simple substitution cipher with a twist
            shift = self.rng.randint(1, 3)
            input_seq = [self.rng.choice(symbols) for _ in range(5)]
            
            examples = []
            for s in symbols[:3]:
                idx = symbols.index(s)
                new_idx = (idx + shift) % len(symbols)
                examples.append(f"{s} → {symbols[new_idx]}")
            
            test_seq = [self.rng.choice(symbols) for _ in range(4)]
            answer_seq = [
                symbols[(symbols.index(s) + shift) % len(symbols)]
                for s in test_seq
            ]
            
            problem = (
                f"Given this transformation rule:\n"
                + "\n".join(f"  {e}" for e in examples)
                + f"\n\nApply the rule to: {' '.join(test_seq)}"
                + f"\nWhat is the result?"
            )
            
            answer = " ".join(answer_seq)
        
        elif rule_type == "reversal_shift":
            n = self.rng.randint(3, 5)
            seq1 = list(range(1, n + 1))
            # Rule: reverse and add 1 to each
            result1 = [x + 1 for x in reversed(seq1)]
            
            seq2 = [self.rng.randint(1, 9) for _ in range(n)]
            result2 = [x + 1 for x in reversed(seq2)]
            
            problem = (
                f"Given the transformation:\n"
                f"  {seq1} → {result1}\n"
                f"\nApply the same rule to: {seq2}\nWhat is the result?"
            )
            answer = str(result2)
        
        else:  # conditional
            problem = (
                "Apply these rules to transform the sequence:\n"
                "  - If a number is even, multiply by 2\n"
                "  - If a number is odd, add 3\n"
                "  - Apply the rules twice\n"
            )
            
            seq = [self.rng.randint(1, 10) for _ in range(4)]
            
            # Apply once
            step1 = [x * 2 if x % 2 == 0 else x + 3 for x in seq]
            # Apply twice
            step2 = [x * 2 if x % 2 == 0 else x + 3 for x in step1]
            
            problem += f"\nInput: {seq}\nWhat is the final result?"
            answer = str(step2)
        
        return {
            "input": problem,
            "target": answer,
            "reasoning_steps": 6,
            "metadata": {"rule_type": rule_type},
        }
