"""
Arithmetic Reasoning Task Generator

Generates arithmetic problems requiring varying numbers of reasoning steps:
    Level 1: Single operation (2 + 3)
    Level 2: Two operations (2 + 3 * 4)
    Level 3: Multi-step with order of operations ((2 + 3) * 4 - 1)
    Level 4: Multi-step with variables (if x = 3 and y = x + 2, what is x * y?)
    Level 5: Complex word problems requiring multiple steps
"""

import random
from typing import Dict


class ArithmeticGenerator:
    """Generates arithmetic reasoning tasks at various difficulty levels."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
    
    def generate(self, difficulty: int) -> Dict:
        """Generate a single arithmetic task at the given difficulty."""
        generators = {
            1: self._level_1,
            2: self._level_2,
            3: self._level_3,
            4: self._level_4,
            5: self._level_5,
        }
        return generators[difficulty]()
    
    def _level_1(self) -> Dict:
        """Single arithmetic operation."""
        a = self.rng.randint(1, 100)
        b = self.rng.randint(1, 100)
        op = self.rng.choice(['+', '-', '*'])
        
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        else:
            a = self.rng.randint(1, 12)
            b = self.rng.randint(1, 12)
            result = a * b
        
        return {
            "input": f"Calculate: {a} {op} {b} = ?",
            "target": str(result),
            "reasoning_steps": 1,
            "metadata": {"operation": op, "operands": [a, b]},
        }
    
    def _level_2(self) -> Dict:
        """Two arithmetic operations."""
        a = self.rng.randint(1, 50)
        b = self.rng.randint(1, 50)
        c = self.rng.randint(1, 20)
        
        templates = [
            (f"{a} + {b} - {c}", a + b - c),
            (f"{a} * {c} + {b}", a * c + b),
            (f"({a} + {b}) * {c}", (a + b) * c),
            (f"{a} * {b} - {c}", a * b - c),
        ]
        
        expr, result = self.rng.choice(templates)
        
        return {
            "input": f"Calculate: {expr} = ?",
            "target": str(result),
            "reasoning_steps": 2,
            "metadata": {"expression": expr},
        }
    
    def _level_3(self) -> Dict:
        """Multi-step with order of operations."""
        a = self.rng.randint(2, 20)
        b = self.rng.randint(2, 15)
        c = self.rng.randint(2, 10)
        d = self.rng.randint(1, 10)
        
        templates = [
            (
                f"({a} + {b}) * ({c} - {d})",
                (a + b) * (c - d),
                4,
            ),
            (
                f"{a} * {b} + {c} * {d} - {a}",
                a * b + c * d - a,
                4,
            ),
            (
                f"({a} * {b} + {c}) / {d}" if (a * b + c) % d == 0 
                else f"({a} + {b}) * {c} - {d}",
                (a * b + c) // d if (a * b + c) % d == 0 
                else (a + b) * c - d,
                4,
            ),
        ]
        
        expr, result, steps = self.rng.choice(templates)
        
        return {
            "input": f"Calculate step by step: {expr} = ?",
            "target": str(result),
            "reasoning_steps": steps,
            "metadata": {"expression": expr},
        }
    
    def _level_4(self) -> Dict:
        """Multi-step with variables."""
        x = self.rng.randint(2, 15)
        y_offset = self.rng.randint(1, 10)
        y = x + y_offset
        
        operations = [
            (
                f"If x = {x} and y = x + {y_offset}, what is x * y?",
                x * y,
                3,
            ),
            (
                f"If a = {x}, b = a * 2, and c = a + b, what is c?",
                x + x * 2,
                3,
            ),
            (
                f"If n = {x}, then compute n^2 + 2*n + 1",
                x**2 + 2*x + 1,
                4,
            ),
            (
                f"If p = {x} and q = {y}, what is (p + q)^2 - (p - q)^2?",
                (x + y)**2 - (x - y)**2,
                5,
            ),
        ]
        
        question, result, steps = self.rng.choice(operations)
        
        return {
            "input": question,
            "target": str(result),
            "reasoning_steps": steps,
            "metadata": {"variables": {"x": x, "y": y}},
        }
    
    def _level_5(self) -> Dict:
        """Complex word problems."""
        # Generate a multi-step word problem
        price = self.rng.randint(5, 50)
        quantity = self.rng.randint(2, 10)
        discount_pct = self.rng.choice([10, 15, 20, 25])
        tax_pct = self.rng.choice([5, 8, 10])
        
        subtotal = price * quantity
        discount = subtotal * discount_pct / 100
        after_discount = subtotal - discount
        tax = after_discount * tax_pct / 100
        total = after_discount + tax
        
        problem = (
            f"A store sells items for ${price} each. A customer buys {quantity} items. "
            f"The store offers a {discount_pct}% discount on the total. "
            f"After the discount, a {tax_pct}% tax is applied. "
            f"What is the final total? (Round to 2 decimal places)"
        )
        
        return {
            "input": problem,
            "target": f"{total:.2f}",
            "reasoning_steps": 5,
            "metadata": {
                "price": price,
                "quantity": quantity,
                "discount_pct": discount_pct,
                "tax_pct": tax_pct,
                "subtotal": subtotal,
                "after_discount": after_discount,
                "total": total,
            },
        }
