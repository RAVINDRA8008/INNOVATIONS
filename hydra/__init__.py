"""
HYDRA: Hybrid Dynamic Routing Architecture

A research-grade neural architecture that dynamically routes tokens through
multiple computational paradigms based on learned complexity estimation.

Pathways:
    - Stream (SSM): O(n) state-space model for routine/low-complexity tokens
    - Focus (Windowed Attention): O(n*w) local attention for moderate complexity
    - Reason (Global Attention): O(n^2) full attention for critical reasoning tokens

Key Innovations:
    1. Differentiable multi-pathway routing via Gumbel-Softmax
    2. Adaptive Compute Budget Loss for compute-proportional training
    3. Curriculum Complexity Scheduling for stable pathway emergence
    4. CERA Benchmark for compute-efficient reasoning assessment

Author: INNOVATION Research
"""

__version__ = "0.1.0"
__codename__ = "HYDRA"
