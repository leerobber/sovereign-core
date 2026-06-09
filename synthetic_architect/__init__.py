"""
synthetic_architect/__init__.py
Synthetic Architect (SAS-1, KAN-88) — NAS-driven micro-model evolution engine.
First primitive: mHC (Manifold-Constrained Hyper-Connections, arXiv:2512.24880).
"""
from .mhc_layer import MHCLayer, SinkhornKnopp, OrthostochasticNewtonSchulz
from .nas_primitives import NASPrimitive, PRIMITIVE_REGISTRY, register_primitive
from .micro_model_gene import MicroModelGene, GeneSearchSpace

__all__ = [
    "MHCLayer",
    "SinkhornKnopp",
    "OrthostochasticNewtonSchulz",
    "NASPrimitive",
    "PRIMITIVE_REGISTRY",
    "register_primitive",
    "MicroModelGene",
    "GeneSearchSpace",
]
