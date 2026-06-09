"""
synthetic_architect/nas_primitives.py

NAS-Searchable Primitive Registry for Sovereign Core's Synthetic Architect (SAS-1).

A NAS primitive is any architectural building block that can be:
  1. Selected by the NAS controller (MicroModelGene bit-string)
  2. Instantiated from a config dict
  3. Benchmarked for flop/param cost

mHC (Manifold-Constrained Hyper-Connections) is the first registered primitive.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Type

import torch.nn as nn

from .mhc_layer import MHCLayer, ProjectionMode


# ── Registry infrastructure ───────────────────────────────────────────────────

@dataclasses.dataclass
class NASPrimitive:
    """Descriptor for a single NAS-searchable architectural primitive."""
    name: str
    description: str
    factory: Callable[..., nn.Module]       # factory(d_model, **kwargs) → Module
    default_kwargs: Dict[str, Any]
    param_estimate: Callable[[int, Dict[str, Any]], int]  # (d_model, kwargs) → n_params
    flop_estimate:  Callable[[int, int, Dict[str, Any]], int]  # (d_model, seq_len, kwargs) → flops

    def build(self, d_model: int, **override_kwargs) -> nn.Module:
        """Instantiate the primitive with optional kwarg overrides."""
        kwargs = {**self.default_kwargs, **override_kwargs}
        return self.factory(d_model=d_model, **kwargs)

    def estimate_params(self, d_model: int, **override_kwargs) -> int:
        kwargs = {**self.default_kwargs, **override_kwargs}
        return self.param_estimate(d_model, kwargs)

    def estimate_flops(self, d_model: int, seq_len: int, **override_kwargs) -> int:
        kwargs = {**self.default_kwargs, **override_kwargs}
        return self.flop_estimate(d_model, seq_len, kwargs)


PRIMITIVE_REGISTRY: Dict[str, NASPrimitive] = {}


def register_primitive(primitive: NASPrimitive) -> NASPrimitive:
    """Register a NAS primitive globally. Returns the primitive for chaining."""
    PRIMITIVE_REGISTRY[primitive.name] = primitive
    return primitive


# ── mHC primitive ─────────────────────────────────────────────────────────────

def _mhc_param_estimate(d_model: int, kwargs: dict) -> int:
    """
    Parameter count for one MHCLayer:
      H_res_logits:  expansion² 
      H_pre_logits:  expansion × d_model
      H_post_logits: expansion × d_model
      residual_alpha: 1 (if identity_mix)
    """
    n = kwargs.get("expansion", 4)
    base = n * n + 2 * n * d_model
    if kwargs.get("identity_mix", False):
        base += 1
    return base


def _mhc_flop_estimate(d_model: int, seq_len: int, kwargs: dict) -> int:
    """
    FLOPs per forward pass (sequence of length seq_len):
      - H_pre  multiplication: seq_len × expansion × d_model
      - H_res  matmul:         seq_len × expansion × expansion
      - H_post contraction:    seq_len × expansion × d_model
      Roughly: 2 × seq_len × d_model × expansion  (dominant terms)
    """
    n = kwargs.get("expansion", 4)
    return 2 * seq_len * d_model * n * 2   # *2 for mul+add


def _mhc_factory(
    d_model: int,
    expansion: int = 4,
    layer: Optional[nn.Module] = None,
    projection: ProjectionMode = "sinkhorn",
    identity_mix: bool = False,
    sinkhorn_iter: int = 20,
    ns_steps: int = 5,
) -> MHCLayer:
    return MHCLayer(
        d_model=d_model,
        expansion=expansion,
        layer=layer,
        projection=projection,
        identity_mix=identity_mix,
        sinkhorn_iter=sinkhorn_iter,
        ns_steps=ns_steps,
    )


mhc_primitive = register_primitive(NASPrimitive(
    name="mhc",
    description=(
        "Manifold-Constrained Hyper-Connections (DeepSeek, arXiv:2512.24880). "
        "Replaces residual connections with doubly-stochastic H_res and non-negative "
        "H_pre/H_post matrices.  Eliminates gradient vanishing/explosion in 50M–500M param models. "
        "6.7% compute overhead at expansion=4."
    ),
    factory=_mhc_factory,
    default_kwargs={
        "expansion": 4,
        "projection": "sinkhorn",
        "identity_mix": False,
        "sinkhorn_iter": 20,
    },
    param_estimate=_mhc_param_estimate,
    flop_estimate=_mhc_flop_estimate,
))


# ── Vanilla residual primitive (baseline) ─────────────────────────────────────

class _ResidualPassthrough(nn.Module):
    """Standard residual connection: y = x + F(x). Baseline for NAS comparison."""
    def __init__(self, d_model: int, layer: Optional[nn.Module] = None):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        if self.layer is not None:
            return x + self.layer(x)
        return x


residual_primitive = register_primitive(NASPrimitive(
    name="residual",
    description="Standard residual connection: y = x + F(x). NAS baseline.",
    factory=lambda d_model, layer=None, **kw: _ResidualPassthrough(d_model, layer),
    default_kwargs={"layer": None},
    param_estimate=lambda d_model, kw: 0,
    flop_estimate=lambda d_model, seq_len, kw: seq_len * d_model,
))


# ── Convenience helpers ───────────────────────────────────────────────────────

def list_primitives() -> list[str]:
    """Return names of all registered NAS primitives."""
    return list(PRIMITIVE_REGISTRY.keys())


def get_primitive(name: str) -> NASPrimitive:
    """Retrieve a primitive by name, with a helpful error if not found."""
    if name not in PRIMITIVE_REGISTRY:
        available = ", ".join(PRIMITIVE_REGISTRY.keys())
        raise KeyError(f"NAS primitive {name!r} not registered. Available: {available}")
    return PRIMITIVE_REGISTRY[name]
