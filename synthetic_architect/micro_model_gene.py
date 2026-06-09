"""
synthetic_architect/micro_model_gene.py

MicroModelGene — Genetic encoding for 50M–500M parameter micro-model architectures
in Sovereign Core's Synthetic Architect (SAS-1, KAN-88).

Each "gene" is a structured configuration vector representing a complete model
architecture.  The NAS controller (KAIROS evolution loop) operates on genes;
each gene is decoded into a buildable PyTorch model.

mHC is a first-class gene attribute: any layer can independently choose between
standard residual connections and mHC, making it directly NAS-searchable.

Design rationale
----------------
The gene format must be:
  1. Serialisable (JSON / msgpack) for archival in Aegis-Vault (KAN-90)
  2. Diff-able for evolutionary mutation operators
  3. Decodable into a concrete nn.Module without external state
  4. Budget-aware: param/flop estimates before materialisation

Gene structure (per-layer flags)
---------------------------------
  residual_type:   "residual" | "mhc"
  mhc_expansion:   int  (2, 4, 8)  — only used when residual_type == "mhc"
  mhc_projection:  "sinkhorn" | "orthostochastic"
  mhc_identity_mix: bool
  d_model:         int
  n_layers:        int
  n_heads:         int  (for attention; None if MLP-only model)
  ffn_mult:        float  (FFN hidden = ffn_mult * d_model)
  param_budget:    int  (target parameter count, used for pruning)
  seq_len:         int  (max sequence length, used for flop estimation)
"""
from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn

from .mhc_layer import MHCLayer
from .nas_primitives import PRIMITIVE_REGISTRY, get_primitive


ResidualType = Literal["residual", "mhc"]


@dataclasses.dataclass
class LayerGene:
    """Gene for a single transformer/MLP layer."""
    residual_type: ResidualType = "residual"
    mhc_expansion: int = 4
    mhc_projection: str = "sinkhorn"
    mhc_identity_mix: bool = False
    has_attention: bool = True
    n_heads: int = 8

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerGene":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})

    def mutate(
        self,
        flip_residual_prob: float = 0.1,
        expand_prob: float = 0.1,
        projection_flip_prob: float = 0.05,
    ) -> "LayerGene":
        """Return a mutated copy of this gene (used by evolutionary controller)."""
        import random
        g = copy.deepcopy(self)
        if random.random() < flip_residual_prob:
            g.residual_type = "mhc" if g.residual_type == "residual" else "residual"
        if g.residual_type == "mhc":
            if random.random() < expand_prob:
                g.mhc_expansion = random.choice([2, 4, 8])
            if random.random() < projection_flip_prob:
                g.mhc_projection = "orthostochastic" if g.mhc_projection == "sinkhorn" else "sinkhorn"
        return g


@dataclasses.dataclass
class MicroModelGene:
    """
    Full architectural gene for a micro-model (50M–500M parameters).

    Attributes
    ----------
    d_model      : Residual stream dimension.
    n_layers     : Number of transformer/MLP layers.
    ffn_mult     : Hidden dim multiplier for FFN (hidden = ffn_mult * d_model).
    param_budget : Soft upper bound on parameter count (used for validity check).
    seq_len      : Max sequence length for flop estimation.
    layer_genes  : Per-layer residual strategy genes (length == n_layers).
                   Populated automatically on __post_init__ if empty.
    """
    d_model: int = 256
    n_layers: int = 12
    ffn_mult: float = 4.0
    param_budget: int = 100_000_000   # 100M params default
    seq_len: int = 2048
    layer_genes: List[LayerGene] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if not self.layer_genes:
            self.layer_genes = [LayerGene() for _ in range(self.n_layers)]
        elif len(self.layer_genes) != self.n_layers:
            raise ValueError(
                f"layer_genes length ({len(self.layer_genes)}) must match n_layers ({self.n_layers})"
            )

    # ── Estimation ────────────────────────────────────────────────────────────

    def estimate_params(self) -> int:
        """Approximate parameter count (embedding not included)."""
        ffn_hidden = int(self.ffn_mult * self.d_model)
        total = 0
        for lg in self.layer_genes:
            # Attention Q/K/V/O (simplified, assumes d_model heads)
            if lg.has_attention:
                total += 4 * self.d_model * self.d_model
            # FFN (2 weight matrices)
            total += self.d_model * ffn_hidden + ffn_hidden * self.d_model
            # Residual primitive overhead
            prim = get_primitive(lg.residual_type)
            total += prim.estimate_params(
                self.d_model,
                expansion=lg.mhc_expansion,
                projection=lg.mhc_projection,
                identity_mix=lg.mhc_identity_mix,
            )
        return total

    def estimate_flops(self) -> int:
        """Approximate FLOPs per forward pass."""
        ffn_hidden = int(self.ffn_mult * self.d_model)
        total = 0
        for lg in self.layer_genes:
            if lg.has_attention:
                # QKV + attention + O: ~4 * 2 * seq * d²
                total += 8 * self.seq_len * self.d_model * self.d_model
                # attention scores: 2 * seq² * d
                total += 2 * self.seq_len * self.seq_len * self.d_model
            # FFN
            total += 2 * self.seq_len * self.d_model * ffn_hidden
            # mHC overhead
            prim = get_primitive(lg.residual_type)
            total += prim.estimate_flops(
                self.d_model, self.seq_len,
                expansion=lg.mhc_expansion,
            )
        return total

    def is_within_budget(self) -> bool:
        return self.estimate_params() <= self.param_budget

    def mhc_fraction(self) -> float:
        """Fraction of layers using mHC (vs standard residual)."""
        if not self.layer_genes:
            return 0.0
        return sum(1 for lg in self.layer_genes if lg.residual_type == "mhc") / len(self.layer_genes)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MicroModelGene":
        layer_genes = [LayerGene.from_dict(lg) for lg in d.pop("layer_genes", [])]
        gene = cls(**d, layer_genes=layer_genes)
        return gene

    @classmethod
    def from_json(cls, s: str) -> "MicroModelGene":
        return cls.from_dict(json.loads(s))

    # ── Mutation ──────────────────────────────────────────────────────────────

    def mutate(
        self,
        flip_residual_prob: float = 0.1,
        d_model_step: int = 64,
        d_model_prob: float = 0.05,
        n_layers_prob: float = 0.05,
    ) -> "MicroModelGene":
        """
        Return a mutated gene copy.  Called by the KAIROS evolutionary controller.
        Never modifies in-place.
        """
        import random
        g = copy.deepcopy(self)
        # Mutate d_model
        if random.random() < d_model_prob:
            delta = random.choice([-d_model_step, d_model_step])
            g.d_model = max(64, g.d_model + delta)
        # Mutate n_layers
        if random.random() < n_layers_prob:
            delta = random.choice([-2, -1, 1, 2])
            g.n_layers = max(2, g.n_layers + delta)
            # Adjust layer_genes list
            while len(g.layer_genes) < g.n_layers:
                g.layer_genes.append(LayerGene())
            g.layer_genes = g.layer_genes[:g.n_layers]
        # Mutate per-layer genes
        g.layer_genes = [lg.mutate(flip_residual_prob=flip_residual_prob) for lg in g.layer_genes]
        return g

    # ── Materialisation ───────────────────────────────────────────────────────

    def build_residual_modules(self) -> List[nn.Module]:
        """
        Build only the residual-connection modules for each layer.
        Used when plugging mHC into an existing model backbone.
        """
        modules = []
        for lg in self.layer_genes:
            prim = get_primitive(lg.residual_type)
            mod = prim.build(
                d_model=self.d_model,
                expansion=lg.mhc_expansion,
                projection=lg.mhc_projection,
                identity_mix=lg.mhc_identity_mix,
            )
            modules.append(mod)
        return modules


@dataclasses.dataclass
class GeneSearchSpace:
    """
    Defines the discrete search space for the NAS controller.

    The KAIROS evolution loop samples from these lists when generating new genes.
    """
    d_model_choices:     List[int]   = dataclasses.field(default_factory=lambda: [128, 256, 512])
    n_layers_choices:    List[int]   = dataclasses.field(default_factory=lambda: [6, 12, 18, 24])
    ffn_mult_choices:    List[float] = dataclasses.field(default_factory=lambda: [2.0, 4.0, 8.0])
    residual_types:      List[str]   = dataclasses.field(default_factory=lambda: ["residual", "mhc"])
    mhc_expansion_choices: List[int] = dataclasses.field(default_factory=lambda: [2, 4, 8])
    mhc_projection_choices: List[str] = dataclasses.field(default_factory=lambda: ["sinkhorn", "orthostochastic"])
    param_budgets:       List[int]   = dataclasses.field(
        default_factory=lambda: [50_000_000, 100_000_000, 250_000_000, 500_000_000]
    )

    def sample(self, seed: Optional[int] = None) -> MicroModelGene:
        """Uniformly sample a random gene from the search space."""
        import random
        rng = random.Random(seed)
        d_model = rng.choice(self.d_model_choices)
        n_layers = rng.choice(self.n_layers_choices)
        ffn_mult = rng.choice(self.ffn_mult_choices)
        budget   = rng.choice(self.param_budgets)
        layer_genes = []
        for _ in range(n_layers):
            r_type = rng.choice(self.residual_types)
            layer_genes.append(LayerGene(
                residual_type=r_type,
                mhc_expansion=rng.choice(self.mhc_expansion_choices),
                mhc_projection=rng.choice(self.mhc_projection_choices),
                mhc_identity_mix=rng.choice([True, False]),
            ))
        return MicroModelGene(
            d_model=d_model,
            n_layers=n_layers,
            ffn_mult=ffn_mult,
            param_budget=budget,
            layer_genes=layer_genes,
        )
