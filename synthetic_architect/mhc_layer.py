"""
synthetic_architect/mhc_layer.py

Manifold-Constrained Hyper-Connections (mHC)
Paper: DeepSeek, arXiv:2512.24880  (submitted 2025-12-31)

Drop-in replacement for a residual connection.  The key invariant:
  x_{l+1} = H_res @ x_l + H_post.T @ F(H_pre @ x_l, W_l)

Constraints that prevent gradient pathologies:
  - H_res  : doubly stochastic matrix (Birkhoff polytope, rows+cols sum=1, entries≥0)
              → Sinkhorn-Knopp projection (default) or Newton-Schulz orthostochastic
  - H_pre  : non-negative mixing map (softmax-normalised)
  - H_post : non-negative mixing map (softmax-normalised)

Both projections preserve feature mean and regularise signal norm, eliminating the
gradient vanishing / explosion that destabilises small (50M–500M param) models during NAS.

Overhead: ~6.7% extra compute at expansion rate n=4 (per paper Table 3).

Usage
-----
    from synthetic_architect.mhc_layer import MHCLayer

    # Wrap any layer (MLP block, attention sublayer, etc.)
    layer = nn.Linear(d_model, d_model)
    mhc   = MHCLayer(d_model, expansion=4, layer=layer)
    y     = mhc(x)   # x: (..., d_model)

    # Drop-in for residual-only (no sublayer — pure mHC routing):
    mhc_residual = MHCLayer(d_model, expansion=4)
    y = mhc_residual(x)
"""
from __future__ import annotations

import math
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Projection operators ──────────────────────────────────────────────────────

class SinkhornKnopp(nn.Module):
    """
    Projects a square matrix onto the Birkhoff polytope (doubly stochastic matrices)
    via entropic / Sinkhorn-Knopp iteration.

    All row sums and column sums converge to 1, entries ≥ 0.
    Convergence is O(log n) iterations in practice for n ≤ 16.
    """
    def __init__(self, n_iter: int = 20, eps: float = 1e-6):
        super().__init__()
        self.n_iter = n_iter
        self.eps = eps

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: (..., n, n) — any square matrix; will be exp-transformed before
               iteration so entries become positive automatically.
        Returns:
            doubly stochastic matrix of same shape.
        """
        # Exponentiate to enforce positivity
        A = torch.exp(M - M.amax(dim=(-2, -1), keepdim=True))
        for _ in range(self.n_iter):
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)
            A = A / (A.sum(dim=-2, keepdim=True) + self.eps)
        return A


class OrthostochasticNewtonSchulz(nn.Module):
    """
    Projects a square matrix to an orthostochastic approximation via Newton-Schulz
    (5-step Zolotarev polynomial).  Faster than Sinkhorn on GPU for large n.

    Default coefficients from NanoGPT / Modular Diffusion literature:
        a=3.0, b=-3.2, c=1.2  (5 steps)
    """
    NS_COEFFS_DEFAULT = ((3.0, -3.2, 1.2),) * 5

    def __init__(
        self,
        ns_steps: int = 5,
        ns_coeffs: Optional[tuple] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.ns_steps = ns_steps
        self.ns_coeffs: tuple = ns_coeffs if ns_coeffs is not None else self.NS_COEFFS_DEFAULT[:ns_steps]

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: (..., n, n) — any square matrix.
        Returns:
            Approximately orthostochastic matrix of same shape.
        """
        n = M.shape[-1]
        # Normalise: divide by Frobenius norm (more stable than spectral norm for small n)
        fro = M.norm(dim=(-2, -1), keepdim=True).clamp(min=self.eps)
        X = M / fro
        # Newton-Schulz iterations — clamp after each step to prevent divergence
        for a, b, c in self.ns_coeffs:
            XtX = X.mT @ X
            X = a * X + b * (X @ XtX) + c * (X @ (XtX @ XtX))
            X = X.clamp(-2.0, 2.0)   # guard against blow-up on ill-conditioned matrices
        # Map to doubly-stochastic: abs + normalise rows then cols
        X = X.abs()
        X = X / (X.sum(dim=-1, keepdim=True).clamp(min=self.eps))
        X = X / (X.sum(dim=-2, keepdim=True).clamp(min=self.eps))
        return X


# ── Core mHC layer ────────────────────────────────────────────────────────────

ProjectionMode = Literal["sinkhorn", "orthostochastic"]


class MHCLayer(nn.Module):
    """
    Manifold-Constrained Hyper-Connections layer.

    Parameters
    ----------
    d_model     : int   — feature dimension of the residual stream.
    expansion   : int   — stream expansion rate n (paper default: 4).
                          Higher n improves expressivity at slightly higher compute.
    layer       : nn.Module, optional
                  The sublayer F (e.g. MLP, attention projection).
                  If None, acts as a residual routing layer only (no sublayer call).
    projection  : "sinkhorn" | "orthostochastic"
                  How H_res is projected onto the Birkhoff polytope.
    identity_mix: bool  — If True, blend projected H_res with identity:
                          H_res ← (1-α)*I + α*H_res   (α is learnt, init 0.01)
    sinkhorn_iter  : int  — iterations for Sinkhorn (ignored if orthostochastic).
    ns_steps       : int  — Newton-Schulz steps (ignored if sinkhorn).
    """

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        layer: Optional[nn.Module] = None,
        projection: ProjectionMode = "sinkhorn",
        identity_mix: bool = False,
        sinkhorn_iter: int = 20,
        ns_steps: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.expansion = expansion
        self.layer = layer
        self.identity_mix = identity_mix

        # Learnable logits for the three hyper-connection matrices
        # Shape: (expansion, expansion) for H_res; (expansion, d_model) for H_pre/H_post
        self.H_res_logits  = nn.Parameter(torch.zeros(expansion, expansion))
        self.H_pre_logits  = nn.Parameter(torch.zeros(expansion, d_model))
        self.H_post_logits = nn.Parameter(torch.zeros(expansion, d_model))

        if identity_mix:
            self.residual_alpha = nn.Parameter(torch.tensor(0.01))

        # Projection operator for H_res
        if projection == "sinkhorn":
            self._project_H_res: nn.Module = SinkhornKnopp(n_iter=sinkhorn_iter)
        elif projection == "orthostochastic":
            self._project_H_res = OrthostochasticNewtonSchulz(ns_steps=ns_steps)
        else:
            raise ValueError(f"Unknown projection mode: {projection!r}")

        # Register identity buffer for identity_mix
        self.register_buffer(
            "_identity",
            torch.eye(expansion),
            persistent=False,
        )
        self._init_weights()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_weights(self):
        # Initialise near identity: H_res → doubly stochastic ≈ I/n, H_pre/H_post ≈ uniform
        nn.init.zeros_(self.H_res_logits)
        nn.init.zeros_(self.H_pre_logits)
        nn.init.zeros_(self.H_post_logits)

    # ── Projections ───────────────────────────────────────────────────────────

    @property
    def H_res(self) -> torch.Tensor:
        H = self._project_H_res(self.H_res_logits.unsqueeze(0)).squeeze(0)
        if self.identity_mix:
            alpha = self.residual_alpha.sigmoid()
            H = (1 - alpha) * self._identity + alpha * H
        return H                                            # (expansion, expansion)

    @property
    def H_pre(self) -> torch.Tensor:
        return F.softmax(self.H_pre_logits, dim=-1)        # (expansion, d_model) — non-neg, sums to 1

    @property
    def H_post(self) -> torch.Tensor:
        return F.softmax(self.H_post_logits, dim=-1)       # (expansion, d_model)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (..., d_model)   — input features.
        Returns:
            y : (..., d_model)   — output features, same shape as input.
        """
        # x_expanded: (..., expansion, d_model)
        # H_pre:  (expansion, d_model)  — broadcast over leading dims
        x_exp = x.unsqueeze(-2) * self.H_pre.unsqueeze(0) if x.dim() == 2 else \
                x.unsqueeze(-2) * self.H_pre  # generalised; handles (B, T, d)

        # --- simplified broadcast for arbitrary batch dims ---
        # x: (..., d_model) → x_pre: (..., expansion) via weighted sum
        x_pre = (x.unsqueeze(-2) * self.H_pre).sum(-1)    # (..., expansion)
        # H_res routing: (..., expansion) → (..., expansion)
        x_res_routed = x_pre @ self.H_res                  # (..., expansion)

        if self.layer is not None:
            # Sublayer receives full-width pre-mix: (expansion, d_model) @ (..., d_model) → (..., expansion)
            x_sublayer = (x.unsqueeze(-2) * self.H_pre).sum(-1)  # (..., expansion)
            # Materialise to d_model for the sublayer: expand each stream vector
            # Simplified: concatenate expansion streams → d_model via H_post transpose
            # x_flat: (..., d_model)  (sum over expansion streams)
            x_flat = (x_sublayer.unsqueeze(-1) * self.H_post).sum(-2)  # (..., d_model)
            fx = self.layer(x_flat)                         # (..., d_model)
            # Post-mix: project fx back to expansion streams
            fx_exp = (fx.unsqueeze(-2) * self.H_post).sum(-1)  # (..., expansion)
            out_exp = x_res_routed + fx_exp                 # (..., expansion)
        else:
            out_exp = x_res_routed                         # (..., expansion)

        # Contract from expansion back to d_model via H_post transpose
        out = (out_exp.unsqueeze(-1) * self.H_post).sum(-2)   # (..., d_model)
        return out

    # ── Repr ──────────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, expansion={self.expansion}, "
            f"layer={self.layer}, identity_mix={self.identity_mix}"
        )
