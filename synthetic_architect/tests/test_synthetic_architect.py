"""
tests/test_synthetic_architect.py

Unit tests for Sovereign Core Synthetic Architect — RES-04 mHC integration.
Tests run without GPU; all shapes are tiny to keep CI fast.
"""
import json
import math
import pytest
import torch
import torch.nn as nn


# ── We import from the synthetic_architect package ────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from synthetic_architect.mhc_layer import (
    MHCLayer, SinkhornKnopp, OrthostochasticNewtonSchulz
)
from synthetic_architect.nas_primitives import (
    PRIMITIVE_REGISTRY, get_primitive, list_primitives, mhc_primitive, residual_primitive
)
from synthetic_architect.micro_model_gene import (
    LayerGene, MicroModelGene, GeneSearchSpace
)


# ===========================================================================
# SinkhornKnopp
# ===========================================================================
class TestSinkhornKnopp:
    def test_output_shape_preserved(self):
        sk = SinkhornKnopp(n_iter=10)
        M = torch.randn(4, 4)
        out = sk(M.unsqueeze(0)).squeeze(0)
        assert out.shape == M.shape

    def test_rows_sum_to_one(self):
        sk = SinkhornKnopp(n_iter=30)
        M = torch.randn(1, 4, 4)
        out = sk(M)
        row_sums = out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    def test_cols_sum_to_one(self):
        sk = SinkhornKnopp(n_iter=30)
        M = torch.randn(1, 4, 4)
        out = sk(M)
        col_sums = out.sum(dim=-2)
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)

    def test_entries_non_negative(self):
        sk = SinkhornKnopp(n_iter=20)
        M = torch.randn(1, 4, 4)
        out = sk(M)
        assert (out >= -1e-6).all()

    def test_batch_dims(self):
        sk = SinkhornKnopp(n_iter=10)
        M = torch.randn(3, 4, 4)
        out = sk(M)
        assert out.shape == (3, 4, 4)


# ===========================================================================
# OrthostochasticNewtonSchulz
# ===========================================================================
class TestOrthostochasticNewtonSchulz:
    def test_output_shape_preserved(self):
        ns = OrthostochasticNewtonSchulz(ns_steps=5)
        M = torch.randn(1, 4, 4)
        out = ns(M)
        assert out.shape == M.shape

    def test_entries_non_negative(self):
        ns = OrthostochasticNewtonSchulz(ns_steps=5)
        M = torch.randn(1, 4, 4)
        out = ns(M)
        assert (out >= -1e-6).all()

    def test_rows_approximately_sum_to_one(self):
        ns = OrthostochasticNewtonSchulz(ns_steps=5)
        M = torch.randn(1, 4, 4)
        out = ns(M)
        row_sums = out.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.1)


# ===========================================================================
# MHCLayer — output shape and basic properties
# ===========================================================================
class TestMHCLayerShape:
    @pytest.mark.parametrize("d_model,expansion", [(32, 2), (64, 4), (128, 8)])
    def test_output_shape_no_sublayer(self, d_model, expansion):
        mhc = MHCLayer(d_model=d_model, expansion=expansion)
        x = torch.randn(2, 8, d_model)
        y = mhc(x)
        assert y.shape == x.shape

    @pytest.mark.parametrize("d_model,expansion", [(32, 2), (64, 4)])
    def test_output_shape_with_linear_sublayer(self, d_model, expansion):
        layer = nn.Linear(d_model, d_model, bias=False)
        mhc = MHCLayer(d_model=d_model, expansion=expansion, layer=layer)
        x = torch.randn(4, 16, d_model)
        y = mhc(x)
        assert y.shape == x.shape

    def test_2d_input(self):
        mhc = MHCLayer(d_model=64, expansion=4)
        x = torch.randn(8, 64)
        y = mhc(x)
        assert y.shape == x.shape

    def test_batch_size_one(self):
        mhc = MHCLayer(d_model=32, expansion=2)
        x = torch.randn(1, 4, 32)
        y = mhc(x)
        assert y.shape == x.shape


class TestMHCLayerGradients:
    def test_backward_pass(self):
        mhc = MHCLayer(d_model=32, expansion=4)
        x = torch.randn(2, 8, 32, requires_grad=True)
        y = mhc(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None

    def test_gradients_not_nan(self):
        mhc = MHCLayer(d_model=64, expansion=4)
        x = torch.randn(4, 16, 64, requires_grad=True)
        y = mhc(x)
        y.sum().backward()
        for name, p in mhc.named_parameters():
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"

    def test_gradients_not_inf(self):
        mhc = MHCLayer(d_model=64, expansion=4)
        x = torch.randn(4, 16, 64)
        y = mhc(x)
        y.sum().backward()
        for name, p in mhc.named_parameters():
            assert not torch.isinf(p.grad).any(), f"Inf grad in {name}"


class TestMHCLayerProjections:
    def test_sinkhorn_projection(self):
        mhc = MHCLayer(d_model=32, expansion=4, projection="sinkhorn")
        x = torch.randn(2, 4, 32)
        y = mhc(x)
        assert y.shape == x.shape

    def test_orthostochastic_projection(self):
        mhc = MHCLayer(d_model=32, expansion=4, projection="orthostochastic")
        x = torch.randn(2, 4, 32)
        y = mhc(x)
        assert y.shape == x.shape

    def test_invalid_projection_raises(self):
        with pytest.raises(ValueError, match="Unknown projection"):
            MHCLayer(d_model=32, expansion=4, projection="bad_proj")


class TestMHCLayerIdentityMix:
    def test_identity_mix_shape(self):
        mhc = MHCLayer(d_model=32, expansion=4, identity_mix=True)
        x = torch.randn(2, 8, 32)
        y = mhc(x)
        assert y.shape == x.shape

    def test_residual_alpha_is_parameter(self):
        mhc = MHCLayer(d_model=32, expansion=4, identity_mix=True)
        param_names = [n for n, _ in mhc.named_parameters()]
        assert "residual_alpha" in param_names

    def test_no_identity_mix_has_no_alpha(self):
        mhc = MHCLayer(d_model=32, expansion=4, identity_mix=False)
        param_names = [n for n, _ in mhc.named_parameters()]
        assert "residual_alpha" not in param_names


# ===========================================================================
# NAS Primitive Registry
# ===========================================================================
class TestNASPrimitiveRegistry:
    def test_mhc_registered(self):
        assert "mhc" in PRIMITIVE_REGISTRY

    def test_residual_registered(self):
        assert "residual" in PRIMITIVE_REGISTRY

    def test_list_primitives(self):
        prims = list_primitives()
        assert "mhc" in prims
        assert "residual" in prims

    def test_get_existing_primitive(self):
        p = get_primitive("mhc")
        assert p.name == "mhc"

    def test_get_missing_primitive_raises(self):
        with pytest.raises(KeyError, match="not registered"):
            get_primitive("does_not_exist")

    def test_mhc_description_mentions_deepseek(self):
        assert "DeepSeek" in mhc_primitive.description

    def test_mhc_description_mentions_arxiv(self):
        assert "2512.24880" in mhc_primitive.description

    def test_residual_description(self):
        assert "residual" in residual_primitive.description.lower()


class TestNASPrimitiveBuild:
    def test_mhc_build_returns_mhc_layer(self):
        prim = get_primitive("mhc")
        mod = prim.build(d_model=64)
        assert isinstance(mod, MHCLayer)

    def test_mhc_build_with_expansion_override(self):
        prim = get_primitive("mhc")
        mod = prim.build(d_model=64, expansion=8)
        assert mod.expansion == 8

    def test_residual_build_returns_module(self):
        prim = get_primitive("residual")
        mod = prim.build(d_model=64)
        assert isinstance(mod, nn.Module)


class TestNASPrimitiveEstimates:
    def test_mhc_param_estimate_positive(self):
        prim = get_primitive("mhc")
        n = prim.estimate_params(d_model=256)
        assert n > 0

    def test_mhc_param_estimate_scales_with_expansion(self):
        prim = get_primitive("mhc")
        n4 = prim.estimate_params(d_model=128, expansion=4)
        n8 = prim.estimate_params(d_model=128, expansion=8)
        assert n8 > n4

    def test_mhc_flop_estimate_positive(self):
        prim = get_primitive("mhc")
        f = prim.estimate_flops(d_model=256, seq_len=512)
        assert f > 0

    def test_residual_param_estimate_zero(self):
        prim = get_primitive("residual")
        n = prim.estimate_params(d_model=256)
        assert n == 0


# ===========================================================================
# LayerGene
# ===========================================================================
class TestLayerGene:
    def test_default_residual_type(self):
        g = LayerGene()
        assert g.residual_type == "residual"

    def test_serialisation_roundtrip(self):
        g = LayerGene(residual_type="mhc", mhc_expansion=8, mhc_projection="orthostochastic")
        g2 = LayerGene.from_dict(g.as_dict())
        assert g == g2

    def test_mutation_preserves_type(self):
        import random
        random.seed(42)
        g = LayerGene(residual_type="mhc")
        for _ in range(20):
            gm = g.mutate(flip_residual_prob=0)
            assert gm.residual_type == "mhc"

    def test_mutation_can_flip_residual(self):
        import random
        random.seed(0)
        g = LayerGene(residual_type="residual")
        results = {g.mutate(flip_residual_prob=1.0).residual_type for _ in range(10)}
        assert "mhc" in results


# ===========================================================================
# MicroModelGene
# ===========================================================================
class TestMicroModelGene:
    def _small_gene(self) -> MicroModelGene:
        return MicroModelGene(d_model=64, n_layers=4, ffn_mult=2.0, param_budget=10_000_000)

    def test_layer_genes_auto_populated(self):
        g = self._small_gene()
        assert len(g.layer_genes) == 4

    def test_layer_genes_wrong_length_raises(self):
        with pytest.raises(ValueError, match="layer_genes length"):
            MicroModelGene(d_model=64, n_layers=4, layer_genes=[LayerGene()] * 3)

    def test_estimate_params_positive(self):
        g = self._small_gene()
        assert g.estimate_params() > 0

    def test_estimate_flops_positive(self):
        g = self._small_gene()
        assert g.estimate_flops() > 0

    def test_is_within_budget_default_gene(self):
        g = MicroModelGene(d_model=128, n_layers=4, param_budget=100_000_000)
        # 100M budget is very generous for a 128-dim, 4-layer model
        assert g.is_within_budget()

    def test_out_of_budget_detected(self):
        g = MicroModelGene(d_model=512, n_layers=24, param_budget=1)
        assert not g.is_within_budget()

    def test_mhc_fraction_all_residual(self):
        g = MicroModelGene(
            d_model=64, n_layers=4,
            layer_genes=[LayerGene(residual_type="residual")] * 4
        )
        assert g.mhc_fraction() == 0.0

    def test_mhc_fraction_all_mhc(self):
        g = MicroModelGene(
            d_model=64, n_layers=4,
            layer_genes=[LayerGene(residual_type="mhc")] * 4
        )
        assert g.mhc_fraction() == 1.0

    def test_mhc_fraction_half(self):
        genes = [LayerGene(residual_type="mhc")] * 2 + [LayerGene(residual_type="residual")] * 2
        g = MicroModelGene(d_model=64, n_layers=4, layer_genes=genes)
        assert g.mhc_fraction() == 0.5

    def test_json_roundtrip(self):
        g = self._small_gene()
        g2 = MicroModelGene.from_json(g.to_json())
        assert g.d_model == g2.d_model
        assert g.n_layers == g2.n_layers
        assert len(g2.layer_genes) == len(g.layer_genes)

    def test_mutation_returns_new_object(self):
        g = self._small_gene()
        gm = g.mutate()
        assert gm is not g

    def test_mutation_preserves_n_layers_within_bounds(self):
        import random
        random.seed(7)
        g = MicroModelGene(d_model=64, n_layers=6, param_budget=50_000_000)
        for _ in range(20):
            gm = g.mutate(n_layers_prob=1.0)
            assert gm.n_layers >= 2
            assert len(gm.layer_genes) == gm.n_layers

    def test_build_residual_modules_length(self):
        g = MicroModelGene(
            d_model=64, n_layers=3,
            layer_genes=[
                LayerGene(residual_type="mhc"),
                LayerGene(residual_type="residual"),
                LayerGene(residual_type="mhc"),
            ]
        )
        mods = g.build_residual_modules()
        assert len(mods) == 3

    def test_build_residual_modules_types(self):
        g = MicroModelGene(
            d_model=64, n_layers=2,
            layer_genes=[
                LayerGene(residual_type="mhc"),
                LayerGene(residual_type="residual"),
            ]
        )
        mods = g.build_residual_modules()
        assert isinstance(mods[0], MHCLayer)
        assert isinstance(mods[1], nn.Module)

    def test_mhc_modules_forward_pass(self):
        """End-to-end: gene → modules → forward pass."""
        d_model = 64
        g = MicroModelGene(
            d_model=d_model, n_layers=2,
            layer_genes=[LayerGene(residual_type="mhc")] * 2
        )
        mods = g.build_residual_modules()
        x = torch.randn(2, 8, d_model)
        for mod in mods:
            x = mod(x)
        assert x.shape == (2, 8, d_model)


# ===========================================================================
# GeneSearchSpace
# ===========================================================================
class TestGeneSearchSpace:
    def test_sample_returns_gene(self):
        space = GeneSearchSpace()
        g = space.sample(seed=42)
        assert isinstance(g, MicroModelGene)

    def test_sample_is_deterministic_with_seed(self):
        space = GeneSearchSpace()
        g1 = space.sample(seed=123)
        g2 = space.sample(seed=123)
        assert g1.d_model == g2.d_model
        assert g1.n_layers == g2.n_layers

    def test_sample_respects_d_model_choices(self):
        choices = [64, 128]
        space = GeneSearchSpace(d_model_choices=choices)
        for _ in range(20):
            g = space.sample()
            assert g.d_model in choices

    def test_sample_respects_n_layers_choices(self):
        choices = [4, 8]
        space = GeneSearchSpace(n_layers_choices=choices)
        for _ in range(20):
            g = space.sample()
            assert g.n_layers in choices

    def test_sample_includes_mhc_layers(self):
        """With mhc in residual_types, some sampled genes should have mHC layers."""
        space = GeneSearchSpace(residual_types=["mhc"])
        g = space.sample(seed=1)
        assert all(lg.residual_type == "mhc" for lg in g.layer_genes)

    def test_sample_all_residual_space(self):
        space = GeneSearchSpace(residual_types=["residual"])
        g = space.sample(seed=2)
        assert all(lg.residual_type == "residual" for lg in g.layer_genes)
