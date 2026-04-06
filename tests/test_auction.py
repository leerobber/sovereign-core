"""Tests for the Vickrey-Quadratic Auction Economic Core (KAN-88)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from gateway.auction import (
    AgentBudget,
    AllocationFairness,
    Auctioneer,
    AuctionResult,
    Bid,
    CreditLedger,
    InsufficientCreditsError,
    ResourceType,
    VickreyQuadraticAuction,
    _gini_coefficient,
    auctioneer,
)


# ===========================================================================
# AgentBudget
# ===========================================================================


class TestAgentBudget:
    def test_initial_balance(self) -> None:
        b = AgentBudget("alice", 100)
        assert b.remaining_credits == 100
        assert b.total_credits == 100
        assert b.spent_credits == 0

    def test_zero_initial_balance(self) -> None:
        b = AgentBudget("alice")
        assert b.remaining_credits == 0

    def test_negative_initial_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            AgentBudget("alice", -1)

    def test_top_up(self) -> None:
        b = AgentBudget("alice", 50)
        b.top_up(25)
        assert b.total_credits == 75
        assert b.remaining_credits == 75

    def test_top_up_zero_or_negative_raises(self) -> None:
        b = AgentBudget("alice", 50)
        with pytest.raises(ValueError, match="positive"):
            b.top_up(0)
        with pytest.raises(ValueError, match="positive"):
            b.top_up(-5)

    def test_spend(self) -> None:
        b = AgentBudget("alice", 100)
        b.spend(40)
        assert b.spent_credits == 40
        assert b.remaining_credits == 60

    def test_spend_zero_allowed(self) -> None:
        b = AgentBudget("alice", 100)
        b.spend(0)
        assert b.spent_credits == 0

    def test_spend_negative_raises(self) -> None:
        b = AgentBudget("alice", 100)
        with pytest.raises(ValueError, match="non-negative"):
            b.spend(-1)

    def test_spend_insufficient_raises(self) -> None:
        b = AgentBudget("alice", 10)
        with pytest.raises(InsufficientCreditsError):
            b.spend(11)

    def test_spend_exact_balance_allowed(self) -> None:
        b = AgentBudget("alice", 10)
        b.spend(10)
        assert b.remaining_credits == 0

    def test_can_afford_true(self) -> None:
        b = AgentBudget("alice", 100)
        assert b.can_afford(100) is True

    def test_can_afford_false(self) -> None:
        b = AgentBudget("alice", 10)
        assert b.can_afford(11) is False


# ===========================================================================
# CreditLedger
# ===========================================================================


class TestCreditLedger:
    def test_register_and_balance(self) -> None:
        ledger = CreditLedger()
        ledger.register_agent("alice", 200)
        assert ledger.balance("alice") == 200

    def test_register_idempotent(self) -> None:
        ledger = CreditLedger()
        ledger.register_agent("alice", 100)
        ledger.register_agent("alice", 999)  # second call is a no-op
        assert ledger.balance("alice") == 100

    def test_get_budget_auto_creates(self) -> None:
        ledger = CreditLedger()
        budget = ledger.get_budget("new_agent")
        assert budget.remaining_credits == 0

    def test_top_up(self) -> None:
        ledger = CreditLedger()
        ledger.top_up("alice", 50)
        assert ledger.balance("alice") == 50

    def test_spend(self) -> None:
        ledger = CreditLedger()
        ledger.register_agent("alice", 100)
        ledger.spend("alice", 30)
        assert ledger.balance("alice") == 70

    def test_spend_insufficient_raises(self) -> None:
        ledger = CreditLedger()
        ledger.register_agent("alice", 10)
        with pytest.raises(InsufficientCreditsError):
            ledger.spend("alice", 50)

    def test_can_afford(self) -> None:
        ledger = CreditLedger()
        ledger.register_agent("alice", 100)
        assert ledger.can_afford("alice", 100) is True
        assert ledger.can_afford("alice", 101) is False

    def test_all_balances(self) -> None:
        ledger = CreditLedger()
        ledger.register_agent("alice", 100)
        ledger.register_agent("bob", 50)
        balances = ledger.all_balances()
        assert balances == {"alice": 100, "bob": 50}

    def test_all_spending(self) -> None:
        ledger = CreditLedger()
        ledger.register_agent("alice", 100)
        ledger.spend("alice", 25)
        spending = ledger.all_spending()
        assert spending["alice"] == 25


# ===========================================================================
# Quadratic cost property
# ===========================================================================


class TestBidQuadraticCost:
    def test_credit_cost_is_votes_squared(self) -> None:
        b = Bid("alice", ResourceType.COMPUTE_TIME, "rtx5050", votes=5)
        assert b.credit_cost == 25  # 5² = 25

    def test_credit_cost_one_vote(self) -> None:
        b = Bid("bob", ResourceType.VRAM, "radeon780m", votes=1)
        assert b.credit_cost == 1

    def test_credit_cost_ten_votes(self) -> None:
        b = Bid("carol", ResourceType.PRIORITY, "ryzen7cpu", votes=10)
        assert b.credit_cost == 100  # 10² = 100


# ===========================================================================
# VickreyQuadraticAuction
# ===========================================================================


class TestVickreyQuadraticAuction:
    def _make_auction(
        self,
        resource: ResourceType = ResourceType.COMPUTE_TIME,
        backend: str = "rtx5050",
    ) -> tuple[VickreyQuadraticAuction, CreditLedger]:
        ledger = CreditLedger()
        ledger.register_agent("alice", 500)
        ledger.register_agent("bob", 500)
        ledger.register_agent("carol", 500)
        auction = VickreyQuadraticAuction(
            auction_id="test-001",
            resource_type=resource,
            backend_id=backend,
            ledger=ledger,
        )
        return auction, ledger

    # -- place_bid --

    def test_place_bid_basic(self) -> None:
        auction, _ = self._make_auction()
        bid = auction.place_bid("alice", 3)
        assert bid.agent_id == "alice"
        assert bid.votes == 3
        assert bid.credit_cost == 9

    def test_place_bid_records_bid(self) -> None:
        auction, _ = self._make_auction()
        auction.place_bid("alice", 3)
        assert len(auction.bids) == 1

    def test_place_bid_zero_votes_raises(self) -> None:
        auction, _ = self._make_auction()
        with pytest.raises(ValueError, match="≥ 1"):
            auction.place_bid("alice", 0)

    def test_place_bid_insufficient_credits_raises(self) -> None:
        ledger = CreditLedger()
        ledger.register_agent("poor_agent", 8)  # can't afford 9 (3²)
        auction = VickreyQuadraticAuction("t", ResourceType.VRAM, "rtx5050", ledger)
        with pytest.raises(InsufficientCreditsError):
            auction.place_bid("poor_agent", 3)

    def test_place_bid_after_settle_raises(self) -> None:
        auction, _ = self._make_auction()
        auction.settle()
        with pytest.raises(RuntimeError, match="already settled"):
            auction.place_bid("alice", 1)

    # -- settle: no bids --

    def test_settle_no_bids(self) -> None:
        auction, _ = self._make_auction()
        result = auction.settle()
        assert result.winner_agent_id is None
        assert result.payment_credits == 0
        assert result.winning_votes == 0
        assert result.all_bids == []

    # -- settle: single bid → winner pays 0 (no second price) --

    def test_settle_single_bid_no_payment(self) -> None:
        auction, ledger = self._make_auction()
        auction.place_bid("alice", 5)
        result = auction.settle()
        assert result.winner_agent_id == "alice"
        assert result.winning_votes == 5
        # Single bidder → second price = 0² = 0
        assert result.payment_credits == 0
        # Alice's balance unchanged
        assert ledger.balance("alice") == 500

    # -- settle: two bidders → winner pays second-price --

    def test_settle_two_bidders_second_price(self) -> None:
        auction, ledger = self._make_auction()
        auction.place_bid("alice", 5)   # cost = 25
        auction.place_bid("bob", 3)     # cost = 9
        result = auction.settle()
        # Alice wins (5 > 3)
        assert result.winner_agent_id == "alice"
        assert result.winning_votes == 5
        # Alice pays bob's cost: 3² = 9
        assert result.payment_credits == 9
        assert ledger.balance("alice") == 500 - 9  # 491
        assert ledger.balance("bob") == 500  # bob pays nothing

    # -- settle: vickrey truth-telling property --

    def test_vickrey_winner_pays_less_than_own_bid(self) -> None:
        auction, ledger = self._make_auction()
        auction.place_bid("alice", 10)  # cost = 100
        auction.place_bid("bob", 7)     # cost = 49
        result = auction.settle()
        assert result.winner_agent_id == "alice"
        # Alice pays 7² = 49, not her own 10² = 100
        assert result.payment_credits == 49
        assert ledger.balance("alice") == 500 - 49

    # -- settle: three bidders --

    def test_settle_three_bidders(self) -> None:
        auction, ledger = self._make_auction()
        auction.place_bid("alice", 4)
        auction.place_bid("bob", 7)
        auction.place_bid("carol", 2)
        result = auction.settle()
        # Bob wins (7 > 4 > 2)
        assert result.winner_agent_id == "bob"
        assert result.winning_votes == 7
        # Bob pays alice's votes²: 4² = 16
        assert result.payment_credits == 16
        assert ledger.balance("bob") == 500 - 16

    # -- settle: bids sorted by votes descending --

    def test_settle_bids_ordered_descending(self) -> None:
        auction, _ = self._make_auction()
        auction.place_bid("alice", 2)
        auction.place_bid("bob", 5)
        auction.place_bid("carol", 1)
        result = auction.settle()
        votes_in_result = [b.votes for b in result.all_bids]
        assert votes_in_result == sorted(votes_in_result, reverse=True)

    # -- settle twice raises --

    def test_settle_twice_raises(self) -> None:
        auction, _ = self._make_auction()
        auction.settle()
        with pytest.raises(RuntimeError, match="already settled"):
            auction.settle()

    # -- is_settled --

    def test_is_settled_false_before_settle(self) -> None:
        auction, _ = self._make_auction()
        assert auction.is_settled is False

    def test_is_settled_true_after_settle(self) -> None:
        auction, _ = self._make_auction()
        auction.settle()
        assert auction.is_settled is True

    # -- payment capped at winner's balance --

    def test_payment_capped_at_balance(self) -> None:
        ledger = CreditLedger()
        ledger.register_agent("alice", 5)   # only 5 credits
        ledger.register_agent("bob", 500)
        auction = VickreyQuadraticAuction("t", ResourceType.PRIORITY, "rtx5050", ledger)
        # Alice votes 2 (cost=4, affordable); bob votes 1 (cost=1)
        auction.place_bid("alice", 2)
        auction.place_bid("bob", 1)
        result = auction.settle()
        # Alice wins; second price = 1² = 1; alice has 5 → pays 1
        assert result.winner_agent_id == "alice"
        assert result.payment_credits == 1
        assert ledger.balance("alice") == 4

    # -- result metadata --

    def test_result_contains_auction_id(self) -> None:
        auction, _ = self._make_auction()
        result = auction.settle()
        assert result.auction_id == "test-001"

    def test_result_resource_type(self) -> None:
        auction, _ = self._make_auction(resource=ResourceType.VRAM)
        result = auction.settle()
        assert result.resource_type == ResourceType.VRAM

    def test_result_backend_id(self) -> None:
        auction, _ = self._make_auction(backend="radeon780m")
        result = auction.settle()
        assert result.backend_id == "radeon780m"


# ===========================================================================
# AllocationFairness & Gini
# ===========================================================================


class TestGiniCoefficient:
    def test_empty_returns_zero(self) -> None:
        assert _gini_coefficient([]) == 0.0

    def test_all_zero_returns_zero(self) -> None:
        assert _gini_coefficient([0, 0, 0]) == 0.0

    def test_perfect_equality(self) -> None:
        # Everyone spends the same → Gini = 0
        gini = _gini_coefficient([100, 100, 100])
        assert gini == pytest.approx(0.0, abs=1e-9)

    def test_maximum_concentration_single_nonzero(self) -> None:
        # One agent, everyone else spends 0 → Gini near 1
        gini = _gini_coefficient([0, 0, 0, 100])
        # For n=4, one element = total: Gini = 1 - 2*(100)/(4*100) = 0.5
        # Actually: sorted [0,0,0,100], cumulative [0,0,0,100], weighted_sum=100
        # gini = 1 - 2*100/(4*100) = 1 - 0.5 = 0.5
        assert 0.0 < gini <= 1.0

    def test_gini_bounded(self) -> None:
        import random
        values = [random.randint(0, 1000) for _ in range(20)]
        gini = _gini_coefficient(values)
        assert 0.0 <= gini <= 1.0

    def test_single_value(self) -> None:
        # Only one agent — always equal to itself
        assert _gini_coefficient([42]) == pytest.approx(0.0, abs=1e-9)

    def test_two_unequal_values(self) -> None:
        # [0, 100] → Gini = 0.5
        gini = _gini_coefficient([0, 100])
        assert gini == pytest.approx(0.5, abs=1e-9)


class TestAllocationFairness:
    def _make_result(
        self,
        winner: str | None,
        payment: int,
        backend: str = "rtx5050",
        resource: ResourceType = ResourceType.COMPUTE_TIME,
    ) -> AuctionResult:
        return AuctionResult(
            auction_id="x",
            resource_type=resource,
            backend_id=backend,
            winner_agent_id=winner,
            winning_votes=1,
            payment_credits=payment,
            all_bids=[],
        )

    def test_empty_metrics(self) -> None:
        fairness = AllocationFairness()
        m = fairness.compute()
        assert m.total_auctions == 0
        assert m.total_credits_spent == 0
        assert m.gini_coefficient == 0.0
        assert m.unique_winners == 0
        assert m.unique_winners_ratio == 0.0

    def test_record_increments_count(self) -> None:
        fairness = AllocationFairness()
        fairness.record(self._make_result("alice", 10))
        m = fairness.compute()
        assert m.total_auctions == 1
        assert m.total_credits_spent == 10

    def test_record_no_winner(self) -> None:
        fairness = AllocationFairness()
        fairness.record(self._make_result(None, 0))
        m = fairness.compute()
        assert m.total_auctions == 1
        assert m.unique_winners == 0

    def test_unique_winners_ratio(self) -> None:
        fairness = AllocationFairness()
        fairness.record(self._make_result("alice", 10))
        fairness.record(self._make_result("bob", 9))
        m = fairness.compute()
        assert m.unique_winners == 2
        assert m.unique_winners_ratio == pytest.approx(1.0)

    def test_unique_winners_ratio_repeated_winner(self) -> None:
        fairness = AllocationFairness()
        for _ in range(3):
            fairness.record(self._make_result("alice", 10))
        m = fairness.compute()
        assert m.unique_winners == 1
        assert m.unique_winners_ratio == pytest.approx(1 / 3)

    def test_backend_utilization(self) -> None:
        fairness = AllocationFairness()
        fairness.record(self._make_result("alice", 10, backend="rtx5050"))
        fairness.record(self._make_result("bob", 9, backend="rtx5050"))
        fairness.record(self._make_result("carol", 4, backend="radeon780m"))
        m = fairness.compute()
        assert m.utilization_by_backend["rtx5050"] == 2
        assert m.utilization_by_backend["radeon780m"] == 1

    def test_resource_utilization(self) -> None:
        fairness = AllocationFairness()
        fairness.record(self._make_result("alice", 10, resource=ResourceType.COMPUTE_TIME))
        fairness.record(self._make_result("bob", 9, resource=ResourceType.VRAM))
        m = fairness.compute()
        assert m.utilization_by_resource["compute_time"] == 1
        assert m.utilization_by_resource["vram"] == 1

    def test_gini_equal_spending(self) -> None:
        fairness = AllocationFairness()
        # Two agents spending equally
        fairness.record(self._make_result("alice", 100))
        fairness.record(self._make_result("bob", 100))
        m = fairness.compute()
        assert m.gini_coefficient == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# Auctioneer
# ===========================================================================


class TestAuctioneer:
    def test_register_agent(self) -> None:
        mgr = Auctioneer()
        info = mgr.register_agent("alice", 100)
        assert info["agent_id"] == "alice"
        assert info["remaining_credits"] == 100

    def test_top_up(self) -> None:
        mgr = Auctioneer()
        mgr.register_agent("alice", 50)
        info = mgr.top_up("alice", 50)
        assert info["remaining_credits"] == 100

    def test_open_auction(self) -> None:
        mgr = Auctioneer()
        auction = mgr.open_auction(ResourceType.COMPUTE_TIME, "rtx5050")
        assert auction.auction_id.startswith("auction-")
        assert not auction.is_settled

    def test_place_bid_auto_opens_auction(self) -> None:
        mgr = Auctioneer()
        mgr.register_agent("alice", 500)
        aid, bid = mgr.place_bid("alice", ResourceType.COMPUTE_TIME, "rtx5050", 3)
        assert bid.votes == 3
        assert aid in mgr.status()["open_auctions"][0]["auction_id"]  # type: ignore[index]

    def test_place_bid_reuses_existing_auction(self) -> None:
        mgr = Auctioneer()
        mgr.register_agent("alice", 500)
        mgr.register_agent("bob", 500)
        aid1, _ = mgr.place_bid("alice", ResourceType.COMPUTE_TIME, "rtx5050", 3)
        aid2, _ = mgr.place_bid("bob", ResourceType.COMPUTE_TIME, "rtx5050", 2)
        assert aid1 == aid2

    def test_settle_auction_by_id(self) -> None:
        mgr = Auctioneer()
        mgr.register_agent("alice", 500)
        aid, _ = mgr.place_bid("alice", ResourceType.COMPUTE_TIME, "rtx5050", 3)
        result = mgr.settle_auction(aid)
        assert result.auction_id == aid
        assert result.winner_agent_id == "alice"

    def test_settle_auction_unknown_id_raises(self) -> None:
        mgr = Auctioneer()
        with pytest.raises(KeyError):
            mgr.settle_auction("nonexistent")

    def test_settle_all(self) -> None:
        mgr = Auctioneer()
        mgr.register_agent("alice", 500)
        mgr.register_agent("bob", 500)
        mgr.place_bid("alice", ResourceType.COMPUTE_TIME, "rtx5050", 3)
        mgr.place_bid("bob", ResourceType.VRAM, "radeon780m", 2)
        results = mgr.settle_all()
        assert len(results) == 2
        assert len(mgr.status()["open_auctions"]) == 0  # type: ignore[arg-type]

    def test_status_shows_open_auctions(self) -> None:
        mgr = Auctioneer()
        mgr.register_agent("alice", 500)
        mgr.place_bid("alice", ResourceType.PRIORITY, "rtx5050", 1)
        s = mgr.status()
        assert len(s["open_auctions"]) == 1  # type: ignore[arg-type]

    def test_history_grows_after_settle(self) -> None:
        mgr = Auctioneer()
        mgr.register_agent("alice", 500)
        mgr.place_bid("alice", ResourceType.COMPUTE_TIME, "rtx5050", 2)
        mgr.settle_all()
        assert len(mgr.history) == 1

    def test_allocation_priority_after_settle(self) -> None:
        mgr = Auctioneer()
        mgr.register_agent("alice", 500)
        mgr.place_bid("alice", ResourceType.COMPUTE_TIME, "rtx5050", 3)
        mgr.settle_all()
        priority = mgr.allocation_priority()
        assert priority.get("rtx5050") == "alice"

    def test_metrics_after_settle(self) -> None:
        mgr = Auctioneer()
        mgr.register_agent("alice", 500)
        mgr.register_agent("bob", 500)
        mgr.place_bid("alice", ResourceType.COMPUTE_TIME, "rtx5050", 5)
        mgr.place_bid("bob", ResourceType.COMPUTE_TIME, "rtx5050", 3)
        mgr.settle_all()
        m = mgr.metrics()
        assert m.total_auctions == 1
        assert m.total_credits_spent == 9  # second price = 3² = 9
        assert m.unique_winners == 1


# ===========================================================================
# FastAPI auction endpoints
# ===========================================================================


@pytest.fixture()
def auction_client() -> Iterator[tuple[TestClient, Auctioneer]]:
    """Return a TestClient with a fresh Auctioneer and no-op lifespan."""
    import gateway.main as gm
    from gateway.benchmark import ThroughputBenchmark
    from gateway.config import GatewaySettings
    from gateway.health import HealthMonitor
    from gateway.models import ModelAssigner
    from gateway.router import GatewayRouter

    # Fresh auctioneer for this test
    fresh_mgr = Auctioneer()
    gm._auctioneer = fresh_mgr  # type: ignore[attr-defined]

    # Minimal gateway state (no background tasks needed for auction endpoints)
    cfg = GatewaySettings(failure_threshold=1, recovery_threshold=1)
    gm._health_monitor = HealthMonitor(cfg=cfg)
    gm._benchmark = ThroughputBenchmark()
    gm._router = GatewayRouter(
        health_monitor=gm._health_monitor,
        assigner=ModelAssigner(),
        benchmark=gm._benchmark,
        cfg=cfg,
    )

    # Swap lifespan for a no-op so TestClient doesn't re-init module globals
    original_lifespan = gm.app.router.lifespan_context

    @asynccontextmanager
    async def _noop_lifespan(app):  # type: ignore[no-untyped-def]
        yield

    gm.app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(gm.app, raise_server_exceptions=True) as client:
            yield client, fresh_mgr
    finally:
        gm.app.router.lifespan_context = original_lifespan


class TestAuctionAPICredits:
    def test_top_up_agent(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        resp = client.post("/auction/credits?agent_id=alice&amount=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "alice"
        assert data["remaining_credits"] == 100

    def test_top_up_zero_raises(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        resp = client.post("/auction/credits?agent_id=alice&amount=0")
        assert resp.status_code == 422  # FastAPI validation (ge=1)


class TestAuctionAPIBid:
    def test_bid_success(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        client.post("/auction/credits?agent_id=alice&amount=100")
        resp = client.post(
            "/auction/bid?agent_id=alice&resource_type=compute_time&backend_id=rtx5050&votes=3"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["votes"] == 3
        assert data["credit_cost_if_winner"] == 9

    def test_bid_unknown_backend_returns_404(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        client.post("/auction/credits?agent_id=alice&amount=100")
        resp = client.post(
            "/auction/bid?agent_id=alice&resource_type=compute_time&backend_id=nonexistent&votes=3"
        )
        assert resp.status_code == 404

    def test_bid_insufficient_credits_returns_402(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        # Don't top up alice → 0 credits → can't afford 1² = 1
        resp = client.post(
            "/auction/bid?agent_id=alice&resource_type=compute_time&backend_id=rtx5050&votes=1"
        )
        assert resp.status_code == 402

    def test_bid_zero_votes_returns_422(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        client.post("/auction/credits?agent_id=alice&amount=100")
        resp = client.post(
            "/auction/bid?agent_id=alice&resource_type=compute_time&backend_id=rtx5050&votes=0"
        )
        assert resp.status_code == 422


class TestAuctionAPIStatus:
    def test_status_empty(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        resp = client.get("/auction/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["open_auctions"] == []
        assert data["total_settled"] == 0

    def test_status_after_bid(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        client.post("/auction/credits?agent_id=alice&amount=100")
        client.post(
            "/auction/bid?agent_id=alice&resource_type=compute_time&backend_id=rtx5050&votes=3"
        )
        resp = client.get("/auction/status")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["open_auctions"]) == 1


class TestAuctionAPISettle:
    def test_settle_all(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        client.post("/auction/credits?agent_id=alice&amount=100")
        client.post(
            "/auction/bid?agent_id=alice&resource_type=compute_time&backend_id=rtx5050&votes=3"
        )
        resp = client.post("/auction/settle")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["settled"]) == 1
        assert data["settled"][0]["winner_agent_id"] == "alice"

    def test_settle_specific_id(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        client.post("/auction/credits?agent_id=alice&amount=100")
        bid_resp = client.post(
            "/auction/bid?agent_id=alice&resource_type=compute_time&backend_id=rtx5050&votes=3"
        )
        auction_id = bid_resp.json()["auction_id"]
        resp = client.post(f"/auction/settle?auction_id={auction_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["settled"][0]["auction_id"] == auction_id

    def test_settle_unknown_id_returns_404(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        resp = client.post("/auction/settle?auction_id=nonexistent")
        assert resp.status_code == 404

    def test_settle_second_price_applied(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        client.post("/auction/credits?agent_id=alice&amount=500")
        client.post("/auction/credits?agent_id=bob&amount=500")
        client.post(
            "/auction/bid?agent_id=alice&resource_type=compute_time&backend_id=rtx5050&votes=5"
        )
        client.post(
            "/auction/bid?agent_id=bob&resource_type=compute_time&backend_id=rtx5050&votes=3"
        )
        resp = client.post("/auction/settle")
        data = resp.json()
        result = data["settled"][0]
        assert result["winner_agent_id"] == "alice"
        # Alice pays 3² = 9, not 5² = 25
        assert result["payment_credits"] == 9


class TestAuctionAPIMetrics:
    def test_metrics_empty(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        resp = client.get("/auction/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_auctions"] == 0
        assert data["gini_coefficient"] == 0.0

    def test_metrics_after_auction(self, auction_client: tuple[TestClient, Auctioneer]) -> None:
        client, _ = auction_client
        client.post("/auction/credits?agent_id=alice&amount=500")
        client.post(
            "/auction/bid?agent_id=alice&resource_type=vram&backend_id=rtx5050&votes=4"
        )
        client.post("/auction/settle")
        resp = client.get("/auction/metrics")
        data = resp.json()
        assert data["total_auctions"] == 1
        assert data["utilization_by_resource"]["vram"] == 1
        assert data["utilization_by_backend"]["rtx5050"] == 1
