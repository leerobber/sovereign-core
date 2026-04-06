"""Vickrey-Quadratic Auction — economic resource allocation core.

Combines two complementary mechanism-design primitives:

**Quadratic Voting (QV)**
    The cost of casting ``v`` votes is ``v²`` credits.  This gives agents with
    larger budgets diminishing marginal influence, making the mechanism more
    equitable than a simple pay-per-vote auction.

**Vickrey Second-Price Rule**
    The auction winner is the highest-vote bidder, but they pay the credit cost
    of the *second*-highest bidder (``second_votes²``), not their own bid.
    This makes truthful bidding a *dominant strategy*: bidding your true value
    is always optimal regardless of what others bid.

Combined (Vickrey-Quadratic)
────────────────────────────
1. Agents express demand in integer *votes* (not raw credits).
2. Winner = agent with the most votes.
3. Payment = ``second_highest_votes²`` credits (0 if only one bidder).
4. This allocates resources to those who value them most, while protecting
   smaller agents from being priced out by well-funded competitors.

Credit / Budget System
──────────────────────
Each agent starts with a configurable credit balance managed by
:class:`CreditLedger`.  Credits cannot go below zero; attempted
over-spend raises :exc:`InsufficientCreditsError`.

Fairness Metrics
────────────────
:class:`AllocationFairness` tracks historical auction outcomes and
computes:

- **Gini coefficient** of credits spent (0 = perfect equality, 1 = maximum
  concentration).
- **Unique winners ratio** — proportion of auctions won by distinct agents.
- Per-backend and per-resource **utilization** counts.

Gateway Integration
───────────────────
The FastAPI application (``gateway/main.py``) exposes auction endpoints and
maintains a module-level :class:`Auctioneer` instance that the inference proxy
consults to honour allocation priority.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ResourceType(str, Enum):
    """Categories of compute resources available for auction."""

    COMPUTE_TIME = "compute_time"
    VRAM = "vram"
    BANDWIDTH = "bandwidth"
    PRIORITY = "priority"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Bid:
    """A single agent's bid for a resource slot.

    Args:
        agent_id: Unique identifier for the bidding agent.
        resource_type: The resource category being bid on.
        backend_id: Target compute backend (e.g. ``"rtx5050"``).
        votes: Number of quadratic votes cast (≥ 1).  Credit cost = votes².
        timestamp: Unix timestamp when the bid was placed.
    """

    agent_id: str
    resource_type: ResourceType
    backend_id: str
    votes: int
    timestamp: float = field(default_factory=time.time)

    @property
    def credit_cost(self) -> int:
        """Quadratic credit cost for this bid: ``votes²``."""
        return self.votes * self.votes


@dataclass
class AuctionResult:
    """Outcome of a settled auction round.

    Args:
        auction_id: Unique identifier for this auction instance.
        resource_type: Resource category that was auctioned.
        backend_id: Target backend for which the slot was auctioned.
        winner_agent_id: Agent that won, or ``None`` if no bids were placed.
        winning_votes: Vote count cast by the winner (0 if no winner).
        payment_credits: Credits charged to the winner (second-price rule).
        all_bids: All bids submitted in this round, ordered by votes descending.
        settled_at: Unix timestamp of settlement.
    """

    auction_id: str
    resource_type: ResourceType
    backend_id: str
    winner_agent_id: Optional[str]
    winning_votes: int
    payment_credits: int
    all_bids: list[Bid]
    settled_at: float = field(default_factory=time.time)


@dataclass
class AllocationMetrics:
    """Snapshot of allocation fairness statistics.

    Args:
        total_auctions: Total number of settled auction rounds.
        total_credits_spent: Sum of all payments across all auctions.
        gini_coefficient: Gini index of per-agent credit expenditure
            (0 = perfectly equal, 1 = maximally concentrated).
        unique_winners: Number of distinct agents that have ever won.
        unique_winners_ratio: Fraction of auctions won by distinct agents.
        utilization_by_backend: Auction count per backend ID.
        utilization_by_resource: Auction count per resource type.
    """

    total_auctions: int
    total_credits_spent: int
    gini_coefficient: float
    unique_winners: int
    unique_winners_ratio: float
    utilization_by_backend: dict[str, int]
    utilization_by_resource: dict[str, int]


# ---------------------------------------------------------------------------
# Credit / Budget system
# ---------------------------------------------------------------------------


class InsufficientCreditsError(Exception):
    """Raised when an agent attempts to spend more credits than available."""


class AgentBudget:
    """Tracks the credit balance for a single agent.

    Args:
        agent_id: Unique agent identifier.
        initial_credits: Starting credit balance (non-negative).
    """

    def __init__(self, agent_id: str, initial_credits: int = 0) -> None:
        if initial_credits < 0:
            raise ValueError("initial_credits must be non-negative")
        self.agent_id = agent_id
        self._total: int = initial_credits
        self._spent: int = 0

    @property
    def total_credits(self) -> int:
        """Total credits ever allocated to this agent."""
        return self._total

    @property
    def spent_credits(self) -> int:
        """Credits consumed so far."""
        return self._spent

    @property
    def remaining_credits(self) -> int:
        """Credits still available."""
        return self._total - self._spent

    def top_up(self, amount: int) -> None:
        """Add *amount* credits to this agent's balance.

        Args:
            amount: Positive integer credit amount to add.

        Raises:
            ValueError: If *amount* is not positive.
        """
        if amount <= 0:
            raise ValueError("top_up amount must be positive")
        self._total += amount
        logger.debug("Agent %s topped up by %d → balance %d", self.agent_id, amount, self._total)

    def spend(self, amount: int) -> None:
        """Deduct *amount* credits from the agent's balance.

        Args:
            amount: Non-negative credits to spend.

        Raises:
            InsufficientCreditsError: If the agent cannot afford *amount*.
            ValueError: If *amount* is negative.
        """
        if amount < 0:
            raise ValueError("spend amount must be non-negative")
        if amount > self.remaining_credits:
            raise InsufficientCreditsError(
                f"Agent {self.agent_id!r} has {self.remaining_credits} credits "
                f"but tried to spend {amount}"
            )
        self._spent += amount
        logger.debug(
            "Agent %s spent %d credits (remaining=%d)", self.agent_id, amount, self.remaining_credits
        )

    def can_afford(self, amount: int) -> bool:
        """Return ``True`` if the agent has at least *amount* credits."""
        return self.remaining_credits >= amount


class CreditLedger:
    """Manages credit budgets for all agents in the system.

    Agents are created on first reference with zero balance.  Use
    :meth:`top_up` to allocate credits before bidding.
    """

    def __init__(self) -> None:
        self._budgets: dict[str, AgentBudget] = {}

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, initial_credits: int = 0) -> AgentBudget:
        """Register a new agent (no-op if already registered).

        Args:
            agent_id: Unique agent identifier.
            initial_credits: Starting balance for new agents.

        Returns:
            The agent's :class:`AgentBudget`.
        """
        if agent_id not in self._budgets:
            self._budgets[agent_id] = AgentBudget(agent_id, initial_credits)
            logger.info("Registered agent %s with %d credits", agent_id, initial_credits)
        return self._budgets[agent_id]

    def get_budget(self, agent_id: str) -> AgentBudget:
        """Return the budget for *agent_id*, creating it with zero balance if absent.

        Args:
            agent_id: Unique agent identifier.

        Returns:
            The agent's :class:`AgentBudget`.
        """
        return self.register_agent(agent_id)

    def top_up(self, agent_id: str, amount: int) -> None:
        """Add *amount* credits to *agent_id*'s balance.

        Args:
            agent_id: Unique agent identifier.
            amount: Positive credit amount to add.
        """
        self.get_budget(agent_id).top_up(amount)

    def balance(self, agent_id: str) -> int:
        """Return the remaining credit balance for *agent_id*."""
        return self.get_budget(agent_id).remaining_credits

    def can_afford(self, agent_id: str, amount: int) -> bool:
        """Return ``True`` if *agent_id* can afford *amount* credits."""
        return self.get_budget(agent_id).can_afford(amount)

    def spend(self, agent_id: str, amount: int) -> None:
        """Deduct *amount* credits from *agent_id*.

        Raises:
            InsufficientCreditsError: If the agent cannot afford *amount*.
        """
        self.get_budget(agent_id).spend(amount)

    def all_balances(self) -> dict[str, int]:
        """Return a snapshot of all agent remaining balances."""
        return {aid: b.remaining_credits for aid, b in self._budgets.items()}

    def all_spending(self) -> dict[str, int]:
        """Return a snapshot of all agent cumulative spending."""
        return {aid: b.spent_credits for aid, b in self._budgets.items()}


# ---------------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------------


class AllocationFairness:
    """Tracks historical auction outcomes and computes fairness statistics.

    Maintains running counts of auction results to enable efficient metric
    calculation without replaying full history.
    """

    def __init__(self) -> None:
        self._total_auctions: int = 0
        self._total_credits_spent: int = 0
        self._agent_spending: dict[str, int] = {}
        self._winner_counts: dict[str, int] = {}
        self._backend_utilization: dict[str, int] = {}
        self._resource_utilization: dict[str, int] = {}

    def record(self, result: AuctionResult) -> None:
        """Update statistics from a settled auction *result*.

        Args:
            result: The :class:`AuctionResult` to incorporate.
        """
        self._total_auctions += 1
        self._total_credits_spent += result.payment_credits

        # Track per-backend utilization
        bid = result.backend_id
        self._backend_utilization[bid] = self._backend_utilization.get(bid, 0) + 1

        # Track per-resource utilization
        rtype = result.resource_type.value
        self._resource_utilization[rtype] = self._resource_utilization.get(rtype, 0) + 1

        if result.winner_agent_id is not None:
            wid = result.winner_agent_id
            self._winner_counts[wid] = self._winner_counts.get(wid, 0) + 1
            self._agent_spending[wid] = (
                self._agent_spending.get(wid, 0) + result.payment_credits
            )

    def compute(self) -> AllocationMetrics:
        """Compute and return a current :class:`AllocationMetrics` snapshot."""
        gini = _gini_coefficient(list(self._agent_spending.values()))
        unique_winners = len(self._winner_counts)
        ratio = unique_winners / self._total_auctions if self._total_auctions > 0 else 0.0
        return AllocationMetrics(
            total_auctions=self._total_auctions,
            total_credits_spent=self._total_credits_spent,
            gini_coefficient=gini,
            unique_winners=unique_winners,
            unique_winners_ratio=ratio,
            utilization_by_backend=dict(self._backend_utilization),
            utilization_by_resource=dict(self._resource_utilization),
        )


def _gini_coefficient(values: list[int]) -> float:
    """Compute the Gini coefficient for a list of non-negative integer values.

    Returns 0.0 for empty or all-zero inputs (perfect equality).

    Uses the standard sorted-list formula:
    ``G = (2 * Σ (i+1)*x[i] − (n+1) * total) / (n * total)``
    where *x* is sorted ascending and *i* is 0-indexed.

    Args:
        values: Non-negative integers (e.g. per-agent credit expenditures).

    Returns:
        Gini coefficient in ``[0.0, 1.0]``.
    """
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    sorted_vals = sorted(values)
    weighted = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * weighted - (n + 1) * total) / (n * total)


# ---------------------------------------------------------------------------
# Auction mechanism
# ---------------------------------------------------------------------------


class VickreyQuadraticAuction:
    """Vickrey second-price auction with quadratic vote costs.

    Each auction instance manages bids for a single ``(resource_type,
    backend_id)`` pair within one round.  After all agents have bid, call
    :meth:`settle` to determine the winner and deduct credits via the
    :class:`CreditLedger`.

    Args:
        auction_id: Unique identifier for this auction round.
        resource_type: The resource category being auctioned.
        backend_id: The compute backend being allocated.
        ledger: Shared :class:`CreditLedger` for credit accounting.
    """

    def __init__(
        self,
        auction_id: str,
        resource_type: ResourceType,
        backend_id: str,
        ledger: CreditLedger,
    ) -> None:
        self._auction_id = auction_id
        self._resource_type = resource_type
        self._backend_id = backend_id
        self._ledger = ledger
        self._bids: list[Bid] = []
        self._settled: bool = False

    @property
    def auction_id(self) -> str:
        return self._auction_id

    @property
    def resource_type(self) -> ResourceType:
        """Resource category being auctioned."""
        return self._resource_type

    @property
    def backend_id(self) -> str:
        """Target compute backend for this auction."""
        return self._backend_id

    @property
    def is_settled(self) -> bool:
        """``True`` if this auction has already been settled."""
        return self._settled

    @property
    def bids(self) -> list[Bid]:
        """Read-only view of bids placed so far."""
        return list(self._bids)

    def place_bid(self, agent_id: str, votes: int) -> Bid:
        """Place a bid of *votes* quadratic votes for this auction.

        Validates that the agent has sufficient credits to cover the quadratic
        cost (``votes²``) without deducting them yet — credits are only spent
        at :meth:`settle` time.

        Args:
            agent_id: Bidding agent identifier.
            votes: Number of votes to cast (must be ≥ 1).

        Returns:
            The recorded :class:`Bid`.

        Raises:
            RuntimeError: If the auction has already been settled.
            ValueError: If *votes* < 1.
            InsufficientCreditsError: If the agent cannot afford ``votes²``
                credits.
        """
        if self._settled:
            raise RuntimeError(f"Auction {self._auction_id!r} is already settled")
        if votes < 1:
            raise ValueError("votes must be ≥ 1")

        cost = votes * votes
        if not self._ledger.can_afford(agent_id, cost):
            balance = self._ledger.balance(agent_id)
            raise InsufficientCreditsError(
                f"Agent {agent_id!r} needs {cost} credits for {votes} votes "
                f"but only has {balance}"
            )

        bid = Bid(
            agent_id=agent_id,
            resource_type=self._resource_type,
            backend_id=self._backend_id,
            votes=votes,
        )
        self._bids.append(bid)
        logger.info(
            "Bid placed: auction=%s agent=%s votes=%d cost=%d",
            self._auction_id,
            agent_id,
            votes,
            cost,
        )
        return bid

    def settle(self) -> AuctionResult:
        """Settle this auction and return the outcome.

        Determines the winner using the Vickrey second-price rule applied to
        quadratic vote costs:

        - **Winner**: agent with the highest vote count.
        - **Payment**: ``second_highest_votes²`` credits (0 if ≤ 1 bidder).

        Credits are deducted from the winner's ledger balance.  The auction
        is marked settled and no further bids can be placed.

        Returns:
            :class:`AuctionResult` describing the outcome.

        Raises:
            RuntimeError: If the auction has already been settled.
        """
        if self._settled:
            raise RuntimeError(f"Auction {self._auction_id!r} is already settled")
        self._settled = True

        if not self._bids:
            logger.info("Auction %s settled with no bids", self._auction_id)
            return AuctionResult(
                auction_id=self._auction_id,
                resource_type=self._resource_type,
                backend_id=self._backend_id,
                winner_agent_id=None,
                winning_votes=0,
                payment_credits=0,
                all_bids=[],
            )

        # Sort descending by votes to find winner and second-price
        sorted_bids = sorted(self._bids, key=lambda b: b.votes, reverse=True)
        winner = sorted_bids[0]

        # Second-price: payment = second highest votes² (0 if only one bid)
        second_votes = sorted_bids[1].votes if len(sorted_bids) > 1 else 0
        payment = second_votes * second_votes

        # Deduct credits (capped at winner's remaining balance for robustness)
        actual_payment = min(payment, self._ledger.balance(winner.agent_id))
        if actual_payment > 0:
            self._ledger.spend(winner.agent_id, actual_payment)

        logger.info(
            "Auction %s settled: winner=%s votes=%d payment=%d",
            self._auction_id,
            winner.agent_id,
            winner.votes,
            actual_payment,
        )
        return AuctionResult(
            auction_id=self._auction_id,
            resource_type=self._resource_type,
            backend_id=self._backend_id,
            winner_agent_id=winner.agent_id,
            winning_votes=winner.votes,
            payment_credits=actual_payment,
            all_bids=sorted_bids,
        )


# ---------------------------------------------------------------------------
# High-level auctioneer (state container used by FastAPI)
# ---------------------------------------------------------------------------

_AUCTION_COUNTER: int = 0


def _next_auction_id() -> str:
    global _AUCTION_COUNTER
    _AUCTION_COUNTER += 1
    return f"auction-{_AUCTION_COUNTER:06d}"


class Auctioneer:
    """Manages multiple concurrent auction rounds and the shared credit ledger.

    Acts as the top-level coordinator between the FastAPI endpoints and the
    underlying :class:`VickreyQuadraticAuction` instances.

    Maintains:
    - A :class:`CreditLedger` for all agent budgets.
    - A registry of open (unsettled) auctions keyed by auction ID.
    - Historical results fed into :class:`AllocationFairness`.
    """

    def __init__(self) -> None:
        self._ledger = CreditLedger()
        self._open: dict[str, VickreyQuadraticAuction] = {}
        self._history: list[AuctionResult] = []
        self._fairness = AllocationFairness()

    # ------------------------------------------------------------------
    # Agent / credit management
    # ------------------------------------------------------------------

    @property
    def ledger(self) -> CreditLedger:
        """Shared credit ledger."""
        return self._ledger

    def register_agent(self, agent_id: str, initial_credits: int = 0) -> dict[str, object]:
        """Register an agent and return their current balance information.

        Args:
            agent_id: Unique agent identifier.
            initial_credits: Credits to award on first registration (ignored on
                subsequent calls).

        Returns:
            Dict with ``agent_id`` and ``remaining_credits``.
        """
        budget = self._ledger.register_agent(agent_id, initial_credits)
        return {
            "agent_id": agent_id,
            "remaining_credits": budget.remaining_credits,
            "total_credits": budget.total_credits,
            "spent_credits": budget.spent_credits,
        }

    def top_up(self, agent_id: str, amount: int) -> dict[str, object]:
        """Add *amount* credits to *agent_id*.

        Returns:
            Updated balance dict.
        """
        self._ledger.top_up(agent_id, amount)
        budget = self._ledger.get_budget(agent_id)
        return {
            "agent_id": agent_id,
            "remaining_credits": budget.remaining_credits,
            "total_credits": budget.total_credits,
            "spent_credits": budget.spent_credits,
        }

    # ------------------------------------------------------------------
    # Auction lifecycle
    # ------------------------------------------------------------------

    def open_auction(
        self, resource_type: ResourceType, backend_id: str
    ) -> VickreyQuadraticAuction:
        """Create and register a new auction round.

        Args:
            resource_type: Resource category to auction.
            backend_id: Target compute backend.

        Returns:
            The newly created :class:`VickreyQuadraticAuction`.
        """
        aid = _next_auction_id()
        auction = VickreyQuadraticAuction(
            auction_id=aid,
            resource_type=resource_type,
            backend_id=backend_id,
            ledger=self._ledger,
        )
        self._open[aid] = auction
        logger.info(
            "Opened auction %s resource=%s backend=%s",
            aid,
            resource_type.value,
            backend_id,
        )
        return auction

    def place_bid(
        self,
        agent_id: str,
        resource_type: ResourceType,
        backend_id: str,
        votes: int,
    ) -> tuple[str, Bid]:
        """Place a bid, opening a new auction if none is open for this resource+backend.

        If an open auction already exists for the given ``(resource_type,
        backend_id)`` pair, the bid is added to it.  Otherwise a new auction
        is created automatically.

        Args:
            agent_id: Bidding agent.
            resource_type: Resource category.
            backend_id: Target backend.
            votes: Quadratic votes to cast (≥ 1).

        Returns:
            Tuple of ``(auction_id, Bid)``.
        """
        # Find existing open auction for this resource+backend
        existing = next(
            (
                a
                for a in self._open.values()
                if a.resource_type == resource_type and a.backend_id == backend_id
            ),
            None,
        )
        auction = existing if existing is not None else self.open_auction(resource_type, backend_id)
        bid = auction.place_bid(agent_id, votes)
        return auction.auction_id, bid

    def settle_auction(self, auction_id: str) -> AuctionResult:
        """Settle the auction identified by *auction_id*.

        Args:
            auction_id: ID of the auction to settle.

        Returns:
            :class:`AuctionResult`.

        Raises:
            KeyError: If *auction_id* is not found among open auctions.
        """
        if auction_id not in self._open:
            raise KeyError(f"No open auction with id {auction_id!r}")
        result = self._open.pop(auction_id).settle()
        self._history.append(result)
        self._fairness.record(result)
        return result

    def settle_all(self) -> list[AuctionResult]:
        """Settle all currently open auctions.

        Returns:
            List of :class:`AuctionResult` for each settled auction.
        """
        auction_ids = list(self._open.keys())
        results = [self.settle_auction(aid) for aid in auction_ids]
        logger.info("Settled %d auction(s)", len(results))
        return results

    # ------------------------------------------------------------------
    # Query / reporting
    # ------------------------------------------------------------------

    def status(self) -> dict[str, object]:
        """Return a summary of open auctions and recent activity."""
        return {
            "open_auctions": [
                {
                    "auction_id": a.auction_id,
                    "resource_type": a.resource_type.value,
                    "backend_id": a.backend_id,
                    "bid_count": len(a.bids),
                }
                for a in self._open.values()
            ],
            "total_settled": len(self._history),
            "agent_balances": self._ledger.all_balances(),
        }

    def metrics(self) -> AllocationMetrics:
        """Return current fairness metrics."""
        return self._fairness.compute()

    def allocation_priority(self) -> dict[str, str]:
        """Return a mapping of ``backend_id → winner_agent_id`` from recent results.

        Only the most recent winning result per backend is included.
        """
        priority: dict[str, str] = {}
        for result in reversed(self._history):
            if result.winner_agent_id is not None and result.backend_id not in priority:
                priority[result.backend_id] = result.winner_agent_id
        return priority

    @property
    def history(self) -> list[AuctionResult]:
        """All settled auction results (oldest first)."""
        return list(self._history)


# ---------------------------------------------------------------------------
# Module-level default instance (used by FastAPI endpoints)
# ---------------------------------------------------------------------------

#: The shared :class:`Auctioneer` instance used by the FastAPI application.
auctioneer = Auctioneer()
