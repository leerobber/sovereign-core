"""
RES-08: Agent Economy Interoperability — MCP/A2A Standard Interface
Sovereign Core — standard auction protocol interface

Implementation path from issue:
  1. Define standard auction protocol interface (MCP-compatible)   ← this file
  2. Implement discovery endpoint for agent capabilities
  3. Extend S-PAX to validate cross-node agent identity
  4. Test with TIA ↔ PC agent marketplace

This makes the VQ auction system discoverable via standard MCP/A2A protocol
so external agents (TIA, IoT, drone agents) can participate without custom integration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP-compatible capability descriptors
# ---------------------------------------------------------------------------

class CapabilityType(str, Enum):
    INFERENCE   = "inference"    # LLM token generation
    COMPUTE     = "compute"      # Raw CPU/GPU compute
    MEMORY      = "memory"       # RAM/VRAM allocation
    BANDWIDTH   = "bandwidth"    # Network I/O
    STORAGE     = "storage"      # Disk I/O


@dataclass
class AgentCapability:
    """MCP-compatible capability advertisement for an agent."""
    capability_type: CapabilityType
    capacity: float              # Available units (tokens/s, GiB, Mbps, etc.)
    unit: str                    # "tokens/s", "GiB", "Mbps", "IOPS"
    backend_id: str              # Which Sovereign Core backend provides this
    price_per_unit: float = 1.0  # Credits per unit (for negotiation)
    metadata: dict = field(default_factory=dict)

    def to_mcp_tool(self) -> dict:
        """Serialize as an MCP Tool descriptor."""
        return {
            "name": f"sovereign_core.{self.capability_type.value}",
            "description": f"Sovereign Core {self.capability_type.value} capability on {self.backend_id}",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "Requesting agent identifier"},
                    "units_requested": {"type": "number", "description": f"Units in {self.unit}"},
                    "duration_s": {"type": "number", "description": "Duration in seconds"},
                },
                "required": ["agent_id", "units_requested"],
            },
            "metadata": {
                "capacity": self.capacity,
                "unit": self.unit,
                "price_per_unit_credits": self.price_per_unit,
                "backend_id": self.backend_id,
                **self.metadata,
            },
        }


# ---------------------------------------------------------------------------
# Standard auction protocol interface (MCP/A2A compatible)
# ---------------------------------------------------------------------------

class BidStatus(str, Enum):
    PENDING  = "pending"
    WON      = "won"
    LOST     = "lost"
    EXPIRED  = "expired"


@dataclass
class StandardBid:
    """
    A2A-compatible bid structure.
    Any agent from any platform can submit this format.
    """
    agent_id: str
    agent_node: str              # Node identifier (e.g. "tia-node-1", "pc-agent-42")
    capability_type: CapabilityType
    backend_id: str
    votes: int                   # Quadratic votes (cost = votes²)
    credits_available: int
    priority: int = 1            # 1=normal, 2=high, 3=critical
    ttl_s: int = 60              # Bid time-to-live
    metadata: dict = field(default_factory=dict)

    # Computed
    bid_id: str = field(default="")
    submitted_at: float = field(default_factory=time.time)
    status: BidStatus = BidStatus.PENDING

    def __post_init__(self) -> None:
        if not self.bid_id:
            self.bid_id = hashlib.sha256(
                f"{self.agent_id}:{self.capability_type}:{self.submitted_at}".encode()
            ).hexdigest()[:12]

    @property
    def credit_cost(self) -> int:
        return self.votes ** 2

    @property
    def is_expired(self) -> bool:
        return time.time() > self.submitted_at + self.ttl_s

    def to_a2a_message(self) -> dict:
        """Serialize as an A2A protocol message."""
        return {
            "type": "sovereign_core.auction.bid",
            "version": "1.0",
            "bid_id": self.bid_id,
            "from": {"agent_id": self.agent_id, "node": self.agent_node},
            "payload": {
                "capability": self.capability_type.value,
                "backend_id": self.backend_id,
                "votes": self.votes,
                "credit_cost": self.credit_cost,
                "priority": self.priority,
                "ttl_s": self.ttl_s,
            },
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Discovery endpoint — agents can find what's available without custom integration
# ---------------------------------------------------------------------------

@dataclass
class SovereignCoreMarketplace:
    """
    Standard discovery interface for the Sovereign Core agent economy.
    Exposes capabilities via MCP Tool format and accepts A2A bids.

    External agents (TIA nodes, PC agents, IoT/drone agents) call:
      GET /marketplace/discover   → list capabilities
      POST /marketplace/bid       → submit a standard bid
      GET /marketplace/status     → auction status
    """
    node_id: str = "sovereign-core-primary"
    capabilities: list[AgentCapability] = field(default_factory=list)
    registered_agents: dict[str, dict] = field(default_factory=dict)

    def register_capability(self, cap: AgentCapability) -> None:
        self.capabilities.append(cap)
        logger.info("Registered capability: %s on %s", cap.capability_type, cap.backend_id)

    def discover(self, filter_type: Optional[CapabilityType] = None) -> dict:
        """MCP-compatible discovery response."""
        caps = self.capabilities
        if filter_type:
            caps = [c for c in caps if c.capability_type == filter_type]

        return {
            "node_id": self.node_id,
            "protocol": "sovereign-core-marketplace/1.0",
            "mcp_compatible": True,
            "a2a_compatible": True,
            "capabilities": [c.to_mcp_tool() for c in caps],
            "auction_endpoint": "/auction/bid",
            "spax_endpoint": "/spax/verify",
            "registered_agents": len(self.registered_agents),
        }

    def register_agent(self, agent_id: str, node: str, capabilities: list[str]) -> str:
        """Register an external agent for participation in the economy."""
        token = hashlib.sha256(f"{agent_id}:{node}:{time.time()}".encode()).hexdigest()[:24]
        self.registered_agents[agent_id] = {
            "agent_id": agent_id,
            "node": node,
            "capabilities": capabilities,
            "registered_at": time.time(),
            "spax_token": token,
        }
        logger.info("Agent %s@%s registered for marketplace", agent_id, node)
        return token


# ---------------------------------------------------------------------------
# S-PAX cross-node identity validation
# ---------------------------------------------------------------------------

class SPAXValidator:
    """
    S-PAX (Sovereign Proof-of-Agent eXchange) — trust backbone for cross-node economy.
    Validates agent identity when bids come from external nodes (TIA, PC agents, etc.)
    """

    def __init__(self, marketplace: SovereignCoreMarketplace) -> None:
        self.marketplace = marketplace
        self._revoked_tokens: set[str] = set()

    def validate(self, agent_id: str, spax_token: str, bid: StandardBid) -> tuple[bool, str]:
        """
        Validate a cross-node agent's identity before accepting their bid.
        Returns (is_valid, reason).
        """
        # Check agent is registered
        if agent_id not in self.marketplace.registered_agents:
            return False, f"Agent {agent_id} not registered in marketplace"

        reg = self.marketplace.registered_agents[agent_id]

        # Validate S-PAX token
        if reg["spax_token"] != spax_token:
            return False, "Invalid S-PAX token"

        if spax_token in self._revoked_tokens:
            return False, "S-PAX token revoked"

        # Validate bid node matches registered node
        if bid.agent_node != reg["node"]:
            return False, f"Node mismatch: bid from {bid.agent_node}, registered as {reg['node']}"

        # Validate capability is in agent's registered capabilities
        if bid.capability_type.value not in reg["capabilities"]:
            return False, f"Agent not registered for {bid.capability_type} capability"

        return True, "OK"

    def revoke(self, spax_token: str) -> None:
        self._revoked_tokens.add(spax_token)
        logger.warning("S-PAX token revoked: %s", spax_token[:8] + "...")


# ---------------------------------------------------------------------------
# Pre-configured marketplace with Sovereign Core backends
# ---------------------------------------------------------------------------

def build_sovereign_marketplace() -> SovereignCoreMarketplace:
    """Build the default Sovereign Core marketplace with all backend capabilities."""
    mp = SovereignCoreMarketplace(node_id="sovereign-core-primary")

    mp.register_capability(AgentCapability(
        capability_type=CapabilityType.INFERENCE,
        capacity=150.0, unit="tokens/s",
        backend_id="nvidia_rtx5050",
        price_per_unit=2.0,
        metadata={"vram_gib": 8, "model": "qwen2.5-32b-awq"},
    ))
    mp.register_capability(AgentCapability(
        capability_type=CapabilityType.INFERENCE,
        capacity=80.0, unit="tokens/s",
        backend_id="radeon_780m",
        price_per_unit=1.0,
        metadata={"vram_gib": 4, "model": "deepseek-coder-33b"},
    ))
    mp.register_capability(AgentCapability(
        capability_type=CapabilityType.COMPUTE,
        capacity=32.0, unit="CPU-cores",
        backend_id="ryzen7_cpu",
        price_per_unit=0.5,
        metadata={"threads": 32},
    ))
    mp.register_capability(AgentCapability(
        capability_type=CapabilityType.MEMORY,
        capacity=32.0, unit="GiB",
        backend_id="ryzen7_cpu",
        price_per_unit=0.3,
    ))

    return mp
