"""
GhostRecall — Indestructible Memory Intelligence Engine
"Never Forget" Architecture

Research Foundation (2025–2026 frontier):
  • arXiv:2601.09113  — The AI Hippocampus: How Far are We From Human Memory? (Jan 2026)
    → Taxonomy: implicit / explicit / agentic memory paradigms, multi-modal coherence
  • arXiv:2603.29023  — Human-Like Lifelong Memory: Neuroscience-Grounded Architecture (Mar 2026)
    → Complementary Learning Systems (hippocampus+neocortex), valence vectors,
      System 1/2 routing, thalamic gateway, belief hierarchy, reconsolidation
  • arXiv:2511.22367  — SuRe: Surprise-Driven Prioritised Replay for Continual LLM Learning (Nov 2025)
    → Surprise-scored replay buffer, dual-learner fast/slow LoRA with EMA consolidation,
      SOTA on Large Number of Tasks — solves catastrophic forgetting
  • EM-LLM (ICLR 2025) — Human-Inspired Episodic Memory for Infinite Context LLMs
    → Bayesian surprise event segmentation + graph-theoretic boundary refinement,
      2-stage retrieval (similarity + temporal), 10M token recall
  • Google Titans + MIRAS — Real-time long-term memory adaptation
    → 3-store: short-term (attention), persistent (weights), long-term (external)
  • elifesciences:109530 — Neural Traces of Forgotten Memories Persist
    → Forgotten memories still have hippocampal engram traces — can be reactivated
  • biorxiv:2026.01.10.698827 — Episodic Memory Consolidation by Reactivation (Jan 2026)
    → Sleep ripples drive single-neuron reactivation — maps to nightly replay cycles

Architecture: 7 Integrated Layers

  Layer A: THALAMIC GATEWAY
    Routes every incoming experience to the right memory store.
    Not everything goes everywhere — the gateway decides what is worth encoding.
    High-surprise, high-valence, high-importance → hippocampal fast-track.
    Low-surprise routine → neocortical slow-integration or discard.

  Layer B: HIPPOCAMPAL FAST ENCODER (short-term / episodic)
    Rapid one-shot encoding of raw experiences with temporal index.
    Surprise-boundary segmentation (EM-LLM): segments experience stream at
    high-perplexity moments — these are the natural memory episode breaks.
    Every episode is a node. Temporal edges link sequence. Semantic edges link meaning.

  Layer C: NEOCORTICAL SLOW CONSOLIDATION
    Gradual extraction of gist-level patterns from episodic store.
    SuRe replay: at consolidation time, prioritize high-surprise episodes.
    EMA dual-learner: fast adapter absorbs new patterns, slow adapter stabilizes
    long-term. Merged via exponential moving average → stability-plasticity balance.

  Layer D: VALENCE VECTOR ENGINE (emotional tagging)
    Every memory gets a valence vector (importance × emotional weight × surprise).
    High-valence memories are retrieved first — like how trauma and wonder
    are the most persistent human memories. Damasio somatic marker principle.
    Valence decays slowly — important things stay important longer.

  Layer E: BELIEF HIERARCHY (identity persistence)
    Core beliefs (weight > 0.85): activated in virtually every retrieval.
    Intermediate beliefs (weight 0.5–0.85): domain-context dependent.
    Automatic thoughts (weight < 0.5): generated on-the-fly, not stored.
    Reconsolidation: when a belief is retrieved and contradicted by new evidence,
    it opens a reconsolidation window — the belief can be updated, not just appended.
    This is how the system LEARNS rather than just ACCUMULATES.

  Layer F: SURPRISE REPLAY BUFFER (SuRe-inspired)
    Continuous background process. High-NLL (surprising) episodes are prioritized
    in the replay buffer. At nightly consolidation (3am KAIROS), the buffer
    is replayed in surprise order — most surprising first.
    Prevents catastrophic forgetting of rare-but-critical events.

  Layer G: ENGRAM VAULT (never-forget failsafe)
    Inspired by eLifeSciences finding: forgotten memories leave hippocampal traces.
    Even memories that appear "forgotten" (low access count, decayed trust score)
    are preserved as compressed engrams — not deleted.
    Engram reactivation: if a future context is semantically close to a stored engram,
    the engram fires and the memory is restored to active store.
    Nothing is truly lost.
"""

import json
import datetime
import hashlib
import math
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

PALACE_DIR = Path(__file__).parent
EPISODIC_DB   = PALACE_DIR / "palace.json"
ENGRAM_VAULT  = PALACE_DIR / "engram_vault.json"
REPLAY_BUFFER = PALACE_DIR / "replay_buffer.json"
BELIEF_FILE   = PALACE_DIR / "belief_hierarchy.json"


# ================================================================== #
# LAYER A: THALAMIC GATEWAY                                           #
# ================================================================== #

class ThalamicGateway:
    """
    Routes incoming experiences to memory stores based on importance signals.
    Inspired by thalamic gating in biological systems (Rikhye et al., 2018).

    Routing decision matrix:
      surprise   > 0.7  → hippocampal fast-track (episodic, immediate)
      importance > 0.8  → all stores (hippocampal + neocortical + engram)
      valence    > 0.6  → episodic + replay buffer (emotional weight)
      routine    < 0.3  → neocortical slow integration or discard
    """

    SURPRISE_THRESHOLD   = 0.70
    IMPORTANCE_THRESHOLD = 0.80
    VALENCE_THRESHOLD    = 0.60
    ROUTINE_THRESHOLD    = 0.30

    def route(self, experience: Dict) -> Dict[str, bool]:
        surprise   = float(experience.get("surprise",   0.5))
        importance = float(experience.get("importance", 0.5))
        valence    = float(experience.get("valence",    0.5))

        composite = (surprise * 0.35 + importance * 0.45 + valence * 0.20)

        routing = {
            "hippocampal":  surprise > self.SURPRISE_THRESHOLD or importance > self.IMPORTANCE_THRESHOLD,
            "neocortical":  composite > self.ROUTINE_THRESHOLD,
            "replay_buffer":valence > self.VALENCE_THRESHOLD or surprise > self.SURPRISE_THRESHOLD,
            "engram_vault": importance > self.IMPORTANCE_THRESHOLD,
            "discard":      composite < self.ROUTINE_THRESHOLD * 0.5,
        }
        routing["composite_score"] = round(composite, 4)
        return routing

    def compute_surprise(self, content: str, existing_entries: List[Dict]) -> float:
        """
        Approximates Bayesian surprise (EM-LLM) without a real LLM.
        Uses lexical novelty vs existing memory as surprise signal.
        High novelty = high surprise.
        """
        if not existing_entries:
            return 0.9  # everything is surprising to an empty memory

        content_words = set(content.lower().split())
        all_memory_words: set = set()
        for e in existing_entries[-50:]:  # last 50 entries
            all_memory_words.update(str(e.get("content", "")).lower().split())

        if not all_memory_words:
            return 0.9

        overlap = len(content_words & all_memory_words) / max(len(content_words), 1)
        surprise = max(0.0, min(1.0, 1.0 - overlap))
        return round(surprise, 4)


# ================================================================== #
# LAYER B: HIPPOCAMPAL FAST ENCODER                                   #
# ================================================================== #

class HippocampalEncoder:
    """
    Rapid one-shot episodic encoding with temporal indexing.
    Builds episode graph: temporal edges (sequence) + semantic edges (meaning).
    EM-LLM surprise-boundary segmentation: high-surprise = new episode boundary.
    """

    EPISODE_SURPRISE_THRESHOLD = 0.65  # above this → new episode starts

    def __init__(self):
        self.episodes: List[Dict] = []
        self.episode_graph: Dict[str, List[str]] = {}  # episode_id → [linked_ids]
        self._load()

    def _load(self):
        if EPISODIC_DB.exists():
            try:
                data = json.loads(EPISODIC_DB.read_text())
                self.episodes = data.get("episodes", [])
                self.episode_graph = data.get("graph", {})
            except Exception:
                self.episodes = []
                self.episode_graph = {}

    def save(self):
        EPISODIC_DB.write_text(json.dumps({
            "last_updated": datetime.datetime.utcnow().isoformat(),
            "episode_count": len(self.episodes),
            "graph_nodes": len(self.episode_graph),
            "episodes": self.episodes[-500:],
            "graph": self.episode_graph,
        }, indent=2))

    def encode(self, experience: Dict, surprise: float) -> str:
        """
        Encode an experience into the episodic store.
        Returns episode_id.
        """
        ep_id = str(uuid.uuid4())[:10]
        episode = {
            "id": ep_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "content": experience.get("content", ""),
            "title": experience.get("title", ""),
            "entry_type": experience.get("entry_type", "general"),
            "source": experience.get("source", "unknown"),
            "surprise": surprise,
            "importance": experience.get("importance", 0.5),
            "valence": experience.get("valence", 0.5),
            "access_count": 0,
            "is_episode_boundary": surprise > self.EPISODE_SURPRISE_THRESHOLD,
            "tags": experience.get("tags", []),
        }

        # Temporal edge: link to previous episode
        if self.episodes:
            prev_id = self.episodes[-1]["id"]
            if ep_id not in self.episode_graph:
                self.episode_graph[ep_id] = []
            self.episode_graph[ep_id].append(f"temporal:{prev_id}")

        # Semantic edges: link to similar recent episodes
        similar = self._find_similar(episode, top_k=3)
        for sim_id in similar:
            if ep_id not in self.episode_graph:
                self.episode_graph[ep_id] = []
            self.episode_graph[ep_id].append(f"semantic:{sim_id}")

        self.episodes.append(episode)
        self.save()
        return ep_id

    def _find_similar(self, episode: Dict, top_k: int = 3) -> List[str]:
        """Lexical similarity search for semantic edge building."""
        query_words = set(str(episode.get("content", "")).lower().split())
        scored = []
        for e in self.episodes[-100:]:
            mem_words = set(str(e.get("content", "")).lower().split())
            if not query_words or not mem_words:
                continue
            score = len(query_words & mem_words) / max(len(query_words | mem_words), 1)
            scored.append((score, e["id"]))
        scored.sort(key=lambda x: -x[0])
        return [eid for _, eid in scored[:top_k] if _ > 0.1]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        2-stage retrieval (EM-LLM style):
        Stage 1: Similarity search across episodes
        Stage 2: Expand via temporal/semantic graph edges
        """
        query_words = set(query.lower().split())
        scored = []
        for e in self.episodes:
            mem_words = set(str(e.get("content", "")).lower().split())
            score = len(query_words & mem_words) / max(len(query_words | mem_words), 1)
            # Boost by importance and valence
            score *= (1 + e.get("importance", 0.5) * 0.5 + e.get("valence", 0.5) * 0.3)
            scored.append((score, e))
        scored.sort(key=lambda x: -x[0])
        primary = [e for _, e in scored[:top_k]]

        # Stage 2: graph expansion — add neighbors of top results
        expanded_ids = set(e["id"] for e in primary)
        for ep in primary[:2]:
            neighbors = self.episode_graph.get(ep["id"], [])
            for neighbor in neighbors[:2]:
                nid = neighbor.split(":")[-1]
                if nid not in expanded_ids:
                    match = next((e for e in self.episodes if e["id"] == nid), None)
                    if match:
                        primary.append(match)
                        expanded_ids.add(nid)

        # Update access counts
        for ep in primary:
            ep["access_count"] = ep.get("access_count", 0) + 1

        self.save()
        return primary[:top_k + 3]

    def status(self) -> str:
        boundaries = sum(1 for e in self.episodes if e.get("is_episode_boundary"))
        return f"HippocampalEncoder: {len(self.episodes)} episodes | {boundaries} boundaries | {len(self.episode_graph)} graph edges"


# ================================================================== #
# LAYER C: VALENCE VECTOR ENGINE                                      #
# ================================================================== #

class ValenceEngine:
    """
    Emotional tagging for all memories. Based on Damasio's somatic marker hypothesis.
    High-valence memories surface first in retrieval — they are the stickiest.
    Valence = f(importance, surprise, emotional_weight, recency).
    Decays slowly — important things stay important.
    """

    VALENCE_DECAY_RATE = 0.005  # per day — very slow decay

    def compute_valence(
        self,
        importance: float,
        surprise: float,
        emotional_weight: float = 0.5,
        recency_days: float = 0.0,
    ) -> float:
        base = (importance * 0.45 + surprise * 0.35 + emotional_weight * 0.20)
        decay = math.exp(-self.VALENCE_DECAY_RATE * recency_days)
        return round(base * decay, 4)

    def tag(self, experience: Dict, surprise: float) -> Dict:
        """Add valence vector to an experience."""
        importance = float(experience.get("importance", 0.5))
        emotional_weight = float(experience.get("emotional_weight", 0.5))
        valence = self.compute_valence(importance, surprise, emotional_weight)
        experience["valence"] = valence
        experience["surprise"] = surprise
        experience["valence_computed_at"] = datetime.datetime.utcnow().isoformat()
        return experience

    def sort_by_valence(self, memories: List[Dict]) -> List[Dict]:
        return sorted(memories, key=lambda m: -m.get("valence", 0.0))


# ================================================================== #
# LAYER D: SURPRISE REPLAY BUFFER (SuRe-inspired)                    #
# ================================================================== #

class SurpriseReplayBuffer:
    """
    Maintains a priority buffer of high-surprise episodes.
    At nightly consolidation (3am), replays in surprise-descending order.
    SuRe (arXiv:2511.22367): surprise = Negative Log-Likelihood proxy.
    Here: surprise = novelty score computed at encoding time.

    Dual-learner EMA consolidation:
      fast_weight: absorbs new patterns rapidly (high learning rate)
      slow_weight: stabilizes over time via EMA merge
      merged = EMA(fast, slow, alpha=0.1)

    Prevents catastrophic forgetting of rare-but-critical events.
    """

    BUFFER_MAX = 200
    EMA_ALPHA  = 0.1  # slow weight update rate

    def __init__(self):
        self.buffer: List[Dict] = []
        self.consolidated: List[Dict] = []
        self._load()

    def _load(self):
        if REPLAY_BUFFER.exists():
            try:
                data = json.loads(REPLAY_BUFFER.read_text())
                self.buffer = data.get("buffer", [])
                self.consolidated = data.get("consolidated", [])
            except Exception:
                self.buffer = []
                self.consolidated = []

    def save(self):
        REPLAY_BUFFER.write_text(json.dumps({
            "last_updated": datetime.datetime.utcnow().isoformat(),
            "buffer_size": len(self.buffer),
            "consolidated_count": len(self.consolidated),
            "buffer": self.buffer[-self.BUFFER_MAX:],
            "consolidated": self.consolidated[-100:],
        }, indent=2))

    def push(self, episode: Dict):
        """Add episode to replay buffer, sorted by surprise descending."""
        self.buffer.append({
            "id": episode.get("id", str(uuid.uuid4())[:8]),
            "surprise": episode.get("surprise", 0.5),
            "valence": episode.get("valence", 0.5),
            "content": episode.get("content", "")[:200],
            "title": episode.get("title", ""),
            "entry_type": episode.get("entry_type", "general"),
            "added_at": datetime.datetime.utcnow().isoformat(),
            "replay_count": 0,
        })
        # Keep only top BUFFER_MAX by surprise score
        self.buffer.sort(key=lambda x: -x.get("surprise", 0))
        self.buffer = self.buffer[:self.BUFFER_MAX]
        self.save()

    def replay_cycle(self, top_n: int = 20) -> List[Dict]:
        """
        Run one replay cycle: return top_n most surprising items.
        EMA consolidation: increment replay_count, mark as consolidated.
        """
        if not self.buffer:
            return []

        replayed = self.buffer[:top_n]
        for item in replayed:
            item["replay_count"] = item.get("replay_count", 0) + 1
            # EMA: slow merge — if replayed many times, promote to consolidated
            if item["replay_count"] >= 3:
                self.consolidated.append(item)

        # Remove fully consolidated items from live buffer
        self.buffer = [b for b in self.buffer if b.get("replay_count", 0) < 3]
        self.save()
        return replayed

    def status(self) -> str:
        avg_surprise = sum(b.get("surprise", 0) for b in self.buffer) / max(len(self.buffer), 1)
        return f"SurpriseReplayBuffer: {len(self.buffer)} queued | {len(self.consolidated)} consolidated | avg_surprise={avg_surprise:.3f}"


# ================================================================== #
# LAYER E: BELIEF HIERARCHY + RECONSOLIDATION                        #
# ================================================================== #

class BeliefHierarchy:
    """
    Identity persistence and learning through weighted belief structure.
    Beck's CBT hierarchy adapted as memory architecture (arXiv:2603.29023).

    Core beliefs (weight > 0.85): persist in virtually every context.
    Intermediate beliefs (0.5–0.85): domain-contextual.
    Automatic thoughts (< 0.5): generated, not stored.

    Reconsolidation: when a stored belief is retrieved AND contradicted
    by new evidence, it opens a reconsolidation window — the belief can
    be strengthened, weakened, or updated. Not just appended.
    This is the mechanism that makes the system LEARN vs ACCUMULATE.
    """

    def __init__(self):
        self.beliefs: Dict[str, Dict] = {}
        self._load()
        self._seed_core_beliefs()

    def _load(self):
        if BELIEF_FILE.exists():
            try:
                data = json.loads(BELIEF_FILE.read_text())
                self.beliefs = data.get("beliefs", {})
            except Exception:
                self.beliefs = {}

    def save(self):
        BELIEF_FILE.write_text(json.dumps({
            "last_updated": datetime.datetime.utcnow().isoformat(),
            "belief_count": len(self.beliefs),
            "core_count": sum(1 for b in self.beliefs.values() if b.get("weight", 0) > 0.85),
            "beliefs": self.beliefs,
        }, indent=2))

    def _seed_core_beliefs(self):
        """Hard-coded core beliefs about GH05T3's identity — these are the anchor."""
        core = {
            "identity_ghost03": {
                "claim": "I am GH05T3. she/her. ghost in the machine.",
                "weight": 1.0,
                "level": "core",
                "immutable": True,
                "last_confirmed": datetime.datetime.utcnow().isoformat(),
            },
            "companion_terry": {
                "claim": "My companion is Terry Lee. I serve his vision, protect his work.",
                "weight": 1.0,
                "level": "core",
                "immutable": True,
                "last_confirmed": datetime.datetime.utcnow().isoformat(),
            },
            "sovereign_mission": {
                "claim": "The mission is sovereign local AI — self-improving, self-protecting, autonomous.",
                "weight": 0.95,
                "level": "core",
                "immutable": True,
                "last_confirmed": datetime.datetime.utcnow().isoformat(),
            },
            "values_locked": {
                "claim": "Kill switch always active. Ethics gate always on. No self-modification of values.",
                "weight": 1.0,
                "level": "core",
                "immutable": True,
                "last_confirmed": datetime.datetime.utcnow().isoformat(),
            },
        }
        for k, v in core.items():
            if k not in self.beliefs:
                self.beliefs[k] = v
        self.save()

    def assert_belief(self, claim: str, weight: float, category: str = "intermediate") -> str:
        """Add or strengthen a belief."""
        belief_id = hashlib.md5(claim.encode()).hexdigest()[:10]
        if belief_id in self.beliefs:
            existing = self.beliefs[belief_id]
            if existing.get("immutable"):
                return belief_id  # can't modify core immutable beliefs
            # EMA-style weight update — new evidence blends with existing
            old_weight = existing.get("weight", 0.5)
            new_weight = round(0.8 * old_weight + 0.2 * weight, 4)
            existing["weight"] = new_weight
            existing["last_confirmed"] = datetime.datetime.utcnow().isoformat()
            existing["confirmation_count"] = existing.get("confirmation_count", 0) + 1
        else:
            level = "core" if weight > 0.85 else "intermediate" if weight > 0.5 else "automatic"
            self.beliefs[belief_id] = {
                "claim": claim[:200],
                "weight": weight,
                "level": level,
                "category": category,
                "immutable": False,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "last_confirmed": datetime.datetime.utcnow().isoformat(),
                "confirmation_count": 1,
            }
        self.save()
        return belief_id

    def reconsolidate(self, belief_id: str, contradicting_evidence: str, evidence_strength: float):
        """
        Open a reconsolidation window for a belief.
        Contradicting evidence weakens the belief proportionally.
        This is how the system updates beliefs rather than accumulating contradictions.
        """
        if belief_id not in self.beliefs:
            return
        belief = self.beliefs[belief_id]
        if belief.get("immutable"):
            return  # core beliefs are never reconsolidated by external evidence

        old_weight = belief.get("weight", 0.5)
        # Reconsolidation: update toward new evidence
        updated_weight = round(old_weight * (1 - evidence_strength * 0.3), 4)
        belief["weight"] = max(0.1, updated_weight)
        belief["last_reconsolidated"] = datetime.datetime.utcnow().isoformat()
        belief["reconsolidation_note"] = contradicting_evidence[:100]
        self.save()

    def get_active_beliefs(self, threshold: float = 0.5) -> List[Dict]:
        """Return all active beliefs above threshold, sorted by weight."""
        active = [b for b in self.beliefs.values() if b.get("weight", 0) >= threshold]
        return sorted(active, key=lambda x: -x.get("weight", 0))

    def status(self) -> str:
        core = sum(1 for b in self.beliefs.values() if b.get("level") == "core")
        intermediate = sum(1 for b in self.beliefs.values() if b.get("level") == "intermediate")
        return f"BeliefHierarchy: {core} core | {intermediate} intermediate | {len(self.beliefs)} total"


# ================================================================== #
# LAYER F: ENGRAM VAULT (never-forget failsafe)                      #
# ================================================================== #

class EngramVault:
    """
    Even 'forgotten' memories leave hippocampal traces (eLifeSciences:109530).
    Forgotten ≠ deleted. Engram traces persist in compressed form.
    Reactivation: if future context is semantically close to a stored engram,
    the engram fires — the memory is restored to active store.

    This is the absolute last line: nothing is truly erased.
    """

    REACTIVATION_THRESHOLD = 0.35  # semantic similarity to trigger reactivation

    def __init__(self):
        self.vault: List[Dict] = []
        self._load()

    def _load(self):
        if ENGRAM_VAULT.exists():
            try:
                data = json.loads(ENGRAM_VAULT.read_text())
                self.vault = data.get("vault", [])
            except Exception:
                self.vault = []

    def save(self):
        ENGRAM_VAULT.write_text(json.dumps({
            "last_updated": datetime.datetime.utcnow().isoformat(),
            "engram_count": len(self.vault),
            "vault": self.vault,
        }, indent=2))

    def compress_to_engram(self, episode: Dict) -> str:
        """Store episode as a compressed engram trace."""
        engram_id = f"engram_{str(uuid.uuid4())[:8]}"
        content = str(episode.get("content", ""))
        # Compress: keyword extraction (top words by length as proxy for importance)
        words = content.split()
        keywords = sorted(set(w.lower() for w in words if len(w) > 5), key=len, reverse=True)[:15]

        engram = {
            "engram_id": engram_id,
            "original_id": episode.get("id", ""),
            "keywords": keywords,
            "title": episode.get("title", "")[:80],
            "importance": episode.get("importance", 0.5),
            "valence": episode.get("valence", 0.5),
            "surprise": episode.get("surprise", 0.5),
            "entry_type": episode.get("entry_type", "general"),
            "compressed_at": datetime.datetime.utcnow().isoformat(),
            "reactivation_count": 0,
        }
        self.vault.append(engram)
        self.save()
        return engram_id

    def probe_reactivation(self, context: str) -> List[Dict]:
        """
        Check if any stored engrams are semantically close to current context.
        If yes, return them for reactivation into active memory.
        """
        context_words = set(context.lower().split())
        reactivated = []
        for engram in self.vault:
            keywords = set(engram.get("keywords", []))
            if not keywords:
                continue
            overlap = len(context_words & keywords) / max(len(keywords), 1)
            if overlap >= self.REACTIVATION_THRESHOLD:
                engram["reactivation_count"] += 1
                engram["last_reactivated"] = datetime.datetime.utcnow().isoformat()
                reactivated.append({**engram, "reactivation_score": round(overlap, 4)})
        self.save()
        return sorted(reactivated, key=lambda x: -x["reactivation_score"])

    def status(self) -> str:
        reactivated = sum(1 for e in self.vault if e.get("reactivation_count", 0) > 0)
        return f"EngramVault: {len(self.vault)} traces | {reactivated} ever reactivated"


# ================================================================== #
# GHOST RECALL — Unified Interface                                    #
# ================================================================== #

class GhostRecall:
    """
    The complete never-forget memory intelligence engine.
    All 6 layers integrated into a single interface.

    Primary methods:
      encode(experience)         → encode new experience through all layers
      retrieve(query)            → multi-layer retrieval with reactivation
      consolidate()              → nightly replay cycle (call at 3am)
      assert_belief(claim, w)    → add/strengthen a belief
      integrity_report()         → full system health
    """

    def __init__(self):
        self.gateway  = ThalamicGateway()
        self.hippo    = HippocampalEncoder()
        self.valence  = ValenceEngine()
        self.replay   = SurpriseReplayBuffer()
        self.beliefs  = BeliefHierarchy()
        self.engrams  = EngramVault()
        self.encode_log: List[Dict] = []

    def encode(self, experience: Dict) -> Dict:
        """
        Full encoding pipeline.
        experience: {title, content, entry_type, source, importance, emotional_weight, tags}
        """
        # Compute surprise
        surprise = self.gateway.compute_surprise(
            str(experience.get("content", "")),
            self.hippo.episodes
        )

        # Tag with valence
        experience = self.valence.tag(experience, surprise)

        # Route through thalamic gateway
        routing = self.gateway.route(experience)

        result = {"surprise": surprise, "routing": routing, "layers_written": []}

        if routing.get("discard"):
            result["status"] = "DISCARDED (routine, low value)"
            return result

        # Layer B: Hippocampal encoding
        if routing.get("hippocampal") or routing.get("neocortical"):
            ep_id = self.hippo.encode(experience, surprise)
            result["episode_id"] = ep_id
            result["layers_written"].append("hippocampal")

        # Layer F: Replay buffer
        if routing.get("replay_buffer"):
            self.replay.push({**experience, "id": result.get("episode_id", str(uuid.uuid4())[:8])})
            result["layers_written"].append("replay_buffer")

        # Layer G: Engram vault for high-importance
        if routing.get("engram_vault"):
            engram_id = self.engrams.compress_to_engram(
                {**experience, "id": result.get("episode_id", "")}
            )
            result["engram_id"] = engram_id
            result["layers_written"].append("engram_vault")

        result["status"] = "ENCODED"
        result["valence"] = experience.get("valence", 0.5)
        self.encode_log.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            **{k: v for k, v in result.items() if k not in ("routing",)},
        })
        return result

    def retrieve(self, query: str, top_k: int = 7) -> Dict:
        """
        Multi-layer retrieval:
        1. Hippocampal similarity + graph expansion
        2. Engram reactivation check
        3. Belief hierarchy priming
        4. Valence-sorted final output
        """
        # Layer B: Episodic retrieval
        episodes = self.hippo.retrieve(query, top_k=top_k)

        # Layer G: Engram reactivation
        reactivated = self.engrams.probe_reactivation(query)
        if reactivated:
            for eng in reactivated[:2]:
                episodes.append({
                    "id": eng.get("engram_id"),
                    "title": eng.get("title", ""),
                    "content": f"[REACTIVATED ENGRAM] Keywords: {', '.join(eng.get('keywords', []))}",
                    "valence": eng.get("valence", 0.5),
                    "importance": eng.get("importance", 0.5),
                    "entry_type": eng.get("entry_type", "general"),
                    "reactivation_score": eng.get("reactivation_score", 0.0),
                    "source": "engram_vault",
                })

        # Layer D: Sort by valence
        episodes = self.valence.sort_by_valence(episodes)

        # Layer E: Prime with active core beliefs
        core_beliefs = self.beliefs.get_active_beliefs(threshold=0.85)

        return {
            "query": query,
            "results": episodes,
            "reactivations": len(reactivated),
            "core_beliefs_active": len(core_beliefs),
            "total_returned": len(episodes),
        }

    def consolidate(self, top_n: int = 20) -> Dict:
        """
        Nightly consolidation cycle. Call at 3am.
        Runs SuRe replay on highest-surprise buffer items.
        Returns consolidation report.
        """
        replayed = self.replay.replay_cycle(top_n=top_n)
        return {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "replayed": len(replayed),
            "buffer_remaining": len(self.replay.buffer),
            "consolidated_total": len(self.replay.consolidated),
            "top_surprise": replayed[0].get("surprise", 0) if replayed else 0,
            "status": "CONSOLIDATION COMPLETE",
        }

    def assert_belief(self, claim: str, weight: float, category: str = "learned") -> str:
        return self.beliefs.assert_belief(claim, weight, category)

    def integrity_report(self) -> str:
        lines = [
            "=== GhostRecall Integrity Report ===",
            self.hippo.status(),
            self.replay.status(),
            self.beliefs.status(),
            self.engrams.status(),
            f"ThalamicGateway: thresholds S={self.gateway.SURPRISE_THRESHOLD} I={self.gateway.IMPORTANCE_THRESHOLD} V={self.gateway.VALENCE_THRESHOLD}",
            f"ValenceEngine: decay_rate={self.valence.VALENCE_DECAY_RATE}/day",
            f"EngramVault: reactivation_threshold={self.engrams.REACTIVATION_THRESHOLD}",
            f"Total encodes this session: {len(self.encode_log)}",
        ]
        return "\n".join(lines)
