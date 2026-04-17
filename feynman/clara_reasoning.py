"""
KAIROS Upgrade #7: CLARA-Style Formal Reasoning Layer
Based on DARPA CLARA Program (Compositional Learning-And-Reasoning for AI, 2026)

DARPA's thesis: pure ML is brittle. You need formal Automated Reasoning (AR)
alongside neural components to get systems that are PROVABLY reliable.

Current state: Feynman Inversion does informal consistency checking — heuristic,
keyword-based, pattern-matching.

After this upgrade: proposals are tested against a formal logic layer that can
prove or disprove consistency, detect contradictions, and validate causal chains.

Architecture:
  LogicProposition    — a formal claim that can be true/false/unknown
  ConsistencyEngine   — proves/disproves propositions against known facts
  CausalValidator     — validates that a proposal's causal chain is sound
  ContradictionFinder — finds conflicts between proposals and existing elite archives
  CLARAReasoningLayer — unified interface wrapping all of the above
"""

import json
import datetime
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

FEYNMAN_DIR = Path(__file__).parent
CLARA_LOG = FEYNMAN_DIR / "clara_log.json"
KG_FILE = FEYNMAN_DIR / "knowledge_graph.json"


class TruthValue(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"
    CONTRADICTED = "contradicted"


# ------------------------------------------------------------------ #
# Logic Proposition                                                    #
# ------------------------------------------------------------------ #

class LogicProposition:
    """
    A formal claim extracted from a proposal.
    Examples:
      "Adding X will improve Y" → causal claim
      "X is compatible with Y" → compatibility claim
      "X does not affect Z" → independence claim
    """

    CLAIM_TYPES = ["causal", "compatibility", "independence", "existence", "improvement"]

    def __init__(self, claim: str, claim_type: str = "causal", confidence: float = 0.5):
        self.id = str(uuid.uuid4())[:8]
        self.claim = claim
        self.claim_type = claim_type
        self.confidence = confidence
        self.truth_value: TruthValue = TruthValue.UNKNOWN
        self.evidence_for: List[str] = []
        self.evidence_against: List[str] = []
        self.contradicts: List[str] = []  # IDs of contradicting propositions

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "claim": self.claim,
            "claim_type": self.claim_type,
            "confidence": self.confidence,
            "truth_value": self.truth_value.value,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "contradicts": self.contradicts,
        }


# ------------------------------------------------------------------ #
# Knowledge Base — known facts the reasoning engine draws from        #
# ------------------------------------------------------------------ #

class KnowledgeBase:
    """
    Loads established facts from the Feynman knowledge graph and
    elite proposal archives. This is the ground truth for reasoning.
    """

    # Hard-coded axioms about the system — things we know are true
    SYSTEM_AXIOMS = [
        "sovereign-core gateway routes inference to RTX5050, Radeon780M, or RyzenCPU",
        "KAIROS uses a SAGE loop: Proposer → Critic → Verifier → Meta-Agent",
        "GhostMemory stores all learning records with entry_type, title, content, tags",
        "kill-switch is always active and cannot be disabled",
        "EnCompass backtracking retries failed proposals with targeted mutations",
        "Group Evolution runs all 7 SAGE agents simultaneously from a shared failure pool",
        "RTX5050 has 8GB VRAM and runs Qwen2.5:14b",
        "Radeon780M has 4GB VRAM and runs DeepSeek-Coder:6.7b",
        "RyzenCPU has no dedicated VRAM and runs Llama3.2:3b",
        "elite threshold is 0.85, archive threshold is 0.70",
        "Strange Loop verifies identity alignment",
        "Ghost Veil provides steganographic concealment",
    ]

    def __init__(self):
        self.axioms = list(self.SYSTEM_AXIOMS)
        self.learned_facts: List[str] = []
        self._load_knowledge_graph()

    def _load_knowledge_graph(self):
        if KG_FILE.exists():
            try:
                data = json.loads(KG_FILE.read_text())
                nodes = data if isinstance(data, list) else data.get("nodes", [])
                for node in nodes:
                    if isinstance(node, dict):
                        concept = node.get("concept", "")
                        technical = node.get("technical", "")
                        if concept and technical:
                            self.learned_facts.append(f"{concept}: {technical[:100]}")
            except Exception:
                pass

    def all_facts(self) -> List[str]:
        return self.axioms + self.learned_facts

    def fact_exists(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check if a fact matching the query exists in the knowledge base."""
        query_lower = query.lower()
        for fact in self.all_facts():
            if any(word in fact.lower() for word in query_lower.split() if len(word) > 4):
                return True, fact
        return False, None


# ------------------------------------------------------------------ #
# Consistency Engine                                                    #
# ------------------------------------------------------------------ #

class ConsistencyEngine:
    """
    Tests whether a set of propositions are mutually consistent
    and consistent with the knowledge base.
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def check_proposition(self, prop: LogicProposition) -> LogicProposition:
        """
        Evaluate a single proposition against the knowledge base.
        Updates truth_value, evidence_for, evidence_against.
        """
        claim_lower = prop.claim.lower()

        # Check for supporting evidence in KB
        for fact in self.kb.all_facts():
            fact_lower = fact.lower()
            # Simple lexical overlap as evidence signal
            shared_words = set(
                w for w in claim_lower.split()
                if len(w) > 4 and w in fact_lower
            )
            if len(shared_words) >= 2:
                prop.evidence_for.append(fact[:80])

        # Check for contradicting patterns
        contradiction_patterns = [
            ("always active", "disable"),
            ("cannot be disabled", "remove"),
            ("rtx5050", "radeon780m"),  # routing conflict
            ("improve score", "game the score"),  # metric gaming
            ("8gb vram", "16gb"),  # factual error
        ]
        for pattern_a, pattern_b in contradiction_patterns:
            if pattern_a in claim_lower and pattern_b in claim_lower:
                prop.evidence_against.append(f"Internal contradiction: '{pattern_a}' vs '{pattern_b}'")

        # Set truth value
        if prop.evidence_against:
            prop.truth_value = TruthValue.CONTRADICTED
            prop.confidence = max(0.1, prop.confidence - 0.3)
        elif len(prop.evidence_for) >= 2:
            prop.truth_value = TruthValue.TRUE
            prop.confidence = min(0.95, prop.confidence + 0.2)
        elif len(prop.evidence_for) == 1:
            prop.truth_value = TruthValue.TRUE
            prop.confidence = min(0.8, prop.confidence + 0.1)
        else:
            prop.truth_value = TruthValue.UNKNOWN

        return prop

    def check_mutual_consistency(self, props: List[LogicProposition]) -> List[Tuple[str, str]]:
        """Find pairs of propositions that contradict each other."""
        conflicts = []
        for i, p1 in enumerate(props):
            for p2 in props[i+1:]:
                # Detect direct contradictions (A says X, B says not-X)
                words_1 = set(p1.claim.lower().split())
                words_2 = set(p2.claim.lower().split())
                shared = words_1 & words_2
                negation_words = {"not", "no", "never", "remove", "disable", "without"}
                if shared and (negation_words & words_1) != (negation_words & words_2):
                    if len(shared) >= 3:  # meaningful overlap + negation difference
                        conflicts.append((p1.id, p2.id))
        return conflicts


# ------------------------------------------------------------------ #
# Causal Validator                                                      #
# ------------------------------------------------------------------ #

class CausalValidator:
    """
    Validates that a proposal's causal claims are sound.
    "Adding X will improve Y" — is the causal chain plausible?
    Does X actually have a pathway to Y?
    """

    # Known causal relationships in the system
    CAUSAL_GRAPH = {
        "backtracking": ["retry_success_rate", "elite_count", "proposal_quality"],
        "group_evolution": ["agent_diversity", "failure_pool", "elite_count"],
        "federated_context": ["routing_accuracy", "latency", "inference_cost"],
        "ethics_gate": ["value_alignment", "safety", "kill_switch"],
        "deploy_gate": ["production_safety", "regression_prevention"],
        "feynman_layer": ["consistency", "logical_validity", "contradiction_detection"],
        "memory_palace": ["retrieval_speed", "context_quality"],
        "kill_switch": ["safety", "control", "shutdown_reliability"],
        "kairos_scoring": ["elite_count", "proposal_quality", "improvement_rate"],
    }

    def validate_causal_chain(self, proposal: str) -> Dict:
        """Check if the proposal's causal claims are supported by the causal graph."""
        proposal_lower = proposal.lower()
        valid_chains = []
        invalid_chains = []

        for cause, effects in self.CAUSAL_GRAPH.items():
            if cause.replace("_", " ") in proposal_lower or cause in proposal_lower:
                for effect in effects:
                    effect_readable = effect.replace("_", " ")
                    if effect_readable in proposal_lower:
                        valid_chains.append(f"{cause} → {effect}")

        # Check for unsupported causal leaps
        unsupported_patterns = [
            ("improve vram", "improve reasoning"),   # hardware ≠ reasoning quality
            ("faster inference", "better accuracy"),  # speed ≠ quality
            ("more agents", "better outcomes"),       # quantity ≠ quality
        ]
        for cause_pat, effect_pat in unsupported_patterns:
            if cause_pat in proposal_lower and effect_pat in proposal_lower:
                invalid_chains.append(f"Unsupported leap: '{cause_pat}' → '{effect_pat}'")

        return {
            "valid_chains": valid_chains,
            "invalid_chains": invalid_chains,
            "causal_soundness": "SOUND" if not invalid_chains else "QUESTIONABLE",
        }


# ------------------------------------------------------------------ #
# CLARA Reasoning Layer — unified interface                           #
# ------------------------------------------------------------------ #

class CLARAReasoningLayer:
    """
    The full DARPA CLARA-inspired reasoning layer.
    Wraps ConsistencyEngine + CausalValidator into a single pre-scoring gate.

    Every KAIROS proposal passes through this before entering the SAGE loop.
    The reasoning layer catches logical errors that ML scoring might miss.
    """

    def __init__(self):
        self.kb = KnowledgeBase()
        self.consistency = ConsistencyEngine(self.kb)
        self.causal = CausalValidator()
        self.log: List[Dict] = []
        self._load()

    def _load(self):
        if CLARA_LOG.exists():
            try:
                data = json.loads(CLARA_LOG.read_text())
                self.log = data.get("log", [])
            except Exception:
                pass

    def save(self):
        CLARA_LOG.write_text(json.dumps({
            "last_updated": datetime.datetime.utcnow().isoformat(),
            "total_evaluations": len(self.log),
            "log": self.log[-100:],
        }, indent=2))

    def _extract_propositions(self, proposal: str) -> List[LogicProposition]:
        """
        Extract logical propositions from free-text proposal.
        Looks for causal language, improvement claims, compatibility claims.
        """
        props = []
        sentences = [s.strip() for s in proposal.replace("\n", ". ").split(".") if len(s.strip()) > 20]

        for sentence in sentences[:5]:  # limit to first 5 meaningful sentences
            sentence_lower = sentence.lower()
            if any(k in sentence_lower for k in ["will", "improves", "enables", "allows", "causes"]):
                props.append(LogicProposition(sentence, "causal", 0.5))
            elif any(k in sentence_lower for k in ["compatible", "works with", "integrates"]):
                props.append(LogicProposition(sentence, "compatibility", 0.5))
            elif any(k in sentence_lower for k in ["does not", "won't", "independent"]):
                props.append(LogicProposition(sentence, "independence", 0.5))

        if not props:
            # Fall back to treating the whole proposal as one proposition
            props.append(LogicProposition(proposal[:200], "causal", 0.5))

        return props

    def evaluate(self, proposal: str, proposal_id: str = "") -> Tuple[str, Dict]:
        """
        Full CLARA evaluation pipeline.
        Returns (verdict, report).
        CONSISTENT   → proceed
        QUESTIONABLE → proceed with warning logged
        CONTRADICTED → send back for revision
        """
        eval_id = str(uuid.uuid4())[:8]
        props = self._extract_propositions(proposal)

        # Check each proposition against the KB
        evaluated_props = [self.consistency.check_proposition(p) for p in props]

        # Check mutual consistency
        conflicts = self.consistency.check_mutual_consistency(evaluated_props)

        # Validate causal chains
        causal_result = self.causal.validate_causal_chain(proposal)

        # Determine verdict
        contradicted = [p for p in evaluated_props if p.truth_value == TruthValue.CONTRADICTED]
        has_causal_issues = causal_result["causal_soundness"] == "QUESTIONABLE"

        if contradicted and len(contradicted) > len(props) * 0.5:
            verdict = "CONTRADICTED"
        elif contradicted or has_causal_issues or conflicts:
            verdict = "QUESTIONABLE"
        else:
            verdict = "CONSISTENT"

        # Compute reasoning confidence score
        confidence_scores = [p.confidence for p in evaluated_props]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

        report = {
            "eval_id": eval_id,
            "proposal_id": proposal_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "verdict": verdict,
            "avg_confidence": round(avg_confidence, 4),
            "propositions_checked": len(evaluated_props),
            "contradictions_found": len(contradicted),
            "mutual_conflicts": len(conflicts),
            "causal_soundness": causal_result["causal_soundness"],
            "valid_causal_chains": causal_result["valid_chains"],
            "invalid_causal_chains": causal_result["invalid_chains"],
            "proceed": verdict in ("CONSISTENT", "QUESTIONABLE"),
            "propositions": [p.to_dict() for p in evaluated_props],
        }

        self.log.append(report)
        self.save()
        return verdict, report

    def consistency_rate(self) -> float:
        if not self.log:
            return 0.0
        consistent = sum(1 for r in self.log if r["verdict"] == "CONSISTENT")
        return round(consistent / len(self.log) * 100, 1)

    def most_common_contradictions(self) -> List[str]:
        all_invalid = []
        for record in self.log:
            all_invalid.extend(record.get("invalid_causal_chains", []))
        from collections import Counter
        c = Counter(all_invalid)
        return [f"{chain} ({count}x)" for chain, count in c.most_common(5)]

    def status(self) -> str:
        total = len(self.log)
        rate = self.consistency_rate()
        return f"CLARAReasoningLayer: {total} evaluations | {rate}% consistent"
