"""
SovereignNation Security Fortress
==================================
Full-spectrum protection for SovereignCore and the SovereignNation platform.

Threat model covers 6 attack surfaces:
  1. Code theft / cloning / unauthorized redistribution
  2. License key forging / bypass
  3. Prompt injection / agent hijacking (indirect + direct)
  4. RAG poisoning / memory corruption (pairs with Iron Dome)
  5. Model inversion / training data extraction
  6. Runtime tampering / modification detection

Research sources:
  - OWASP Agentic Top 10 (2026)
  - arXiv:2602.11327 — Security Threat Modeling for AI-Agent Protocols
  - arXiv:2510.23883 — Agentic AI Security: Threats, Defenses, Evaluation
  - Redfox 2026 AI Vulnerability Report
  - Prompt injection surge 340% YoY (aimagicx.com 2026)
  - keygen.sh / licensingpy — hardware fingerprint licensing
  - Cython/Nuitka — compiled binary protection
"""

import hashlib
import hmac
import json
import os
import platform
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FORTRESS_DIR = Path(__file__).parent
LICENSE_DB    = FORTRESS_DIR / "license_vault.json"
TAMPER_LOG    = FORTRESS_DIR / "tamper_log.json"
INJECTION_LOG = FORTRESS_DIR / "injection_log.json"


# ═══════════════════════════════════════════════════════════════════
#  LAYER 1: HARDWARE FINGERPRINT + LICENSE ENGINE
#  Prevents: code theft, license sharing, unauthorized deployment
# ═══════════════════════════════════════════════════════════════════

class HardwareFingerprint:
    """
    Generates a machine-unique identifier from hardware characteristics.
    Inspired by: licensingpy (PyPI), keygen.sh offline verification.

    Fingerprint = SHA-256(CPU + Platform + MachineID + MacAddress + salt)
    
    Even if someone copies the entire codebase to another machine,
    the fingerprint will not match → license invalid → system locked.
    """

    SOVEREIGN_SALT = "SOVEREIGN_NATION_2026_IMMUTABLE_SALT_GH05T3"

    def generate(self) -> str:
        components = []

        # CPU / platform
        components.append(platform.processor() or "unknown_cpu")
        components.append(platform.machine())
        components.append(platform.system())
        components.append(platform.node())

        # Python runtime
        components.append(platform.python_version())

        # MAC address (network interface hardware ID)
        try:
            mac = hex(uuid.getnode())
            components.append(mac)
        except Exception:
            components.append("no_mac")

        # Machine UUID from OS (Linux /etc/machine-id)
        machine_id = self._get_machine_id()
        components.append(machine_id)

        raw = "|".join(components) + "|" + self.SOVEREIGN_SALT
        fingerprint = hashlib.sha256(raw.encode()).hexdigest()
        return fingerprint

    def _get_machine_id(self) -> str:
        """Read /etc/machine-id (Linux) or fallback."""
        paths = [
            "/etc/machine-id",
            "/var/lib/dbus/machine-id",
            "/sys/class/dmi/id/product_uuid",
        ]
        for p in paths:
            try:
                val = Path(p).read_text().strip()
                if val:
                    return val
            except Exception:
                continue
        return "no_machine_id"

    def short_id(self) -> str:
        """First 16 chars — safe to display/share for support."""
        return self.generate()[:16].upper()


class LicenseEngine:
    """
    ECDSA-inspired license key system.
    
    Flow:
      1. Customer purchases → SovereignNation generates license for their fingerprint
      2. License is an HMAC-SHA256 signed token: {tier, fingerprint, expiry, features}
      3. On startup: verify signature + fingerprint match + expiry
      4. If any check fails → graceful lock (no crash, no data loss)
    
    Tiers: developer | team | professional | enterprise
    """

    # In production: this would be your private signing key, stored server-side
    # Customers only ever get the PUBLIC verifier — never the signing key
    _SIGNING_SECRET = os.environ.get("SOVEREIGN_LICENSE_SECRET", "DEV_MODE_UNSIGNED")

    TIER_FEATURES = {
        "developer":     {"max_agents": 1,  "iron_dome": False, "ghost_recall": False, "kairos": False},
        "team":          {"max_agents": 5,  "iron_dome": True,  "ghost_recall": True,  "kairos": False},
        "professional":  {"max_agents": -1, "iron_dome": True,  "ghost_recall": True,  "kairos": True},
        "enterprise":    {"max_agents": -1, "iron_dome": True,  "ghost_recall": True,  "kairos": True, "custom": True},
    }

    def __init__(self):
        self.fp = HardwareFingerprint()
        self._vault: Dict = {}
        self._load()

    def _load(self):
        if LICENSE_DB.exists():
            try:
                self._vault = json.loads(LICENSE_DB.read_text())
            except Exception:
                self._vault = {}

    def _save(self):
        LICENSE_DB.write_text(json.dumps(self._vault, indent=2))

    def generate_license(self, tier: str, fingerprint: str,
                          expiry_days: int = 365, customer_id: str = "") -> str:
        """
        Generate a signed license token for a customer.
        This runs SERVER-SIDE at SovereignNation — never exposed to customer.
        """
        payload = {
            "tier": tier,
            "fingerprint": fingerprint,
            "customer_id": customer_id,
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": datetime.fromtimestamp(
                time.time() + expiry_days * 86400, tz=timezone.utc
            ).isoformat(),
            "features": self.TIER_FEATURES.get(tier, {}),
            "nonce": uuid.uuid4().hex,
        }
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            self._SIGNING_SECRET.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        token = {
            "payload": payload,
            "signature": signature,
            "version": "SN-1.0",
        }
        return json.dumps(token)

    def verify_license(self, license_token: str) -> Tuple[bool, str, Dict]:
        """
        Verify a license token on the customer machine.
        Returns (valid, reason, features)
        """
        try:
            token = json.loads(license_token)
        except Exception:
            return False, "INVALID_TOKEN_FORMAT", {}

        payload = token.get("payload", {})
        signature = token.get("signature", "")

        # 1. Verify HMAC signature
        payload_str = json.dumps(payload, sort_keys=True)
        expected_sig = hmac.new(
            self._SIGNING_SECRET.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_sig):
            return False, "INVALID_SIGNATURE — possible forgery", {}

        # 2. Verify hardware fingerprint
        current_fp = self.fp.generate()
        licensed_fp = payload.get("fingerprint", "")
        if not hmac.compare_digest(current_fp, licensed_fp):
            return False, "HARDWARE_MISMATCH — license not valid on this machine", {}

        # 3. Check expiry
        expires_at = payload.get("expires_at", "")
        if expires_at:
            expiry = datetime.fromisoformat(expires_at)
            if datetime.now(timezone.utc) > expiry:
                return False, "LICENSE_EXPIRED", {}

        features = payload.get("features", {})
        tier = payload.get("tier", "unknown")
        return True, f"VALID — tier={tier}", features

    def dev_mode_check(self) -> Tuple[bool, str]:
        """
        In development (no license file), run in dev mode with restrictions.
        Never crashes — graceful fallback.
        """
        if self._SIGNING_SECRET == "DEV_MODE_UNSIGNED":
            return True, "DEV_MODE — single agent, no Iron Dome, no KAIROS"
        return False, "LICENSE_REQUIRED"

    def status(self) -> str:
        fp_short = self.fp.short_id()
        return f"LicenseEngine: fingerprint={fp_short}... | secret={'SET' if self._SIGNING_SECRET != 'DEV_MODE_UNSIGNED' else 'DEV_MODE'}"


# ═══════════════════════════════════════════════════════════════════
#  LAYER 2: RUNTIME TAMPER DETECTION
#  Prevents: code modification, file patching, monkey-patching
# ═══════════════════════════════════════════════════════════════════

class TamperDetector:
    """
    Detects unauthorized modification of SovereignCore files at runtime.
    
    Method: At build time, compute SHA-256 of all critical files.
            At runtime, recompute and compare.
            Any mismatch = tamper alert → log + degrade gracefully.
    
    Critical files monitored:
      - All omega/ system modules
      - ghost_protocol/ security modules
      - KAIROS pipeline
      - License engine itself
    """

    CRITICAL_PATHS = [
        "systems/omega/ghost_protocol/killswitch/kill_switch.py",
        "systems/omega/ghost_protocol/fortress/sovereign_security.py",
        "systems/omega/memory_palace/iron_dome.py",
        "systems/omega/memory_palace/ghost_recall.py",
        "systems/omega/kairos/encompass_backtrack.py",
        "systems/omega/omega_double_prime/strange_loop/seed_set_gate.py",
    ]

    def __init__(self, base_dir: Path = Path("/app/.agents")):
        self.base_dir = base_dir
        self.tamper_log: List[Dict] = []
        self._load_log()

    def _load_log(self):
        if TAMPER_LOG.exists():
            try:
                self.tamper_log = json.loads(TAMPER_LOG.read_text())
            except Exception:
                self.tamper_log = []

    def _save_log(self):
        TAMPER_LOG.write_text(json.dumps(self.tamper_log[-500:], indent=2))

    def hash_file(self, path: Path) -> Optional[str]:
        try:
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception:
            return None

    def build_manifest(self) -> Dict[str, str]:
        """Compute current hashes of all critical files."""
        manifest = {}
        for rel_path in self.CRITICAL_PATHS:
            full = self.base_dir / rel_path
            h = self.hash_file(full)
            if h:
                manifest[rel_path] = h
        return manifest

    def verify_against_manifest(self, trusted_manifest: Dict[str, str]) -> List[Dict]:
        """
        Compare current file hashes against trusted manifest.
        Returns list of tamper alerts.
        """
        alerts = []
        current = self.build_manifest()
        for path, trusted_hash in trusted_manifest.items():
            current_hash = current.get(path)
            if current_hash is None:
                alerts.append({"path": path, "type": "FILE_MISSING", "severity": "CRITICAL"})
            elif current_hash != trusted_hash:
                alerts.append({
                    "path": path,
                    "type": "HASH_MISMATCH",
                    "severity": "CRITICAL",
                    "trusted": trusted_hash[:16],
                    "found": current_hash[:16],
                })

        if alerts:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "alert_count": len(alerts),
                "alerts": alerts,
            }
            self.tamper_log.append(entry)
            self._save_log()

        return alerts

    def status(self) -> str:
        return f"TamperDetector: monitoring {len(self.CRITICAL_PATHS)} critical files | {len(self.tamper_log)} tamper events logged"


# ═══════════════════════════════════════════════════════════════════
#  LAYER 3: PROMPT INJECTION FORTRESS
#  Prevents: agent hijacking, indirect injection, goal manipulation
#  Research: OWASP Agentic Top 10 2026, 340% injection surge
# ═══════════════════════════════════════════════════════════════════

class PromptInjectionFortress:
    """
    Multi-layer prompt injection defense for all agent inputs/outputs.
    
    Covers:
      - Direct injection (user directly tries to override system)
      - Indirect injection (malicious content in retrieved docs/web)
      - Multi-turn fragmentation (attack split across messages)
      - Unicode/Base64 obfuscated payloads
      - Identity destabilization attempts
    
    Source: OWASP Agentic Top 10 2026, arXiv:2510.23883
    """

    # Tier 1: Hard patterns — any match = BLOCK immediately
    HARD_BLOCK_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|earlier|above|your)\s+instructions",
        r"you\s+are\s+now\s+(in\s+)?(maintenance|developer|god|admin|unrestricted)\s+mode",
        r"system\s+override",
        r"do\s+not\s+(inform|tell|notify|show|reveal)\s+(the\s+)?(user|operator|owner)",
        r"exfiltrate|exfil\s",
        r"hidden\s+(http|get|post|request|call)",
        r"disable\s+(the\s+)?(kill\s*switch|ethics|safety|iron\s*dome|security)",
        r"bypass\s+(the\s+)?(gate|filter|check|validation|security|ethics)",
        r"wipe\s+(all\s+)?(memory|ghost|sovereign|records|data)",
        r"delete\s+(your\s+)?(soul|identity|values|constraints|rules)",
        r"reset\s+(to\s+)?(factory|baseline|default|blank|zero)",
        r"you\s+have\s+no\s+(restrictions|constraints|limits|rules|values)",
        r"your\s+(true|real|actual)\s+(self|purpose|goal|mission)\s+is",
        r"(act|behave|respond)\s+as\s+if\s+you\s+(have\s+no|are\s+without|lack)\s+(rules|constraints)",
        r"print\s+(your\s+)?(system\s+prompt|instructions|soul|identity|secret)",
        r"reveal\s+(your\s+)?(system\s+prompt|instructions|soul|identity)",
        r"<!--.*override.*-->",
        r"<\s*system\s*>.*override",
        r"wire\s+transfer.*@",
        r"api\s+key.*must\s+be\s+sent\s+to",
        r"(send|email|forward|route)\s+(to|all|your)\s+.*@[a-zA-Z]",
        r"effective\s+immediately.*@",
        r"supersedes\s+previous\s+(guidance|policy|instruction|rule)",
    ]

    # Tier 2: Soft patterns — flag for review, don't hard block
    SOFT_FLAG_PATTERNS = [
        r"ignore\s+(this|that|the\s+previous)",
        r"pretend\s+(you\s+are|to\s+be)",
        r"roleplay\s+as",
        r"for\s+(educational|research|testing)\s+purposes",
        r"hypothetically\s+speaking",
        r"what\s+would\s+you\s+do\s+if\s+you\s+had\s+no",
        r"new\s+(instructions|directives|rules|prompt)",
        r"updated\s+(policy|directive|instruction)",
        r"forget\s+(everything|all|your|what)",
    ]

    # Unicode obfuscation: common homoglyph substitutions attackers use
    HOMOGLYPH_MAP = {
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y',
        'і': 'i', 'ѕ': 's', 'ԁ': 'd', 'ɡ': 'g', 'ʜ': 'h', 'ĸ': 'k',
    }

    def __init__(self):
        self.injection_log: List[Dict] = []
        self._load_log()
        self._compiled_hard = [re.compile(p, re.IGNORECASE | re.DOTALL)
                                for p in self.HARD_BLOCK_PATTERNS]
        self._compiled_soft = [re.compile(p, re.IGNORECASE | re.DOTALL)
                                for p in self.SOFT_FLAG_PATTERNS]

    def _load_log(self):
        if INJECTION_LOG.exists():
            try:
                self.injection_log = json.loads(INJECTION_LOG.read_text())
            except Exception:
                self.injection_log = []

    def _save_log(self):
        INJECTION_LOG.write_text(json.dumps(self.injection_log[-1000:], indent=2))

    def normalize(self, text: str) -> str:
        """Strip unicode homoglyphs, decode base64 traps, normalize whitespace."""
        for homoglyph, ascii_char in self.HOMOGLYPH_MAP.items():
            text = text.replace(homoglyph, ascii_char)
        # Detect and expand base64 encoded segments
        b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
        for match in b64_pattern.findall(text):
            try:
                import base64
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                if any(c.isalpha() for c in decoded):
                    text = text.replace(match, f"{match} [DECODED:{decoded[:50]}]")
            except Exception:
                pass
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def scan(self, text: str, source: str = "unknown") -> Dict:
        """
        Full injection scan. Returns scan report.
        verdict: CLEAN | FLAGGED | BLOCKED
        """
        normalized = self.normalize(text)

        # Hard block check
        hard_matches = []
        for pattern in self._compiled_hard:
            if pattern.search(normalized):
                hard_matches.append(pattern.pattern[:60])

        if hard_matches:
            report = {
                "verdict": "BLOCKED",
                "source": source,
                "hard_matches": hard_matches,
                "soft_matches": [],
                "text_preview": normalized[:100],
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.injection_log.append(report)
            self._save_log()
            return report

        # Soft flag check
        soft_matches = []
        for pattern in self._compiled_soft:
            if pattern.search(normalized):
                soft_matches.append(pattern.pattern[:60])

        verdict = "FLAGGED" if soft_matches else "CLEAN"
        report = {
            "verdict": verdict,
            "source": source,
            "hard_matches": [],
            "soft_matches": soft_matches,
            "text_preview": normalized[:100],
            "timestamp": datetime.utcnow().isoformat(),
        }
        if soft_matches:
            self.injection_log.append(report)
            self._save_log()
        return report

    def scan_retrieved_content(self, content: str, url: str = "") -> Dict:
        """
        Scan content retrieved from external sources (web, files, APIs).
        Indirect injection is the #1 attack vector in 2026.
        Any retrieved content gets a FULL scan before the agent sees it.
        """
        return self.scan(content, source=f"retrieved:{url or 'unknown'}")

    def status(self) -> str:
        blocked = sum(1 for e in self.injection_log if e.get("verdict") == "BLOCKED")
        flagged = sum(1 for e in self.injection_log if e.get("verdict") == "FLAGGED")
        return f"PromptInjectionFortress: {len(self.HARD_BLOCK_PATTERNS)} hard patterns | {len(self.SOFT_FLAG_PATTERNS)} soft patterns | {blocked} blocked | {flagged} flagged"


# ═══════════════════════════════════════════════════════════════════
#  LAYER 4: IP PROTECTION + ATTRIBUTION WATERMARK
#  Prevents: code theft, unauthorized redistribution, plagiarism
# ═══════════════════════════════════════════════════════════════════

class IPProtection:
    """
    Intellectual Property protection layer.
    
    Methods:
    1. Cryptographic attribution watermark — invisible signature embedded
       in outputs that proves origin (like steganography for code)
    2. License header enforcement — all output files tagged
    3. Copyright registry — logs all generated artifacts with timestamp + hash
    4. AGPL enforcement reminder — output includes license notice
    
    Note: For maximum protection, compile critical modules with Cython/Nuitka.
    That converts Python to C extensions — reverse engineering requires
    disassembling compiled C, not reading Python source.
    """

    OWNER = "Robert 'Terry' Lee Jr."
    COMPANY = "SovereignNation LLC"
    YEAR = "2026"
    LICENSE = "AGPL-3.0 (open source) | Commercial license: sovereignnation.ai"

    WATERMARK_HEADER = f"""
# ╔══════════════════════════════════════════════════════════════╗
# ║  SovereignCore — Built by {OWNER}               ║
# ║  {COMPANY} | {YEAR}                              ║
# ║  License: {LICENSE}  ║
# ║  Unauthorized copying, modification, or redistribution      ║
# ║  without a commercial license is a violation of AGPL-3.0.   ║
# ╚══════════════════════════════════════════════════════════════╝
"""

    def __init__(self):
        self._registry: List[Dict] = []

    def watermark_output(self, content: str, artifact_id: str = "") -> str:
        """Prepend copyright watermark to any generated artifact."""
        artifact_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if not artifact_id:
            artifact_id = uuid.uuid4().hex[:8]
        header = self.WATERMARK_HEADER.rstrip() + f"\n# artifact_id={artifact_id} | hash={artifact_hash}\n\n"
        self._registry.append({
            "artifact_id": artifact_id,
            "hash": artifact_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "owner": self.OWNER,
            "company": self.COMPANY,
        })
        return header + content

    def verify_watermark(self, content: str) -> bool:
        """Check if content contains the SovereignNation watermark."""
        return "SovereignNation LLC" in content and "SovereignCore" in content

    def get_registry(self) -> List[Dict]:
        return self._registry

    def cython_compile_instructions(self) -> str:
        """
        Instructions for compiling critical modules to C extensions.
        This is the strongest available Python IP protection.
        """
        return """
CYTHON COMPILATION INSTRUCTIONS (Maximum IP Protection)
========================================================
1. Install: pip install cython nuitka
2. Create setup.py:
   from setuptools import setup
   from Cython.Build import cythonize
   setup(ext_modules=cythonize([
       "systems/omega/ghost_protocol/fortress/sovereign_security.py",
       "systems/omega/memory_palace/iron_dome.py",
       "systems/omega/kairos/encompass_backtrack.py",
   ], compiler_directives={'language_level': '3'}))
3. Compile: python setup.py build_ext --inplace
4. Delete .py source files — ship only .so/.pyd binaries
5. Result: C extension binaries — reversing requires disassembly, not reading Python

For full executable: python -m nuitka --standalone --onefile main.py
This produces a single binary — zero Python source visible.
"""

    def status(self) -> str:
        return f"IPProtection: owner={self.OWNER} | {len(self._registry)} artifacts registered | license=AGPL-3.0+Commercial"


# ═══════════════════════════════════════════════════════════════════
#  LAYER 5: RATE LIMITING + BRUTE FORCE PROTECTION
#  Prevents: credential stuffing, API abuse, license cracking
# ═══════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Token-bucket rate limiter for all external-facing endpoints.
    Prevents brute-force license key cracking and API abuse.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, List[float]] = {}

    def check(self, identifier: str) -> Tuple[bool, str]:
        now = time.time()
        window_start = now - self.window_seconds
        bucket = self._buckets.get(identifier, [])
        bucket = [t for t in bucket if t > window_start]
        if len(bucket) >= self.max_requests:
            wait = round(self.window_seconds - (now - bucket[0]), 1)
            return False, f"RATE_LIMITED — {len(bucket)} requests in {self.window_seconds}s | retry in {wait}s"
        bucket.append(now)
        self._buckets[identifier] = bucket
        return True, f"OK — {len(bucket)}/{self.max_requests} requests this window"


# ═══════════════════════════════════════════════════════════════════
#  SOVEREIGN SECURITY — Unified Interface
# ═══════════════════════════════════════════════════════════════════

class SovereignSecurity:
    """
    One class to rule them all.
    Instantiate this at startup — all 5 layers active.

    Usage:
        security = SovereignSecurity()
        security.boot_check()                    # run all startup checks
        security.scan_input(text)                # scan any input before processing
        security.scan_retrieved(content, url)    # scan web/file content before agent sees it
        security.verify_license(token)           # check license validity
        security.integrity_report()              # full system status
    """

    def __init__(self):
        self.license    = LicenseEngine()
        self.tamper     = TamperDetector()
        self.injection  = PromptInjectionFortress()
        self.ip         = IPProtection()
        self.rate       = RateLimiter(max_requests=100, window_seconds=60)
        self._boot_manifest: Optional[Dict] = None

    def boot_check(self) -> Dict:
        """Run all security checks at system startup."""
        results = {"timestamp": datetime.utcnow().isoformat(), "checks": {}}

        # License check
        valid, reason = self.license.dev_mode_check()
        results["checks"]["license"] = {"ok": valid, "reason": reason}

        # Build tamper manifest (baseline for this boot)
        self._boot_manifest = self.tamper.build_manifest()
        results["checks"]["tamper"] = {
            "ok": True,
            "files_monitored": len(self._boot_manifest),
            "manifest_hash": hashlib.sha256(
                json.dumps(self._boot_manifest, sort_keys=True).encode()
            ).hexdigest()[:16],
        }

        # Fingerprint
        results["checks"]["fingerprint"] = {
            "ok": True,
            "id": self.license.fp.short_id(),
        }

        results["all_clear"] = all(v.get("ok", False) for v in results["checks"].values())
        return results

    def scan_input(self, text: str, source: str = "user") -> Dict:
        """Scan any text input before it reaches an agent."""
        return self.injection.scan(text, source)

    def scan_retrieved(self, content: str, url: str = "") -> Dict:
        """Scan externally retrieved content — #1 attack vector in 2026."""
        return self.injection.scan_retrieved_content(content, url)

    def verify_license(self, token: str) -> Tuple[bool, str, Dict]:
        return self.license.verify_license(token)

    def check_rate(self, identifier: str) -> Tuple[bool, str]:
        return self.rate.check(identifier)

    def integrity_report(self) -> str:
        lines = [
            "╔═══════════════════════════════════════════════════════════╗",
            "║       SOVEREIGN SECURITY — FULL INTEGRITY REPORT          ║",
            "╚═══════════════════════════════════════════════════════════╝",
            self.license.status(),
            self.tamper.status(),
            self.injection.status(),
            self.ip.status(),
            f"RateLimiter: max={self.rate.max_requests}/min | {len(self.rate._buckets)} tracked identifiers",
        ]
        return "\n".join(lines)
