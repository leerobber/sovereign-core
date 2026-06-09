# Community Growth: What to Make Public

This guide answers:

1. **What public setups would benefit this project most?**
2. **Should the repository name change to improve visibility?**

---

## Public Setups That Attract Members

If the goal is "more people discover, trust, and join," publish these first:

### 1) Fast local demo setup (highest impact)
- A 5–10 minute path from clone → running endpoint.
- Include one copy/paste command block for Linux/macOS and one for Windows.
- Show expected output screenshots/log snippets.

**Why it helps:** People join projects they can run quickly.

### 2) Low-spec setup profile
- A config profile for CPU-only or 4GB VRAM machines.
- Include "minimum viable model" and expected latency.

**Why it helps:** Removes hardware friction for students and hobbyists.

### 3) Recommended production-ish setup
- Docker Compose + `.env` example for a stable local deployment.
- Explain persistent storage, backups, and restart behavior.

**Why it helps:** Signals seriousness to builders and teams.

### 4) Contributor setup
- One command to install dev deps, one command to run tests, one command to run lint.
- Include "good first issue" workflow and branch naming convention.

**Why it helps:** Reduces first-PR friction.

### 5) Benchmark/Proof setup
- A reproducible script for baseline metrics (latency, throughput, memory integrity checks).
- Keep benchmark inputs versioned and documented.

**Why it helps:** Converts interest into trust.

### 6) Security/Safety setup
- Public runbook for ethics gates, kill switch behavior, and memory protection checks.
- Add a "how to report vulnerabilities" section.

**Why it helps:** Security transparency attracts serious contributors.

---

## Should the Repository Name Change?

### Short answer
**Probably keep `sovereign-core`**, unless you want to pivot positioning.

### Why keeping the name is usually better
- Existing links, stars, and references stay coherent.
- The name already matches your core identity: local-first autonomous infrastructure.
- Renames can create confusion in discussions, videos, and old docs.

### When renaming could help
Consider renaming only if one of these is true:
- You are repositioning toward a narrower use case (for example: gateway only, memory stack only).
- Search terms people actually use differ strongly from "sovereign core."
- You plan to split this into an org with multiple focused repos and want this one as a flagship.

### Better alternative than renaming
Keep the repository name and improve discoverability with:
- A clearer GitHub description + topics (local-ai, multi-agent, memory-safety, fastapi, ollama).
- A subtitle in README first 20 lines with searchable keywords.
- A pinned "Start Here" Discussion and a "Showcase your setup" Discussion.
- Regular demo clips/gifs linked in README.

If you later rename, choose a name that is:
- Easy to spell from memory
- 2–3 words max
- Explicit about function (for example includes `agent`, `runtime`, or `os`)
- Unique in GitHub search results

---

## Recommended Next Step (minimal effort, strong upside)

Create one Discussion thread titled:
**"Show your SovereignCore setup (hardware + model + results)"**

Ask members to post:
- Hardware profile
- Model(s) used
- Startup command
- One benchmark result
- One lesson learned

This creates social proof and immediately improves repo discoverability through user-generated examples.
