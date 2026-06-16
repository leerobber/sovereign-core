# Agent Training Status

## Current State (2026-05-18)

| Agent | HF Repo | Bootstrap Pairs | Trained | Loss | Deployed |
|-------|---------|----------------|---------|------|----------|
| Avery | tastytator/avery-sovereign-lora | 712 (economy) | Yes — ORPO 5.1min L40 | 2.3771 | Pending (GGUF downloading) |
| FORGE | tastytator/forge-sovereign-lora | 9 (bootstrap) | Yes — ORPO 0.4min RTX5090 | 1.2094 | No |
| ORACLE | tastytator/oracle-sovereign-lora | 5 (bootstrap) | No | — | No |
| CODEX | tastytator/codex-sovereign-lora | 4 (bootstrap) | No | — | No |
| SENTINEL | tastytator/sentinel-sovereign-lora | 4 (bootstrap) | No | — | No |
| NEXUS | tastytator/nexus-sovereign-lora | 3 (bootstrap) | No | — | No |

## Avery Deployment Status
- LoRA: `tastytator/avery-sovereign-lora` ✅
- GGUF Q8_0 (8.1GB): `tastytator/avery-sovereign-lora/avery-sovereign-q8.gguf` ✅
- Downloading to: `C:\Users\leer4\GH05T3\avery-sovereign-q8.gguf`
- Modelfile: `C:\Users\leer4\GH05T3\Modelfile.avery`
- Next: `ollama create avery-sovereign -f Modelfile.avery`

## Note on API Credits
Anthropic API balance depleted as of 2026-05-18. New bootstrap pairs for remaining agents
(SENTINEL, NEXUS, ORACLE, CODEX) require credits. FORGE trained on 9 existing pairs.

## Steps to Train (Free Path — No API Credits Needed)

### Step 1: Generate Static Bootstrap Data (FREE)
```
cd C:\Users\leer4\GH05T3
python generate_static_bootstrap.py
```
Output: `data/agents_bootstrap.jsonl` (~34 elite DPO pairs)

### Step 2: Upload to HuggingFace
```
python pre_train.py
```
Uploads agents config to tastytator/sovereign-economy dataset

### Step 3: Train on Kaggle (FREE - 30hr/week T4 GPU)
1. Open kaggle_train.ipynb on Kaggle
2. Set AGENT = 'avery', MODE = 'orpo' 
3. Run all cells
4. Repeat for each agent

### Step 4: Deploy to Ollama
```
train.bat deploy
```

## Training Priority Order

1. **Avery** — most critical for business planning + monetization
2. **FORGE** — needed for building platform features
3. **SENTINEL** — needed before going live with users
4. **NEXUS** — coordinates all other agents
5. **ORACLE** — enhances recall quality
6. **CODEX** — nice-to-have for docs generation

## Quality Benchmarks (Test After Training)

### Avery benchmark prompts:
- "Build a go-to-market strategy for SovereignNation Phase 1"
- "Design the 90-day launch roadmap with specific milestones"
- "Create pricing tiers that cover infrastructure at 100 users"

### FORGE benchmark prompts:
- "Write a FastAPI health check endpoint for all 5 SovereignNation services"
- "Build a JWT middleware that checks subscription tier"

### SENTINEL benchmark prompts:
- "Review this JWT implementation for OWASP Top 10 vulnerabilities"
- "Audit .env file management for credential exposure risks"
