# SOVEREIGN CORE — SESSION MEMORY

# READ THIS FIRST every single session. Update it every session.

# This file exists because Claude has no memory between sessions.

# Last updated: 2026-04-23

---

## WHO YOU ARE

- Name: Terry (Robert Lee Jr.)
- GitHub: leer4030 (NOT leerobber — leerobber is wrong handle)
- Email: [tastytatortot@gmail.com](mailto:tastytatortot@gmail.com)
- Machine: Lenovo LOQ 15AHP10 "TatorTot"
- OS: Windows 11
- Location: Kennesaw, GA

---

## THE MACHINE — TatorTot Hardware

- GPU: NVIDIA RTX 5050 Laptop GPU (8GB VRAM) — Blackwell architecture
- iGPU: AMD Radeon 780M
- CPU: Ryzen 7 (also runs inference)
- RAM: Check with: Get-CimInstance -ClassName Win32_PhysicalMemory | Measure-Object Capacity -Sum

---

## CRITICAL KNOWN ISSUES — DO NOT REPEAT THESE MISTAKES

### 1. OLLAMA GPU — BIGGEST RECURRING PROBLEM

- Ollama kept defaulting to CPU instead of GPU
- ROOT CAUSE: RTX 5050 is Blackwell (SM_120) — Ollama 0.18.3 missing CUDA 13 DLLs
  - Missing: cublaslt64_13.dll, nvrtc64_130_0.dll
- STATUS AS OF 2026-04-23: Upgrading Ollama from 0.18.3 → 0.21.1 via winget
- FIX FILES ALREADY ON MACHINE:
  - C:\\Users\\leer4\\Documents\\fix_ollama_gpu.bat (CUDA DLL fix)
  - C:\\Users\\leer4\\Documents\\fix_ollama_gpu_vulkan.bat (Vulkan fallback)
  - C:\\Users\\leer4\\Documents\\start_ollama_gpu.bat (startup script)
- VERIFY GPU WORKING: run `ollama ps` — PROCESSOR column must say GPU not CPU
- Vulkan flag: set OLLAMA_VULKAN=true && set GGML_VK_VISIBLE_DEVICES=0

### 2. OLLAMA PATH — NOT IN SYSTEM PATH

- ollama.exe is NOT in PATH for PowerShell
- Full path: C:\\Users\\leer4\\AppData\\Local\\Programs\\Ollama\\ollama.exe
- Always use full path OR use cmd shell not powershell

### 3. GATEWAY PORT

- Gateway runs on port 9000 NOT 8000
- URL: <http://127.0.0.1:9000>
- Ollama: <http://127.0.0.1:11434>

### 4. WINGET SEARCH ISSUE

- `winget search` triggers interactive prompt that hangs
- Use: winget upgrade Ollama.Ollama --silent (works fine)
- Do NOT pipe winget output with &gt;nul on same line as taskkill

### 5. CMD QUOTE ESCAPING

- JSON via curl in cmd always fails due to quote escaping
- ALWAYS use PowerShell for JSON/REST calls: $body = '{"key":"value"}'; Invoke-RestMethod -Uri "..." -Method POST -Body $body -ContentType "application/json"

---

## MODELS AVAILABLE (ollama list as of 2026-04-23)

- gemma3:12b (8.1GB) — too big for GPU alone
- llama3.2:3b (2.0GB) — fast, CPU/GPU
- nomic-embed-text (274MB) — embeddings
- dolphin-phi (1.6GB) — small fast
- dolphin-llama3:8b (4.7GB) — fits in GPU
- qwen2.5:7b (4.7GB) — PROPOSER slot — fits in GPU
- llama3:8b (4.7GB) — fits in GPU

## MISSING PREFERRED MODELS (need pull)

- deepseek-coder:6.7b — VERIFIER slot
- llama3.1 — CRITIC slot

---

## PROJECT LOCATIONS ON DISK

- C:\\Users\\leer4\\sovereign-core\\ — MAIN PROJECT (gateway, kairos, ghost_protocol, etc.)
- C:\\Users\\leer4\\Documents\\local-ai-mesh\\ — ChromaDB, LoRA pipeline, agent economy
- C:\\Users\\leer4\\MYTHOS\\ — Cybersecurity research platform
- C:\\Users\\leer4\\verelene_v5\\ — Agent system with dashboard
- C:\\Users\\leer4\\Jarvis\\ — Jarvis agent build
- C:\\Users\\leer4\\openclaw\\ — Full monorepo (Docker, MCP, skills)
- C:\\Users\\leer4\\Documents\\agent-economy\\ — Agent economy system

## KEY FILES

- C:\\Users\\leer4\\sovereign-core\\gateway\\main.py — FastAPI gateway
- C:\\Users\\leer4\\sovereign-core\\ghost_protocol\\ghost_agent.py — pairing agent (BUILT 2026-04-23)
- C:\\Users\\leer4\\sovereign-core\\ghost_protocol\\fortress\\sovereign_security.py — security layer
- C:\\Users\\leer4\\Desktop\\GH05T3.html — GH05T3 web interface
- C:\\Users\\leer4\\Desktop\\START_EVERYTHING.bat — startup script

---

## WHAT TERRY IS BUILDING

Goal: Sovereign Core — self-improving autonomous AI platform Partner agent: GH05T3 — the AI that helps Terry build Sovereign Core and run the business Business goal: Wire all GitHub projects together, create SovereignNation LLC business

### The 6 silos that need to be wired together:

1. sovereign-core — the core platform
2. local-ai-mesh — ChromaDB knowledge + LoRA training
3. MYTHOS — cybersecurity research module
4. verelene_v5 — agent evolution system
5. Jarvis — personal assistant layer
6. openclaw — MCP integration hub

### GH05T3's role:

- Lives in sovereign-core/ghost_protocol/
- Pairs with Emergent web UI via pair codes
- Routes inference through gateway (port 9000)
- Bridges all 6 silos
- Helps build the business layer

---

## SESSION LOG — WHAT WAS DONE EACH SESSION

### 2026-04-23

- DISCOVERED: Ollama running on CPU not GPU (RTX 5050 CUDA 13 DLL issue)
- DISCOVERED: Gateway was on port 9000 not 8000
- BUILT: ghost_agent.py (pairing agent for Emergent)
  - Location: C:\\Users\\leer4\\sovereign-core\\ghost_protocol\\ghost_agent.py
  - Supports: --pair-code, --status, --list-models, --serve
  - Runs companion service on port 8006
- STARTED: Ollama upgrade 0.18.3 → 0.21.1 (in progress at end of session)
- TODO NEXT SESSION:
  1. Verify Ollama 0.21.1 installed and GPU working (ollama ps → should show GPU)
  2. Pull missing models: deepseek-coder:6.7b and llama3.1
  3. Start gateway: cd C:\\Users\\leer4\\sovereign-core && python launch_gateway.py
  4. Test ghost_agent.py pairing
  5. Begin wiring the 6 silos together

---

## HOW TO START EACH SESSION

Claude: Read this file FIRST before doing anything else. Command to read it: Desktop Commander read_file C:\\Users\\leer4\\SOVEREIGN_STATUS.md Then check Ollama: C:\\Users\\leer4\\AppData\\Local\\Programs\\Ollama\\ollama.exe ps Then check gateway: curl <http://127.0.0.1:9000/health>Then proceed with TODO from last session above.

### 2026-04-23 — UPDATE: OLLAMA GPU FIXED

- Ollama upgraded to 0.21.1 — RTX 5050 GPU NOW WORKING
- Confirmed: qwen2.5:7b running 100% GPU, 6.3GB VRAM, 16384 context
- GPU fix is PERMANENT — no more CPU fallback
- LESSON: Always check ollama ps immediately after any model load. Do not wait.
- TODO NEXT SESSION:
  1. Pull missing models: deepseek-coder:6.7b and llama3.1
  2. Start gateway: cd C:\\Users\\leer4\\sovereign-core && python launch_gateway.py
  3. Verify gateway routes to GPU (should be fast now)
  4. Test ghost_agent.py: python ghost_protocol\\ghost_agent.py --pair-code 555215
  5. Begin wiring 6 silos into GH05T3
