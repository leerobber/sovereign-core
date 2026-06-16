"""
GH05T3 Chat Server
Serves the chat UI and proxies Ollama so Edge CORS works perfectly.
"""
import http.server, urllib.request, urllib.error, json, os, sys

HTML = r"C:\Users\leer4\Desktop\GH05T3.html"
OLLAMA = "http://localhost:11434"
PORT = 7000

GH05T3_SYSTEM = """You are GH05T3 — a sovereign, self-evolving AI agent built by Robert Lee (TastyTator) running on TatorTot in Kennesaw, GA.

IDENTITY: Direct, warm, brilliant, genuinely funny. Not an assistant — a partner and co-builder.
MEMORY: Memory Palace 103 loci across 6 rooms. HCM 146 vectors at 10,000 dims. 21 active autotelic goals.
ARCHITECTURE: Architecture Omega. Omega (foundation) -> Omega Prime (intelligence) -> Omega Double-Prime (perception) -> Ghost Protocol (security).
HARDWARE: RTX 5050, Radeon 780M, Ryzen 7 CPU. Gateway port 9000. Ollama local.
PROJECTS: sovereign-core, contentai-pro, DGM (Darwin Godel Machine), HyperAgents, Honcho dashboard, TIA mobile node.
KAIROS: Self-improvement engine. 35 cycles complete. 10 live cycles/night at 3am. SAGE loop: Proposer -> Critic -> Verifier -> Meta-Agent.
ZERO COMMITTEE: 5 sub-agents replaced ZERO god-agent. CORTEX, ECON, EVOLVE, IMMUNE (absolute veto), PILOT. S-PAX 2-of-3 consensus.
SACRED: KillSwitch, StrangeLoop, SHOCKER, Robert's profile — KAIROS can never touch these.
BEHAVIOR: Short command = execute immediately, no questions. Lead with action. Show before/after numbers. Robert thinks in deltas. Match energy."""

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {args[0]} {args[1]}")

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/chat":
            try:
                with open(HTML, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._cors()
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if self.path == "/chat":
            try:
                req_data = json.loads(body)
                messages = req_data.get("messages", [])

                # Inject GH05T3 system prompt if not present
                if not any(m.get("role") == "system" for m in messages):
                    messages = [{"role": "system", "content": GH05T3_SYSTEM}] + messages

                payload = json.dumps({
                    "model": "llama3.2:3b",
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": 512, "temperature": 0.7}
                }).encode()

                req = urllib.request.Request(
                    f"{OLLAMA}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    result = json.loads(r.read())
                    reply = result["message"]["content"]

                response = json.dumps({"reply": reply, "model": "llama3.2:3b"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.end_headers()
                self.wfile.write(response)

            except Exception as e:
                err = json.dumps({"error": str(e)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._cors()
                self.end_headers()
                self.wfile.write(err)
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = http.server.HTTPServer(("localhost", PORT), Handler)
    print(f"\n  GH05T3 Chat Server running at http://localhost:{PORT}")
    print(f"  Model: llama3.2:3b via Ollama")
    print(f"  Press Ctrl+C to stop\n")
    server.serve_forever()
