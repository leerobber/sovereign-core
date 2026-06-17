"""
Sovereign Core — Mission Control Dashboard
Animated Rich TUI with 10 switchable themes, spinning 3D cube, particle effects.
Default theme: TRON (#6)
"""

import asyncio
import math
import time
import sys
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box

# ── Themes ──────────────────────────────────────────────────────

THEMES = {
    1: {
        "name": "CYBERPUNK",
        "primary": "#ff2079", "secondary": "#00f0ff", "accent": "#ffef00",
        "bg": "#0a0a1a", "text": "#e0e0ff", "success": "#00ff9f",
        "warning": "#ffef00", "error": "#ff2079", "dim": "#4a4a6a",
        "border": "bright_magenta", "bar_complete": "#ff2079", "bar_remaining": "#2a2a4a",
        "header_char": "◈", "separator": "═", "bullet": "▸",
        "status_on": "◉", "status_off": "◯",
    },
    2: {
        "name": "MATRIX",
        "primary": "#00ff41", "secondary": "#008f11", "accent": "#00ff41",
        "bg": "#000000", "text": "#00ff41", "success": "#00ff41",
        "warning": "#ccff00", "error": "#ff0000", "dim": "#003b00",
        "border": "green", "bar_complete": "#00ff41", "bar_remaining": "#002200",
        "header_char": "█", "separator": "─", "bullet": ">",
        "status_on": "[●]", "status_off": "[ ]",
    },
    3: {
        "name": "STEAMPUNK",
        "primary": "#d4a574", "secondary": "#8b6914", "accent": "#cd7f32",
        "bg": "#1a1408", "text": "#d4a574", "success": "#9acd32",
        "warning": "#daa520", "error": "#8b0000", "dim": "#4a3c28",
        "border": "yellow", "bar_complete": "#cd7f32", "bar_remaining": "#2a2010",
        "header_char": "⚙", "separator": "─", "bullet": "⚡",
        "status_on": "⊛", "status_off": "⊘",
    },
    4: {
        "name": "SOLARPUNK",
        "primary": "#2dd4a8", "secondary": "#059669", "accent": "#fbbf24",
        "bg": "#022c22", "text": "#a7f3d0", "success": "#34d399",
        "warning": "#fbbf24", "error": "#f87171", "dim": "#064e3b",
        "border": "green", "bar_complete": "#2dd4a8", "bar_remaining": "#064e3b",
        "header_char": "❋", "separator": "━", "bullet": "✦",
        "status_on": "✿", "status_off": "○",
    },
    5: {
        "name": "HACKER",
        "primary": "#33ff33", "secondary": "#1a9a1a", "accent": "#33ff33",
        "bg": "#0a0a0a", "text": "#33ff33", "success": "#33ff33",
        "warning": "#ffff33", "error": "#ff3333", "dim": "#1a3a1a",
        "border": "bright_green", "bar_complete": "#33ff33", "bar_remaining": "#0a1a0a",
        "header_char": "#", "separator": "-", "bullet": "$",
        "status_on": "[ACTIVE]", "status_off": "[NULL]",
    },
    6: {
        "name": "TRON",
        "primary": "#00d4ff", "secondary": "#ff6600", "accent": "#ffffff",
        "bg": "#000814", "text": "#7df9ff", "success": "#00d4ff",
        "warning": "#ff6600", "error": "#ff0040", "dim": "#001a33",
        "border": "bright_cyan", "bar_complete": "#00d4ff", "bar_remaining": "#001a33",
        "header_char": "◆", "separator": "━", "bullet": "▹",
        "status_on": "◈", "status_off": "◇",
    },
    7: {
        "name": "PHANTOM",
        "primary": "#bf5fff", "secondary": "#8b5cf6", "accent": "#e0e0ff",
        "bg": "#0d0221", "text": "#c4b5fd", "success": "#a78bfa",
        "warning": "#fbbf24", "error": "#f43f5e", "dim": "#2e1065",
        "border": "bright_magenta", "bar_complete": "#bf5fff", "bar_remaining": "#1a0536",
        "header_char": "◊", "separator": "─", "bullet": "⊹",
        "status_on": "☽", "status_off": "☾",
    },
    8: {
        "name": "BLOODMOON",
        "primary": "#dc143c", "secondary": "#ff4500", "accent": "#ff6347",
        "bg": "#0a0000", "text": "#ff8a80", "success": "#ff4500",
        "warning": "#ff6347", "error": "#dc143c", "dim": "#330000",
        "border": "red", "bar_complete": "#dc143c", "bar_remaining": "#1a0000",
        "header_char": "☠", "separator": "═", "bullet": "†",
        "status_on": "⦿", "status_off": "⦾",
    },
    9: {
        "name": "ARCTIC OPS",
        "primary": "#88ccff", "secondary": "#4488cc", "accent": "#ffffff",
        "bg": "#0a0e14", "text": "#b0d4f1", "success": "#88ccff",
        "warning": "#ffcc44", "error": "#ff4444", "dim": "#1a2a3a",
        "border": "bright_blue", "bar_complete": "#88ccff", "bar_remaining": "#0a1520",
        "header_char": "◇", "separator": "─", "bullet": "›",
        "status_on": "◆", "status_off": "◇",
    },
    0: {
        "name": "VAPORWAVE",
        "primary": "#ff71ce", "secondary": "#b967ff", "accent": "#01cdfe",
        "bg": "#1a0025", "text": "#fffb96", "success": "#05ffa1",
        "warning": "#fffb96", "error": "#ff71ce", "dim": "#2a0040",
        "border": "magenta", "bar_complete": "#b967ff", "bar_remaining": "#1a0025",
        "header_char": "✧", "separator": "～", "bullet": "☆",
        "status_on": "♡", "status_off": "♢",
    },
}


# ── 3D Spinning Cube ───────────────────────────────────────────

class SpinningCube:
    VERTICES = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1,  1), (1, -1,  1), (1, 1,  1), (-1, 1,  1),
    ]
    EDGES = [
        (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]

    def __init__(self, size=6):
        self.size = size
        self.ax = self.ay = self.az = 0.0

    def _rot(self, x, y, z):
        ca, sa = math.cos(self.ax), math.sin(self.ax)
        y, z = y*ca - z*sa, y*sa + z*ca
        cb, sb = math.cos(self.ay), math.sin(self.ay)
        x, z = x*cb + z*sb, -x*sb + z*cb
        cc, sc = math.cos(self.az), math.sin(self.az)
        x, y = x*cc - y*sc, x*sc + y*cc
        return x, y, z

    def _proj(self, x, y, z):
        s = 4.0 / (z + 3.0)
        return int(x*s*self.size + self.size*2), int(y*s*self.size*0.5 + self.size)

    def render(self, theme):
        w, h = self.size*4+1, self.size*2+1
        canvas = [[' ']*w for _ in range(h)]
        pts = [self._proj(*self._rot(*v)) for v in self.VERTICES]
        for px, py in pts:
            if 0 <= py < h and 0 <= px < w:
                canvas[py][px] = '◈'
        for a, b in self.EDGES:
            x0,y0 = pts[a]; x1,y1 = pts[b]
            steps = max(abs(x1-x0), abs(y1-y0), 1)
            for i in range(steps+1):
                t = i/steps if steps else 0
                x, y = int(x0+(x1-x0)*t), int(y0+(y1-y0)*t)
                if 0 <= y < h and 0 <= x < w:
                    canvas[y][x] = "·∙•●"[i % 4]
        self.ax += 0.04; self.ay += 0.06; self.az += 0.02
        return '\n'.join(''.join(r) for r in canvas)


class ParticleSystem:
    def __init__(self, width=48):
        self.width = width
        self.particles = []
        self._tick = 0

    def emit(self, label=""):
        self.particles.append({"pos": 0.0, "speed": 0.8 + (hash(label) % 10) * 0.15})

    def tick(self):
        self._tick += 1
        trail = "░▒▓█▓▒░"
        active = []
        for p in self.particles:
            p["pos"] += p["speed"]
            if p["pos"] < self.width:
                active.append(p)
        self.particles = active[-8:]
        row = [' '] * self.width
        for p in self.particles:
            pos = int(p["pos"])
            for j, ch in enumerate(trail):
                idx = pos - j
                if 0 <= idx < self.width:
                    row[idx] = ch
        return ''.join(row)


class Waveform:
    BLOCKS = " ▁▂▃▄▅▆▇█"

    def __init__(self, width=44):
        self.width = width
        self.values = deque(maxlen=width)
        self._phase = 0.0

    def push(self, v):
        self.values.append(min(v, 1.0))

    def render(self, theme):
        self._phase += 0.15
        while len(self.values) < self.width:
            self.values.append(0.3 + 0.2 * math.sin(self._phase + len(self.values) * 0.3))
        chars = []
        for i, v in enumerate(self.values):
            a = max(0, min(1, v + 0.05 * math.sin(self._phase + i * 0.4)))
            chars.append(self.BLOCKS[int(a * (len(self.BLOCKS) - 1))])
        return ''.join(chars)


class RadarSweep:
    def __init__(self, radius=4):
        self.radius = radius
        self.angle = 0.0

    def render(self, theme, healthy=True):
        r = self.radius
        size = r * 2 + 1
        canvas = [[' '] * (size * 2) for _ in range(size)]
        for a in range(0, 360, 10):
            rad = math.radians(a)
            x = int(r + r * math.cos(rad))
            y = int(r + r * 0.5 * math.sin(rad))
            if 0 <= y < size and 0 <= x*2 < size*2:
                canvas[y][x*2] = '·'
        for i, ch in [(0, '█'), (1, '▓'), (2, '░'), (3, '░')]:
            ang = self.angle - i * 0.15
            for ri in range(r + 1):
                x = int(r + ri * math.cos(ang))
                y = int(r + ri * 0.5 * math.sin(ang))
                if 0 <= y < size and 0 <= x*2 < size*2:
                    canvas[y][x*2] = ch if i == 0 else '░'
        canvas[r][r*2] = '◉' if healthy else '✕'
        self.angle += 0.2
        return '\n'.join(''.join(row) for row in canvas)



# ── Dashboard ──────────────────────────────────────────────────

class Dashboard:
    GATEWAY_URL = os.environ.get("SOVEREIGN_URL", "http://localhost:8000")

    def __init__(self, theme_id=6):
        self.console = Console()
        self.theme_id = theme_id
        self.theme = THEMES[theme_id]
        self.cube = SpinningCube()
        self.particles = ParticleSystem()
        self.waveform = Waveform()
        self.radar = RadarSweep()
        self.request_log = deque(maxlen=20)
        self.health_data = {}
        self.metrics_data = {}
        self.models_data = []
        self.frame = 0
        self._session = None
        self._running = True
        self._last_req_count = 0

    @property
    def t(self):
        return self.theme

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        return self._session

    async def fetch_health(self):
        try:
            s = await self._get_session()
            async with s.get(f"{self.GATEWAY_URL}/health") as r:
                if r.status == 200:
                    self.health_data = await r.json()
        except Exception:
            self.health_data = {"status": "offline"}

    async def fetch_metrics(self):
        try:
            s = await self._get_session()
            async with s.get(f"{self.GATEWAY_URL}/metrics") as r:
                if r.status == 200:
                    data = await r.json()
                    total = sum(m.get("total_requests", 0) for m in data.get("models", {}).values())
                    if total > self._last_req_count:
                        for _ in range(min(total - self._last_req_count, 3)):
                            self.particles.emit("req")
                        self._last_req_count = total
                    self.metrics_data = data
        except Exception:
            pass

    async def fetch_models(self):
        try:
            s = await self._get_session()
            async with s.get(f"{self.GATEWAY_URL}/v1/models") as r:
                if r.status == 200:
                    self.models_data = (await r.json()).get("data", [])
        except Exception:
            pass

    def bar(self, val, width=20):
        filled = int(val / 100 * width)
        t = Text()
        t.append("█" * filled, style=self.t["bar_complete"])
        t.append("░" * (width - filled), style=self.t["dim"])
        t.append(f" {val:.0f}%", style=self.t["text"])
        return t

    def make_header(self):
        self.frame += 1
        t = self.t
        hdr = Text()
        hdr.append(f" {t['header_char']} SOVEREIGN CORE — MISSION CONTROL {t['header_char']} ", style=f"bold {t['primary']}")
        hdr.append("  ")
        hdr.append(f"[ {t['name']} ]", style=f"bold {t['secondary']}")
        hdr.append("  ")
        sp = self.frame % 20
        scan = ''.join('◈' if i == sp else ('·' if abs(i - sp) < 3 else ' ') for i in range(20))
        hdr.append(scan, style=t["accent"])
        return Panel(Align.center(hdr), border_style=t["border"], box=box.DOUBLE)

    def make_system_panel(self):
        t = self.t
        sd = self.health_data.get("system", {})
        ram = sd.get("ram", {})
        status = self.health_data.get("status", "offline")
        c = Text()
        ind = t["status_on"] if status == "healthy" else t["status_off"]
        sc = t["success"] if status == "healthy" else t["error"]
        c.append(f"  {ind} ", style=sc)
        c.append(f"STATUS: {status.upper()}\n\n", style=f"bold {sc}")
        c.append(f"  {t['bullet']} CPU  ", style=t["text"])
        c.append_text(self.bar(sd.get("cpu_percent", 0)))
        c.append("\n")
        rp = ram.get("percent", 0)
        c.append(f"  {t['bullet']} RAM  ", style=t["text"])
        c.append_text(self.bar(rp))
        c.append(f"\n         {ram.get('used_gb', 0)}/{ram.get('total_gb', 0)} GB\n", style=t["dim"])
        up = sd.get("uptime_seconds", 0)
        c.append(f"\n  {t['bullet']} UPTIME ", style=t["text"])
        c.append(f"{int(up//3600):02d}h {int(up%3600//60):02d}m {int(up%60):02d}s", style=f"bold {t['primary']}")
        return Panel(c, title=f"[{t['primary']}]SYSTEM[/]", border_style=t["border"], box=box.HEAVY)

    def make_gpu_panel(self):
        t = self.t
        gpu = self.health_data.get("system", {}).get("gpu", {})
        c = Text()
        if gpu.get("available"):
            vu = gpu.get("vram_used_mb", 0)
            vt = gpu.get("vram_total_mb", 1)
            c.append(f"  {t['status_on']} ", style=t["success"])
            c.append(f"{gpu.get('name', '?')}\n\n", style=f"bold {t['primary']}")
            c.append(f"  {t['bullet']} VRAM ", style=t["text"])
            c.append_text(self.bar(vu / vt * 100))
            c.append(f"\n         {vu}/{vt} MB\n", style=t["dim"])
            c.append(f"  {t['bullet']} UTIL ", style=t["text"])
            c.append_text(self.bar(gpu.get("utilization_percent", 0)))
            c.append("\n\n")
            for line in self.radar.render(t, True).split('\n'):
                c.append(f"    {line}\n", style=t["primary"])
        else:
            c.append(f"  {t['status_off']} GPU NOT DETECTED\n", style=t["error"])
        return Panel(c, title=f"[{t['primary']}]GPU[/]", border_style=t["border"], box=box.HEAVY)

    def make_models_panel(self):
        t = self.t
        ms = self.health_data.get("models", {})
        met = self.metrics_data.get("models", {})
        tbl = Table(box=box.SIMPLE_HEAVY, border_style=t["dim"], header_style=f"bold {t['primary']}", show_edge=False)
        tbl.add_column("Model", style=f"bold {t['text']}")
        tbl.add_column("Device", style=t["secondary"])
        tbl.add_column("Status", justify="center")
        tbl.add_column("Reqs", justify="right", style=t["accent"])
        tbl.add_column("Avg Lat", justify="right", style=t["text"])
        for mn, dev in [("llama3.2:3b", "CPU"), ("qwen2.5:7b", "GPU")]:
            st = ms.get(mn, "unknown")
            sd = Text()
            if st == "ready":
                sd.append(f" {t['status_on']} READY ", style=f"bold {t['success']}")
            else:
                sd.append(f" {t['status_off']} {st.upper()} ", style=t["error"])
            m = met.get(mn, {})
            al = f"{m.get('avg_latency_ms', 0)/1000:.1f}s" if m.get("avg_latency_ms") else "—"
            tbl.add_row(mn, dev, sd, str(m.get("total_requests", 0)), al)
        return Panel(tbl, title=f"[{t['primary']}]MODELS[/]", border_style=t["border"], box=box.HEAVY)

    def make_cube_panel(self):
        t = self.t
        c = Text()
        for line in self.cube.render(t).split('\n'):
            c.append(f"{line}\n", style=t["primary"])
        return Panel(c, title=f"[{t['primary']}]CORE[/]", border_style=t["border"], box=box.HEAVY)

    def make_particles_panel(self):
        t = self.t
        c = Text()
        c.append("  REQUEST FLOW\n", style=f"bold {t['text']}")
        c.append(f"  {self.particles.tick()}\n", style=t["primary"])
        c.append("  ", style=t["dim"])
        c.append("INPUT ", style=t["accent"])
        c.append("→→→ ", style=t["primary"])
        c.append("ROUTER ", style=t["secondary"])
        c.append("→→→ ", style=t["primary"])
        c.append("MODEL", style=t["success"])
        return Panel(c, title=f"[{t['primary']}]DATA FLOW[/]", border_style=t["border"], box=box.HEAVY)

    def make_latency_panel(self):
        t = self.t
        pcts = self.metrics_data.get("percentiles", {})
        c = Text()
        for mn in ["llama3.2:3b", "qwen2.5:7b"]:
            p = pcts.get(mn, {})
            c.append(f"  {t['bullet']} {mn}\n", style=f"bold {t['text']}")
            c.append("    p50 ", style=t["dim"])
            c.append(f"{p.get('p50',0)/1000:.2f}s", style=t["success"])
            c.append("  p95 ", style=t["dim"])
            c.append(f"{p.get('p95',0)/1000:.2f}s", style=t["warning"])
            c.append("  p99 ", style=t["dim"])
            c.append(f"{p.get('p99',0)/1000:.2f}s\n", style=t["error"])
        wv = 0.3
        for p in pcts.values():
            if p.get("p50", 0) > 0:
                wv = min(p["p50"] / 10000, 1.0)
        self.waveform.push(wv)
        c.append(f"\n  {self.waveform.render(t)}", style=t["primary"])
        return Panel(c, title=f"[{t['primary']}]LATENCY[/]", border_style=t["border"], box=box.HEAVY)

    def make_log_panel(self):
        t = self.t
        c = Text()
        if not self.request_log:
            idle = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            c.append(f"  {idle[self.frame % len(idle)]} Awaiting requests...\n", style=t["dim"])
            c.append(f"    Send traffic to {self.GATEWAY_URL}/v1/chat/completions\n", style=t["dim"])
        else:
            for e in list(self.request_log)[-10:]:
                c.append(f"  {e['time']} ", style=t["dim"])
                c.append("→ ", style=t["primary"])
                c.append(f"{e['model']:<14}", style=f"bold {t['text']}")
                c.append(" │ ", style=t["dim"])
                cc = t["success"] if e["complexity"] == "low" else t["warning"]
                c.append(f"{e['complexity'].upper():<5}", style=cc)
                c.append(" │ ", style=t["dim"])
                c.append(f"{e['latency']}\n", style=t["accent"])
        return Panel(c, title=f"[{t['primary']}]LIVE LOG[/]", border_style=t["border"], box=box.HEAVY)

    def make_footer(self):
        t = self.t
        c = Text()
        for key, label in [("0-9","Theme"),("T","Test"),("S","Stress"),("C","Clear"),("Q","Quit")]:
            c.append(f" [{key}] ", style=f"bold {t['accent']}")
            c.append(f"{label} ", style=t["dim"])
        c.append("  │  ", style=t["dim"])
        c.append(f"Theme: {t['name']}", style=f"bold {t['primary']}")
        return Panel(c, border_style=t["border"], box=box.HEAVY)

    def build_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="upper", size=16),
            Layout(name="middle", size=7),
            Layout(name="lower", size=14),
            Layout(name="footer", size=3),
        )
        layout["upper"].split_row(Layout(name="system", ratio=1), Layout(name="gpu", ratio=1), Layout(name="cube", ratio=1))
        layout["middle"].split_row(Layout(name="models", ratio=2), Layout(name="particles", ratio=1))
        layout["lower"].split_row(Layout(name="latency", ratio=1), Layout(name="log", ratio=2))
        return layout

    def render(self):
        lo = self.build_layout()
        lo["header"].update(self.make_header())
        lo["system"].update(self.make_system_panel())
        lo["gpu"].update(self.make_gpu_panel())
        lo["cube"].update(self.make_cube_panel())
        lo["models"].update(self.make_models_panel())
        lo["particles"].update(self.make_particles_panel())
        lo["latency"].update(self.make_latency_panel())
        lo["log"].update(self.make_log_panel())
        lo["footer"].update(self.make_footer())
        return lo

    async def send_test_prompt(self, prompt="Hello, what can you do?"):
        try:
            s = await self._get_session()
            start = time.time()
            async with s.post(f"{self.GATEWAY_URL}/v1/chat/completions",
                              json={"messages": [{"role": "user", "content": prompt}]}) as r:
                if r.status == 200:
                    data = await r.json()
                    rt = data.get("routing", {})
                    self.request_log.append({
                        "time": time.strftime("%H:%M:%S"),
                        "model": rt.get("model_name", "?"),
                        "complexity": rt.get("complexity", "?"),
                        "latency": f"{time.time()-start:.1f}s",
                    })
                    self.particles.emit(rt.get("model_name", ""))
        except Exception as e:
            self.request_log.append({"time": time.strftime("%H:%M:%S"), "model": "ERROR", "complexity": "—", "latency": str(e)[:20]})

    async def stress_test(self):
        prompts = ["Hi!", "Hello", "Hey", "What's up?", "Yo",
            "Write a Python class implementing a binary search tree with insert delete and search",
            "Explain TCP vs UDP protocols in detail with examples",
            "Implement quicksort in Python with type hints and benchmarks",
            "Debug this: why would a deadlock occur in a producer-consumer pattern?",
            "Thanks!"]
        await asyncio.gather(*[self.send_test_prompt(p) for p in prompts], return_exceptions=True)

    async def run(self):
        layout = self.render()
        with Live(layout, console=self.console, refresh_per_second=8, screen=True) as live:
            fc = 0
            while self._running:
                if fc % 16 == 0:
                    await asyncio.gather(self.fetch_health(), self.fetch_metrics(), self.fetch_models(), return_exceptions=True)
                fc += 1
                live.update(self.render())
                await asyncio.sleep(0.125)


async def main():
    theme_id = 6
    if len(sys.argv) > 1:
        try:
            theme_id = int(sys.argv[1])
        except ValueError:
            pass
    if theme_id not in THEMES:
        theme_id = 6

    dash = Dashboard(theme_id)

    import select, termios, tty
    old = None
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:
        pass

    async def keys():
        while dash._running:
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    k = sys.stdin.read(1)
                    if k.lower() == 'q': dash._running = False
                    elif k.lower() == 't': await dash.send_test_prompt()
                    elif k.lower() == 's': await dash.stress_test()
                    elif k.lower() == 'c': dash.request_log.clear()
                    elif k in '0123456789':
                        tid = int(k)
                        if tid in THEMES:
                            dash.theme_id = tid
                            dash.theme = THEMES[tid]
                else:
                    await asyncio.sleep(0.1)
            except Exception:
                await asyncio.sleep(0.1)

    try:
        await asyncio.gather(dash.run(), keys())
    finally:
        if old:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if dash._session and not dash._session.closed:
            await dash._session.close()


if __name__ == "__main__":
    asyncio.run(main())
