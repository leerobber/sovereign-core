AGENT_MODEL_MAP = {
    "FORGE": "forge-sovereign",
    "ORACLE": "oracle-sovereign",
    "CODEX": "codex-sovereign",
    "NEXUS": "nexus-sovereign",
    "AVERY": "avery-sovereign",
    "SENTINEL": "sentinel-sovereign",
}

SWARM_AGENTS = [
    {"id": "ORACLE", "role": "research", "channels": ["#broadcast", "#omega"]},
    {"id": "FORGE", "role": "code", "channels": ["#broadcast"]},
    {"id": "CODEX", "role": "review", "channels": ["#broadcast"]},
    {"id": "SENTINEL", "role": "security", "channels": ["#broadcast"]},
    {"id": "NEXUS", "role": "integration", "channels": ["#broadcast", "#github"]},
]
