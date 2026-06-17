from src.plugins.base import PluginAdapter
class JarvisPlugin(PluginAdapter):
    name = "jarvis"
    async def health(self): return {"healthy": False, "note": "stub"}
    async def list_tools(self): return []
    async def invoke_tool(self, n, a): raise NotImplementedError(n)
