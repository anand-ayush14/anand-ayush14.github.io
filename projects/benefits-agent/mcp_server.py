"""
MCP server: exposes the benefits engine as tools any MCP client (Claude Desktop, an AI agent, an IDE)
can call. Run:  python mcp_server.py   (stdio transport)

Claude Desktop config (claude_desktop_config.json):
  {"mcpServers": {"amex-benefits-demo": {"command": "python", "args": ["/full/path/to/mcp_server.py"]}}}
"""
from datetime import date

import engine

try:                                   # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:                    # mcp 1.x
    from mcp.server.fastmcp import FastMCP as Server

server = Server("amex-benefits-demo", instructions=(
    "Tools for a card member's statement credits. Members in the demo: A, B, C (synthetic data). "
    "Dates are ISO strings; default as_of is today."))
TX = engine.load_transactions()


def _d(as_of):
    return date.fromisoformat(as_of) if as_of else date.today()


@server.tool()
def get_unused_credits(member: str, as_of: str = "") -> list:
    """Credits still open in the current period with money left, soonest-expiring first."""
    return engine.get_unused_credits(TX, member, _d(as_of))


@server.tool()
def year_to_date(member: str, as_of: str = "") -> list:
    """Per credit this calendar year: amount available so far, used, and missed (expired unused)."""
    return engine.year_to_date(TX, member, _d(as_of))


@server.tool()
def value_vs_fee(member: str, as_of: str = "") -> list:
    """For each card: credits used, projected annual credit value and the gap to the annual fee."""
    return engine.value_vs_fee(TX, member, _d(as_of))


@server.tool()
def recommend_card_for_purchase(member: str, merchant: str, amount: float, as_of: str = "") -> dict:
    """Which of the member's cards gives the most value for a purchase (open credit + points)."""
    return engine.recommend_card_for_purchase(TX, member, _d(as_of), merchant, amount)


if __name__ == "__main__":
    server.run()
