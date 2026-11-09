# Card Benefits Agent (MCP server + AI agent)

**Live demo:** https://anand-ayush14.github.io/projects/benefits-agent/

Premium cards now come with many small statement credits, each with its own reset period (monthly, quarterly, semi-annual, annual) and merchant list. Card members and reviewers call this a "coupon book" that is hard to track, and third-party apps charge to track it. American Express's 2026 strategy puts agentic commerce first (its ACE developer kit and Agent Purchase Protection) and plans to make benefits usable inside AI assistants.

This project builds that piece:

| Tool | What it answers |
|---|---|
| `get_unused_credits` | What credit money is still open, and when does it expire? |
| `year_to_date` | How much did I use, and how much expired unused, per credit? |
| `value_vs_fee` | At my current pace, do the credits cover the annual fee? (renewal-risk signal) |
| `recommend_card_for_purchase` | Which of my cards should I use for this purchase (open credit + points)? |

The tools run three ways:

1. **MCP server** (`mcp_server.py`): any MCP client (Claude Desktop, IDE agents) can call them. Tested over stdio.
2. **LLM agent** (`agent.py`): a Claude tool-use loop that chooses the tools; needs `ANTHROPIC_API_KEY`.
3. **Browser demo** (`index.html`): the same engine ported to JavaScript with a rule-based intent parser, so it works on GitHub Pages with no key. The page checks its own results against the Python output.

## Data

- **Card rules (real):** `cards.json`, covering Platinum ($895), Gold ($325) and Delta SkyMiles Reserve ($650) credits as publicly described in November 2026 (sources listed in the file). Terms change, so check americanexpress.com.
- **Members and transactions (synthetic):** `demo_data.py` creates 3 made-up members with 196 purchases (Jan – Nov 2026). No real member data is used.

## Demo results (as of 9 Nov 2026)

| Member | Cards | Finding |
|---|---|---|
| A (Priya) | Platinum + Gold | Platinum credits beat the fee by ~$301/yr; Gold falls ~$101 short; $564 of credits expired unused (lululemon, Walmart+, half the digital-entertainment credit); Q4's $100 Platinum Resy credit is still open while regular Resy dinners go on Gold |
| B (Rahul) | Gold | Uses 42% of credits; credits cover < 60% of the fee → renewal risk 2 months before the January renewal |
| C (Sam) | Delta Reserve + Platinum | Missed all $300 of Platinum Resy credits so far; Delta Reserve credits ~$280 short of the fee |

## How the engine works

1. Each transaction counts toward at most **one** credit on the card used, picking the most specific merchant match. For example, `UBER ONE` goes to the Uber One credit, not Uber Cash.
2. For every credit and every period started this year, usage = min(credit amount, eligible spend). Periods that have ended become **used** or **missed**; the current one is **open**.
3. Annual credit potential × capture rate = projected annual credit value, compared with the fee.
4. Purchase recommendation = open credit value + points value (1¢ per Membership Rewards point, 1.1¢ per SkyMile, as conservative assumptions).

## Run it

```bash
pip install pandas mcp anthropic
python demo_data.py          # create the synthetic members
python test_engine.py        # sanity checks
python export.py             # tool outputs -> results.json, then: python ../build.py benefits-agent
python mcp_server.py         # MCP server over stdio
ANTHROPIC_API_KEY=... python agent.py --member A
```

Claude Desktop config (`claude_desktop_config.json`):

```json
{"mcpServers": {"amex-benefits-demo": {"command": "python", "args": ["/full/path/to/mcp_server.py"]}}}
```

## Limitations

- Calendar-year periods are assumed (Amex resets most credits on the calendar year); real eligibility depends on merchant codes, enrollment dates and terms.
- Synthetic members show how it works, not real behaviour. With real data, the next step is to test whether low credit capture predicts cancellation at renewal.

*Independent project; not affiliated with or endorsed by American Express.*
