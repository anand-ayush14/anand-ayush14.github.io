"""
LLM agent: a chat loop where Claude decides which benefits tools to call.
Needs an Anthropic API key:  export ANTHROPIC_API_KEY=...   then   python agent.py --member A
Without a key, use the browser demo (index.html) or the MCP server instead.
"""
import argparse
import json
import os
from datetime import date

import anthropic

import engine

MODEL = os.environ.get("AGENT_MODEL", "claude-sonnet-5-5")
TX = engine.load_transactions()
TOOLS = [
    {"name": "get_unused_credits", "description": "Credits still open this period with money left, soonest-expiring first.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "year_to_date", "description": "Per credit this year: available so far, used, missed.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "value_vs_fee", "description": "Per card: credits used, projected annual credit value vs the annual fee.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "recommend_card_for_purchase", "description": "Best card for a purchase (open credits + points).",
     "input_schema": {"type": "object", "properties": {"merchant": {"type": "string"}, "amount": {"type": "number"}},
                      "required": ["merchant", "amount"]}},
]


def run_tool(name, args, member, as_of):
    fn = getattr(engine, name)
    if name == "recommend_card_for_purchase":
        return fn(TX, member, as_of, args["merchant"], float(args["amount"]))
    return fn(TX, member, as_of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="A")
    ap.add_argument("--as-of", default="2026-11-09")
    a = ap.parse_args()
    as_of = date.fromisoformat(a.as_of)
    client = anthropic.Anthropic()
    system = (f"You help an American Express card member (demo member {a.member}, synthetic data) get value from "
              f"their card benefits. Today is {as_of}. Use the tools; never invent amounts. Be brief and specific.")
    messages = []
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"", "quit", "exit"}:
            break
        messages.append({"role": "user", "content": q})
        while True:
            resp = client.messages.create(model=MODEL, max_tokens=800, system=system, tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                print("\nAgent:", "".join(b.text for b in resp.content if b.type == "text"))
                break
            results = []
            for c in calls:
                print(f"  [tool] {c.name}({json.dumps(c.input)})")
                results.append({"type": "tool_result", "tool_use_id": c.id,
                                "content": json.dumps(run_tool(c.name, c.input, a.member, as_of))})
            messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    main()
