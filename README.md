# Agentic Equity Research Assistant

A **multi-agent** system that researches a stock the way an investment desk does.
Role-based **Fundamental**, **Technical**, and **Risk** analyst agents each do one
job, then a **Portfolio-Manager** agent runs a critic/review loop over their findings
and issues a **BUY / HOLD / SELL** research note with cited rationale.

Built with **CrewAI** (agent orchestration + LLM tool-calling), **yfinance** for live
market data, and a pluggable LLM backend.

---

## Architecture

```
            ┌─────────────────────────────────────────┐
  Ticker →  │        Portfolio Manager Agent           │ → Research Note
            │   (orchestrates + critic/review loop)     │   (BUY/HOLD/SELL
            └───────┬───────────┬───────────┬──────────┘    + rationale + risk)
                    │           │           │
            ┌───────▼──┐  ┌─────▼────┐  ┌───▼──────┐
            │Fundamental│  │Technical │  │   Risk   │   ← role-based agents
            │  Analyst  │  │ Analyst  │  │ Analyst  │
            └─────┬─────┘  └────┬─────┘  └────┬─────┘
                  └─────────────┴─────────────┘
                     Stock Data tool (yfinance)
```

### How it works

1. **Fundamental Analyst** pulls financials and judges valuation from P/E, ROE, D/E.
2. **Technical Analyst** reads price action against the 50-day moving average.
3. **Risk Analyst** computes beta and annualized volatility and flags downside risk.
4. **Portfolio Manager** reconciles the three views, **challenges weak claims**
   (critic loop), and issues the final call. A human makes the actual trade — the
   system is decision *support*, not an auto-trader.

### Why multi-agent?

Separation of concerns: each agent reasons only within its domain and uses its own
prompt + tools, which keeps the analysis legible and the disagreements explicit —
mirroring how a real research desk divides work (see the
[AlphaAgents](https://arxiv.org/abs/2508.11152) multi-agent equity-research paper).

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your free GROQ_API_KEY
python research_crew.py AAPL TCS.NS
```

Default LLM is Groq's free tier (no credit card). To use Gemini / OpenAI / Anthropic,
change the `LLM` constant near the top of `research_crew.py` and set the matching key.

A sample generated note lives in [`samples/`](samples/).

---

## Tech stack

| Concern | Choice |
|---|---|
| Agent orchestration | CrewAI (sequential process, shared task context) |
| Tool-calling | CrewAI `@tool` (MCP-compatible pattern) |
| Market data | yfinance |
| LLM | Groq / Gemini / OpenAI / Anthropic (pluggable) |

## Limitations & guardrails

- **Not investment advice.** Decision-support only; a human issues the final call.
- Data is delayed/free-tier; valuation fields can be missing for some tickers.
- The LLM can be wrong — the critic loop reduces, but does not eliminate, weak reasoning.

## Roadmap

- [ ] Expose the `stock_data` tool as a standalone **MCP server**.
- [ ] Add a news/filing retrieval (RAG) agent for qualitative context.
- [ ] Backtest the recommendations against forward returns.
