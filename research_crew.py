"""
Agentic Equity Research Assistant
---------------------------------
A multi-agent system that researches a stock the way an investment desk does:
role-based Fundamental / Technical / Risk analyst agents feed a Portfolio-Manager
agent, which runs a critic/review loop and issues a BUY / HOLD / SELL note.

Stack: CrewAI (agent orchestration) + yfinance (live market data) + an LLM
(Groq free tier by default; swap LLM below for Gemini / OpenAI / Anthropic).

Usage:
    pip install -r requirements.txt
    cp .env.example .env          # then add your GROQ_API_KEY
    python research_crew.py AAPL TCS.NS
"""

import sys
import yfinance as yf
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

load_dotenv()

# Free, no-credit-card default. Swap for "gemini/gemini-1.5-flash",
# "openai/gpt-4o-mini", or "anthropic/claude-3-5-haiku-latest" as you like.
LLM = "groq/llama-3.3-70b-versatile"


@tool("Stock Data")
def stock_data(ticker: str) -> str:
    """Fetch fundamentals, recent price action and risk stats for a ticker."""
    t = yf.Ticker(ticker)
    info = t.info or {}
    hist = t.history(period="6mo")
    if hist.empty:
        return f"No data found for {ticker}."
    last = hist["Close"].iloc[-1]
    sma50 = hist["Close"].rolling(50).mean().iloc[-1]
    daily_ret = hist["Close"].pct_change().dropna()
    vol = daily_ret.std() * (252 ** 0.5)
    return (
        f"{ticker}: price={last:.2f}, "
        f"PE={info.get('trailingPE')}, "
        f"ROE={info.get('returnOnEquity')}, "
        f"debtToEquity={info.get('debtToEquity')}, "
        f"beta={info.get('beta')}, "
        f"50d_SMA={sma50:.2f}, "
        f"annualized_vol={vol:.2%}, "
        f"sector={info.get('sector')}"
    )


fundamental = Agent(
    role="Fundamental Analyst",
    goal="Judge valuation and financial health from ratios (P/E, ROE, D/E).",
    backstory="A disciplined value analyst trained on NISM research methods.",
    tools=[stock_data], llm=LLM, verbose=True,
)
technical = Agent(
    role="Technical Analyst",
    goal="Read price action against the 50-day moving average to judge momentum.",
    backstory="A chartist focused on trend direction and entry timing.",
    tools=[stock_data], llm=LLM, verbose=True,
)
risk = Agent(
    role="Risk Analyst",
    goal="Quantify volatility and beta, and flag downside risk.",
    backstory="A risk manager who sizes positions by risk-adjusted return.",
    tools=[stock_data], llm=LLM, verbose=True,
)
pm = Agent(
    role="Portfolio Manager",
    goal="Reconcile the three analysts and issue a BUY/HOLD/SELL call with rationale.",
    backstory="A PM who demands evidence and pushes back on weak arguments.",
    llm=LLM, verbose=True,
)


def build_crew(ticker: str) -> Crew:
    t1 = Task(description=f"Analyse the fundamentals of {ticker}.",
              agent=fundamental,
              expected_output="A valuation verdict backed by 3 ratios.")
    t2 = Task(description=f"Analyse the technicals of {ticker}.",
              agent=technical,
              expected_output="A momentum verdict vs the 50-day SMA.")
    t3 = Task(description=f"Analyse the risk profile of {ticker}.",
              agent=risk,
              expected_output="Beta, volatility and a risk flag.")
    t4 = Task(
        description=(
            f"Reconcile the fundamental, technical and risk views on {ticker}. "
            "Challenge any weak or unsupported claim before deciding. "
            "Produce a one-page research note: recommendation (BUY/HOLD/SELL), "
            "rationale citing the analysts' evidence, and the single biggest risk."
        ),
        agent=pm, context=[t1, t2, t3],
        expected_output="A concise BUY/HOLD/SELL research note.",
    )
    return Crew(agents=[fundamental, technical, risk, pm],
                tasks=[t1, t2, t3, t4],
                process=Process.sequential, verbose=True)


def main():
    tickers = sys.argv[1:] or ["AAPL"]
    for tk in tickers:
        print(f"\n{'=' * 64}\n  RESEARCH NOTE: {tk}\n{'=' * 64}")
        result = build_crew(tk).kickoff()
        print(result)


if __name__ == "__main__":
    main()
