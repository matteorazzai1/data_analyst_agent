import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.agent import DataAnalystAgent

EXAMPLES = [
    "What was the GDP of India in 2020?",
    "What does the indicator SI.POV.DDAY measure?",
    "Compare CO2 emissions per capita between the United States and China in 2020.",
    "Is GDP data comparable across countries in WDI?",
    "What was the trend in unemployment rate for Brazil between 2010 and 2020, and what should I know from the definition?",
    "Show me the share of renewable energy in GDP."
]


def run_examples():
    agent = DataAnalystAgent()
    for question in EXAMPLES:
        print("=== QUERY ===")
        print(question)
        print()
        answer = agent.answer(question)
        print(agent.format_answer(answer))
        print("\n")


if __name__ == "__main__":
    run_examples()
