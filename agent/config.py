"""
Configuration module for DataAnalystAgent.
Loads environment variables and defines constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/owl-alpha")

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment or add it to a .env file."
    )

# Example questions for the Gradio interface
EXAMPLES = [
    "What was the GDP of India in 2020?",
    "What does the indicator SI.POV.DDAY measure?",
    "Compare CO2 emissions per capita between the United States and China in 2020.",
    "Is GDP data comparable across countries in WDI?",
    "What was the trend in unemployment rate for Brazil between 2010 and 2020?",
    "Show me the share of renewable energy in GDP."
]
