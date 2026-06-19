import os
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llama_index.llms.openrouter import OpenRouter
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

print(ReActAgent)
print(ReActAgent.__module__)


load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WDI_FOLDER = DATA_DIR / "WDI_CSV_2026_04_09"
WDI_MAIN_CSV = WDI_FOLDER / "WDICSV.csv"
WDI_SERIES_CSV = WDI_FOLDER / "WDISeries.csv"
WDI_COUNTRY_CSV = WDI_FOLDER / "WDICountry.csv"
WDI_FOOTNOTE_CSV = WDI_FOLDER / "WDIfootnote.csv"
WDI_COUNTRY_SERIES_CSV = WDI_FOLDER / "WDIcountry-series.csv"
WDI_SERIES_TIME_CSV = WDI_FOLDER / "WDIseries-time.csv"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/owl-alpha")

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment or add it to a .env file."
    )


class WDIStore:
    def __init__(self, csv_path: Path = None):
        # Use only the imported WDI dataset files present in WDI_CSV_2026_04_09.
        if csv_path is None:
            csv_path = WDI_MAIN_CSV

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Required dataset file not found: {csv_path}."
            )

        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self.series_df = self._read_csv(WDI_SERIES_CSV)
        self.country_df = self._read_csv(WDI_COUNTRY_CSV)
        self.footnote_df = self._read_csv(WDI_FOOTNOTE_CSV)
        self.country_series_df = self._read_csv(WDI_COUNTRY_SERIES_CSV)
        self.series_time_df = self._read_csv(WDI_SERIES_TIME_CSV)

        if csv_path == WDI_MAIN_CSV:
            self._normalize_wdi_csv()

        if "year" in self.df.columns:
            try:
                self.df["year"] = self.df["year"].astype(int)
            except Exception:
                pass

        self._build_index()
    
    def _read_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
        return None

    def _normalize_wdi_csv(self):
        year_cols = [c for c in self.df.columns if c.isdigit()]
        if not year_cols:
            return

        id_vars = [c for c in self.df.columns if c not in year_cols]
        self.df = self.df.melt(
            id_vars=id_vars,
            value_vars=year_cols,
            var_name="year",
            value_name="value",
        )
        self.df["year"] = pd.to_numeric(self.df["year"], errors="coerce").astype("Int64")

    def _build_index(self):
        texts = []
        self.index_items = []

        if self.series_df is not None:
            clean = self.series_df.rename(columns=lambda c: c.strip())
            for _, row in clean.iterrows():
                text = " ".join(
                    str(row.get(col, "") or "")
                    for col in [
                        "Indicator Name",
                        "Short definition",
                        "Long definition",
                        "Other notes",
                        "Source",
                        "Statistical concept and methodology",
                        "Development relevance",
                    ]
                )
                texts.append(text)
                self.index_items.append({
                    "indicator_code": row.get("Series Code"),
                    "indicator_name": row.get("Indicator Name"),
                    "short_definition": row.get("Short definition"),
                    "long_definition": row.get("Long definition"),
                    "source": row.get("Source"),
                })
        
        self.vectorizer = TfidfVectorizer(stop_words="english")
        if texts:
            self.tfidf = self.vectorizer.fit_transform(texts)
        else:
            self.tfidf = None

    def retrieve_docs(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top indicator documentation entries matching the query using TF-IDF similarity."""
        if self.tfidf is None:
            return []
        qv = self.vectorizer.transform([query])
        scores = cosine_similarity(qv, self.tfidf).flatten()
        idx = scores.argsort()[::-1][:top_n]
        res = []
        for i in idx:
            item = dict(self.index_items[i])
            item["score"] = float(scores[i])
            res.append(item)
        return res


def run_python_sandbox(code: str, df: pd.DataFrame) -> str:
    """Execute provided python snippet in a restricted sandbox and return result as string."""
    safe_globals = {
        "pd": pd,
        "np": np,
        "df": df.copy(),
        "__builtins__": {
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,
            "sorted": sorted,
            "list": list,
            "dict": dict,
            "float": float,
            "int": int,
            "str": str,
        },
    }
    local = {}
    try:
        exec(code, safe_globals, local)
        if "result" in local:
            return json.dumps(local["result"], default=str, ensure_ascii=False)
        else:
            return "Error: executed code did not assign 'result' variable."
    except Exception as e:
        return f"Execution error: {str(e)}"


class DataAnalystAgent:
    """Agent that uses OpenRouter LLM + llama_index tooling to combine RAG and code execution."""

    def __init__(self):
        self.store = WDIStore()
        # init OpenRouter LLM wrapper
        self.llm = OpenRouter(api_key=OPENROUTER_API_KEY, max_tokens=1024, context_window=4096, model=OPENROUTER_MODEL)

        # define tools
        self.retrieve_docs_tool = FunctionTool.from_defaults(
            fn=self._retrieve_docs_wrapper,
            name="retrieve_docs",
            description="Retrieve top indicator documentation entries matching the query. Input: a short query string. Returns JSON string of top results.",
        )

        self.run_python_tool = FunctionTool.from_defaults(
            fn=self._run_python_wrapper,
            name="run_python",
            description="Execute a small Python snippet operating on the in-memory DataFrame `df`. The snippet must assign the final answer to a variable named `result`. Input: code string. Returns serialized result.",
        )

        # build ReAct agent
        self.agent = ReActAgent(
            name="DataAnalystAgent",
            tools=[self.retrieve_docs_tool, self.run_python_tool],
            llm=self.llm,
            verbose=True,
            max_iterations=5
        )

        print(dir(self.agent))

    def _retrieve_docs_wrapper(self, query: str) -> str:
        docs = self.store.retrieve_docs(query, top_n=5)
        return json.dumps(docs, ensure_ascii=False)

    def _run_python_wrapper(self, code: str) -> str:
        return run_python_sandbox(code, self.store.df)

    async def aquery(self, question: str) -> str:
        try:
            response = await self.agent.run(
                user_msg=question,
                max_iterations=5,
                early_stopping_method="generate",
            )
            return str(response)
        except Exception as e:
            # If the workflow still hits iteration limits, retry with an explicit
            # early stopping method to force generation of a final answer.
            msg = str(e)
            if "Max iterations" in msg or "max_iterations" in msg:
                response = await self.agent.run(user_msg=question, early_stopping_method="generate")
                return str(response)
            raise

    def query(self, question: str, timeout: int = 30) -> str:
        """Sync wrapper around async aquery."""
        return asyncio.run(self.aquery(question))

    def answer(self, question: str, timeout: int = 30) -> str:
        """Compatibility helper for run_examples.py."""
        return self.query(question, timeout=timeout)

    def format_answer(self, answer: Any) -> str:
        """Normalize returned answers for printing."""
        if isinstance(answer, str):
            return answer
        try:
            return str(answer)
        except Exception:
            return json.dumps(answer, ensure_ascii=False)


def interactive():
    agent = DataAnalystAgent()
    print("Data Analyst Agent (OpenRouter + llama_index). Type 'exit' to quit.")
    while True:
        q = input("Question> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("Thinking... (this may take a few seconds)")
        try:
            ans = agent.query(q)
            print("Answer://///////////////////////////////////\n")
            print(ans)
        except Exception as e:
            print(f"Agent error: {e}")


if __name__ == "__main__":
    interactive()
