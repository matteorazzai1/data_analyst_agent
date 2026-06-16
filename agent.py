import os
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llama_index.llms.openrouter import OpenRouter
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_CSV = DATA_DIR / "sample_wdi.csv"
FULL_WDI_CSV = DATA_DIR / "wdi_full.csv"
WDI_FOLDER = DATA_DIR / "WDI_CSV_2026_04_09"
WDI_MAIN_CSV = WDI_FOLDER / "WDICSV.csv"
WDI_SERIES_CSV = WDI_FOLDER / "WDISeries.csv"
WDI_COUNTRY_CSV = WDI_FOLDER / "WDICountry.csv"
WDI_FOOTNOTE_CSV = WDI_FOLDER / "WDIfootnote.csv"
WDI_COUNTRY_SERIES_CSV = WDI_FOLDER / "WDIcountry-series.csv"
WDI_SERIES_TIME_CSV = WDI_FOLDER / "WDIseries-time.csv"
INDICATORS_JSON = DATA_DIR / "indicators.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/owl-alpha")


class WDIStore:
    def __init__(self, csv_path: Path = None, indicators_path: Path = INDICATORS_JSON):
        # load indicators first (needed for download and fallback only)
        with open(indicators_path, "r", encoding="utf-8") as f:
            self.indicators = json.load(f)

        # if no csv_path provided, prefer the local WDI export folder, then full cached dataset, then sample
        if csv_path is None:
            if WDI_MAIN_CSV.exists():
                csv_path = WDI_MAIN_CSV
            elif FULL_WDI_CSV.exists():
                csv_path = FULL_WDI_CSV
            else:
                print("Full WDI dataset not found. Downloading...")
                self.download_full_wdi(FULL_WDI_CSV, limit=None)
                if FULL_WDI_CSV.exists():
                    csv_path = FULL_WDI_CSV
                else:
                    print("Download failed. Falling back to sample dataset.")
                    csv_path = SAMPLE_CSV

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
    
    def download_full_wdi(self, out_csv: Path, indicators: List[str] = None, limit: int = None):
        """Download WDI data for a list of indicators via World Bank bulk CSV endpoint.
        
        This may be slow and produce a large file. The function writes `out_csv` as a merged CSV.
        Use `limit` to restrict number of indicators for testing.
        """
        import requests
        import zipfile
        import io
        
        codes = indicators if indicators else [ind.get("indicator_code") or ind.get("id") for ind in self.indicators]
        if limit:
            codes = codes[:limit]
        
        rows = []
        for idx, code in enumerate(codes, 1):
            try:
                print(f"Downloading indicator {idx}/{len(codes)}: {code}...")
                url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?downloadformat=csv"
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    continue
                z = zipfile.ZipFile(io.BytesIO(resp.content))
                # find the data file inside zip (usually has 'API_' prefix)
                data_files = [n for n in z.namelist() if n.endswith('.csv') and 'Metadata' not in n]
                if not data_files:
                    continue
                with z.open(data_files[0]) as fh:
                    df_part = pd.read_csv(fh, encoding='latin1')
                    rows.append(df_part)
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        if rows:
            print(f"Merging {len(rows)} indicator files...")
            merged = pd.concat(rows, ignore_index=True)
            merged.to_csv(out_csv, index=False)
            print(f"Full WDI dataset saved to {out_csv}")
            return out_csv
        return None

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
        else:
            for ind in self.indicators:
                texts.append(" ".join([ind.get("name", ""), ind.get("source_note", ""), ind.get("description", "")] ))
                self.index_items.append(ind)

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
        self.agent = ReActAgent.from_tools(
            tools=[self.retrieve_docs_tool, self.run_python_tool],
            llm=self.llm,
            verbose=False,
            max_iterations=6,
        )

    def _retrieve_docs_wrapper(self, query: str) -> str:
        docs = self.store.retrieve_docs(query, top_n=5)
        return json.dumps(docs, ensure_ascii=False)

    def _run_python_wrapper(self, code: str) -> str:
        return run_python_sandbox(code, self.store.df)

    async def aquery(self, question: str) -> str:
        """Asynchronously send a question to the ReActAgent and return textual response."""
        response = await self.agent.aquery(question)
        # attempt to extract text
        try:
            text = str(response.response)
        except Exception:
            text = str(response)
        return text

    def query(self, question: str, timeout: int = 30) -> str:
        """Sync wrapper around async aquery."""
        return asyncio.run(self.aquery(question))


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
            print(ans)
        except Exception as e:
            print(f"Agent error: {e}")


if __name__ == "__main__":
    interactive()
