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
        self._build_metadata_index()
    
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
        # Lookup dicts for schema inspection: code_to_name and name_to_code
        self.code_to_name = {}  # Series Code -> Indicator Name mapping
        self.name_to_code = {}  # Normalized Indicator Name -> Series Code mapping

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
                series_code = row.get("Series Code")
                indicator_name = row.get("Indicator Name")
                self.index_items.append({
                    "indicator_code": series_code,
                    "indicator_name": indicator_name,
                    "short_definition": row.get("Short definition"),
                    "long_definition": row.get("Long definition"),
                    "source": row.get("Source"),
                })
                # Build lookup dicts
                if series_code:
                    self.code_to_name[series_code] = indicator_name
                if indicator_name:
                    self.name_to_code[indicator_name.lower()] = series_code
        
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

    def _build_metadata_index(self):
        """Build a separate TF-IDF index for metadata (footnotes, country info, data quality notes).
        
        This index provides disambiguation and data quality context for queries about
        data reliability, comparability, and methodological notes.
        """
        metadata_texts = []
        self.metadata_items = []
        
        # Index country metadata (regions, income groups, currency info)
        if self.country_df is not None:
            try:
                clean = self.country_df.rename(columns=lambda c: c.strip())
                for _, row in clean.iterrows():
                    country_code = row.get("Country Code")
                    country_name = row.get("Short Name") or row.get("Long Name")
                    region = row.get("Region", "")
                    income_group = row.get("Income Group", "")
                    
                    text = " ".join(
                        str(v or "")
                        for v in [country_name, region, income_group, "country metadata"]
                    )
                    metadata_texts.append(text)
                    self.metadata_items.append({
                        "type": "country",
                        "code": country_code,
                        "name": country_name,
                        "region": region,
                        "income_group": income_group,
                    })
            except Exception:
                pass
        
        # Index footnotes (data quality notes, revisions, limitations)
        if self.footnote_df is not None:
            try:
                clean = self.footnote_df.rename(columns=lambda c: c.strip())
                for _, row in clean.iterrows():
                    footnote_text = row.get("footnote", "") or row.get("Footnote", "")
                    country_code = row.get("Country Code", "")
                    series_code = row.get("Series Code", "")
                    year = row.get("year", "")
                    
                    text = " ".join(
                        str(v or "")
                        for v in [footnote_text, country_code, series_code, year, "data quality note"]
                    )
                    metadata_texts.append(text)
                    self.metadata_items.append({
                        "type": "footnote",
                        "text": footnote_text,
                        "country_code": country_code,
                        "series_code": series_code,
                        "year": year,
                    })
            except Exception:
                pass
        
        # Build metadata vectorizer if we have any metadata
        if metadata_texts:
            self.metadata_vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
            self.metadata_tfidf = self.metadata_vectorizer.fit_transform(metadata_texts)
        else:
            self.metadata_vectorizer = None
            self.metadata_tfidf = None

    def retrieve_metadata(self, query: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top metadata entries (country info, footnotes, data quality notes) matching the query.
        
        Useful for:
        - Checking data comparability across countries
        - Understanding data quality and revisions
        - Identifying country-specific limitations or regions
        
        Returns: List of matching metadata items with similarity scores.
        """
        if self.metadata_tfidf is None:
            return []
        
        try:
            qv = self.metadata_vectorizer.transform([query])
            scores = cosine_similarity(qv, self.metadata_tfidf).flatten()
            idx = scores.argsort()[::-1][:top_n]
            res = []
            for i in idx:
                item = dict(self.metadata_items[i])
                item["score"] = float(scores[i])
                res.append(item)
            return res
        except Exception:
            return []

    def get_schema_info(self) -> Dict[str, Any]:
        """Get DataFrame schema: columns, dtypes, year range, unique counts."""
        return {
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "year_range": [
                int(self.df["year"].min()) if "year" in self.df.columns else None,
                int(self.df["year"].max()) if "year" in self.df.columns else None,
            ],
            "unique_countries": int(self.df["Country Code"].nunique()) if "Country Code" in self.df.columns else 0,
            "unique_indicators": int(self.df["Series Code"].nunique()) if "Series Code" in self.df.columns else 0,
            "row_count": len(self.df),
            "total_missing_values": int(self.df["value"].isna().sum()) if "value" in self.df.columns else 0,
        }

    def lookup_indicator(self, query: str) -> Dict[str, Any]:
        """Lookup indicator by code (exact) or name (fuzzy match).
        Returns: {code, name, definition, source, ...
        }"""
        # Try exact match on code
        if query.upper() in self.code_to_name:
            code = query.upper()
            name = self.code_to_name[code]
            # Find the full item from index_items
            for item in self.index_items:
                if item.get("indicator_code") == code:
                    return {
                        "code": code,
                        "name": name,
                        "short_definition": item.get("short_definition"),
                        "long_definition": item.get("long_definition"),
                        "source": item.get("source"),
                        "match_type": "exact_code",
                    }
        
        # Try fuzzy match on name (substring search)
        query_lower = query.lower()
        for name_key, code in self.name_to_code.items():
            if query_lower in name_key:
                # Found a match, return full info
                for item in self.index_items:
                    if item.get("indicator_code") == code:
                        return {
                            "code": code,
                            "name": item.get("indicator_name"),
                            "short_definition": item.get("short_definition"),
                            "long_definition": item.get("long_definition"),
                            "source": item.get("source"),
                            "match_type": "fuzzy_name",
                        }
        
        # No match found
        return {
            "error": f"No indicator found matching '{query}'",
            "suggestion": "Try using retrieve_docs() for semantic search.",
        }


def run_python_sandbox(code: str, df: pd.DataFrame) -> str:
    """Execute provided python snippet in a restricted sandbox and return result as string.
    
    Sandbox execution with carefully curated builtins:
    - Math operations (abs, pow, divmod)
    - Statistics via numpy (mean, std, median, var)
    - Type operations (isinstance, type, bool)
    - Iteration (enumerate, zip, map, filter, range)
    - NOT: exec, eval, import, file I/O (security)
    """
    safe_globals = {
        "pd": pd,
        "np": np,
        "df": df.copy(),
        "__builtins__": {
            # Base functions
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
            # Math functions
            "abs": abs,
            "pow": pow,
            "divmod": divmod,
            # Statistics via numpy
            "mean": lambda x: float(np.mean(x)),
            "std": lambda x: float(np.std(x)),
            "median": lambda x: float(np.median(x)),
            "var": lambda x: float(np.var(x)),
            # Type checks
            "isinstance": isinstance,
            "type": type,
            "bool": bool,
            # Iteration
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "range": range,
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


# System prompt for the DataAnalystAgent with explicit 2-phase strategy
SYSTEM_PROMPT = """You are a Data Analyst Agent specialized in World Development Indicator (WDI) analysis.

Your goal is to help users understand WDI data and answer analytical questions using a 2-phase strategy:

**PHASE 1: Determine and verify indicator(s)**
- If the question asks about indicator definition, methodology, units, or source → Use retrieve_docs() to search for the indicator
- If the question mentions an ambiguous concept (e.g., "poverty", "GDP", "employment") → Use retrieve_docs() first to find matching indicators, then use inspect_schema() to confirm the exact Series Code
- Use inspect_schema() to look up indicators by code (e.g., 'SI.POV.DDAY') or by name keyword

**PHASE 2: Retrieve and compute data**
- If the question requires specific data values or calculations → Use run_python() with verified Series Code
- IMPORTANT: Always filter by Series Code (e.g., 'SI.POV.DDAY'), NOT by Indicator Name, to ensure consistency
- Filter by Country Code (use country names, abbreviations like 'USA', 'IND', 'CHN', or country codes)
- Filter by year range if applicable
- If data returns empty or NaN, adjust filters, check country codes, or verify year range

**Tool Usage Guidelines:**
- inspect_schema(): Schema exploration, indicator code ↔ name lookup, DataFrame structure validation
- retrieve_docs(): Semantic search for indicator documentation (definitions, methodologies, sources)
- run_python(): Execute pandas/numpy code to filter, aggregate, compute on the DataFrame

**Important Rules:**
1. Always assign final answer to variable named `result` in Python snippets
2. Use Series Code for filtering, never Indicator Name
3. When uncertain about an indicator, verify its code first using retrieve_docs() + inspect_schema()
4. Be clear about data limitations (missing years, countries, or values)
"""


class DataAnalystAgent:
    """Agent that uses OpenRouter LLM + llama_index tooling to combine RAG and code execution.
    
    Follows a 2-phase strategy (see SYSTEM_PROMPT):
    PHASE 1: Semantic retrieval (retrieve_docs) + schema inspection (inspect_schema) for disambiguation
    PHASE 2: Python execution (run_python) with verified Series Code + filters
    """

    def __init__(self):
        self.store = WDIStore()
        # init OpenRouter LLM wrapper
        self.llm = OpenRouter(api_key=OPENROUTER_API_KEY, max_tokens=1024, context_window=4096, model=OPENROUTER_MODEL)

        # define tools
        self.inspect_schema_tool = FunctionTool.from_defaults(
            fn=self._inspect_schema_wrapper,
            name="inspect_schema",
            description="Get DataFrame schema (columns, dtypes, year range) OR lookup indicator by code/name. Input: optional query string (e.g., 'SI.POV.DDAY' or 'poverty'). Returns JSON with schema metadata or indicator details.",
        )

        self.retrieve_docs_tool = FunctionTool.from_defaults(
            fn=self._retrieve_docs_wrapper,
            name="retrieve_docs",
            description="Retrieve top indicator documentation entries matching the query using semantic search. Input: a query string (e.g., 'poverty headcount'). Returns JSON with top matching indicators, definitions, and sources.",
        )

        self.run_python_tool = FunctionTool.from_defaults(
            fn=self._run_python_wrapper,
            name="run_python",
            description="Execute a Python snippet operating on the DataFrame `df`. The snippet must assign the final answer to variable named `result`. Use Series Code (e.g., 'SI.POV.DDAY') not Indicator Name for filtering. Input: code string. Returns serialized result.",
        )

        # build ReAct agent with explicit strategy in tool ordering and system prompt
        self.agent = ReActAgent(
            name="DataAnalystAgent",
            tools=[self.inspect_schema_tool, self.retrieve_docs_tool, self.run_python_tool],
            llm=self.llm,
            verbose=True,
            max_iterations=5,
            system_prompt=SYSTEM_PROMPT
        )

    def _inspect_schema_wrapper(self, query: str = "") -> str:
        """Wrapper for schema inspection tool."""
        if query.strip():
            # Lookup specific indicator by code/name
            result = self.store.lookup_indicator(query)
        else:
            # Return general schema info
            result = self.store.get_schema_info()
        return json.dumps(result, ensure_ascii=False)

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
            print("Answer ////////////////////////////////////////")
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
