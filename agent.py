import os
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from llama_index.llms.openrouter import OpenRouter
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

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

# Modello di embedding: locale, gratuito, dimensione contenuta (384-dim, ~80MB).
# Sufficiente per la scala del dataset WDI (poche migliaia di chunk totali).
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "Missing OpenRouter API key. Set OPENROUTER_API_KEY in your environment or add it to a .env file."
    )


# =====================================================================
# SEMANTIC INDEX — wrapper FAISS + sentence-transformers
# =====================================================================
class SemanticIndex:
    """Indice vettoriale generico: embedding + FAISS (cosine similarity via
    normalizzazione L2 + IndexFlatIP). Ogni voce porta con sé un dict di
    metadata arbitrario, restituito tale e quale in fase di query.
    """

    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder
        self.index: Optional[faiss.Index] = None
        self.items: List[Dict[str, Any]] = []

    def build(self, texts: List[str], items: List[Dict[str, Any]]):
        assert len(texts) == len(items), "texts e items devono avere la stessa lunghezza"
        self.items = items
        if not texts:
            self.index = None
            return

        embeddings = self.encoder.encode(
            texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(embeddings)  # necessario per usare IndexFlatIP come cosine similarity

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_n: int = 5,
        filter_fn: Optional[callable] = None,
        over_fetch: int = 4,
    ) -> List[Dict[str, Any]]:
        """Cerca i top_n item più simili semanticamente alla query.

        filter_fn: funzione opzionale (item -> bool) per pre/post-filtrare
        i risultati per metadata (es. solo footnote di un certo indicator_code).
        Quando è presente un filtro, "sovra-peschiamo" più candidati prima di
        filtrare, perché FAISS IndexFlatIP non supporta filtri nativi.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        qv = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(qv)

        k = top_n * over_fetch if filter_fn else top_n
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(qv, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            item = dict(self.items[idx])
            if filter_fn and not filter_fn(item):
                continue
            item["score"] = float(score)
            results.append(item)
            if len(results) >= top_n:
                break
        return results


# =====================================================================
# WDI STORE — caricamento dati + chunking + indici semantici
# =====================================================================
class WDIStore:
    def __init__(self, csv_path: Path = None):
        if csv_path is None:
            csv_path = WDI_MAIN_CSV
        if not csv_path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {csv_path}.")

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

        print(f"[WDIStore] Caricamento embedding model '{EMBEDDING_MODEL_NAME}'...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # lookup dicts (codice <-> nome), invariati rispetto alla versione TF-IDF
        self.code_to_name: Dict[str, str] = {}
        self.name_to_code: Dict[str, str] = {}

        self.indicator_index = SemanticIndex(self.encoder)
        self.metadata_index = SemanticIndex(self.encoder)

        self._build_indicator_index()
        self._build_metadata_index()

    def _read_csv(self, path: Path, **kwargs) -> Optional[pd.DataFrame]:
        if path.exists():
            return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
        return None

    def _normalize_wdi_csv(self):
        year_cols = [c for c in self.df.columns if c.isdigit()]
        if not year_cols:
            return
        id_vars = [c for c in self.df.columns if c not in year_cols]
        self.df = self.df.melt(
            id_vars=id_vars, value_vars=year_cols, var_name="year", value_name="value"
        )
        self.df["year"] = pd.to_numeric(self.df["year"], errors="coerce").astype("Int64")

    # -----------------------------------------------------------------
    # CHUNKING — indicatori (series): un chunk per campo logico, non un
    # unico blob concatenato. Vedi discussione in sezione 8.2 della
    # documentazione: una long_definition lunga annacqua il segnale
    # embedding se mescolata con una short_definition concisa.
    # -----------------------------------------------------------------
    FIELD_LABELS = {
        "short_definition": "Definizione breve",
        "long_definition": "Definizione estesa",
        "methodology": "Metodologia statistica",
        "development_relevance": "Rilevanza per lo sviluppo",
    }

    def _build_indicator_index(self):
        if self.series_df is None:
            return

        clean = self.series_df.rename(columns=lambda c: c.strip())
        texts: List[str] = []
        items: List[Dict[str, Any]] = []

        for _, row in clean.iterrows():
            series_code = row.get("Series Code")
            indicator_name = row.get("Indicator Name")
            source = row.get("Source")
            if not series_code or not indicator_name:
                continue

            self.code_to_name[series_code] = indicator_name
            self.name_to_code[str(indicator_name).lower()] = series_code

            fields = {
                "short_definition": row.get("Short definition"),
                "long_definition": row.get("Long definition"),
                "methodology": row.get("Statistical concept and methodology"),
                "development_relevance": row.get("Development relevance"),
            }

            for field_key, field_value in fields.items():
                field_value = str(field_value or "").strip()
                if not field_value or field_value.lower() == "nan":
                    continue  # non indicizzare chunk vuoti

                # Il nome dell'indicatore viene ripetuto in ogni chunk per
                # dare contesto all'embedding anche quando il chunk viene
                # recuperato isolatamente (senza gli altri campi accanto).
                chunk_text = f"{indicator_name}. {self.FIELD_LABELS[field_key]}: {field_value}"
                texts.append(chunk_text)
                items.append({
                    "indicator_code": series_code,
                    "indicator_name": indicator_name,
                    "field_type": field_key,
                    "text": field_value,
                    "source": source,
                })

        print(f"[WDIStore] Indicizzazione semantica: {len(texts)} chunk indicatori "
              f"(da {clean['Series Code'].nunique()} indicatori).")
        self.indicator_index.build(texts, items)

    # -----------------------------------------------------------------
    # CHUNKING — footnote: il record è già il chunk naturale (una nota =
    # un fatto puntuale paese+indicatore+anno). L'unica accortezza è
    # arricchire il testo con contesto leggibile prima dell'embedding,
    # invece di indicizzare la nota grezza isolata dal suo contesto.
    #
    # CHUNKING — country: pochi record, corti, stabili: un record = un
    # chunk, nessuno splitting necessario.
    # -----------------------------------------------------------------
    def _build_metadata_index(self):
        texts: List[str] = []
        items: List[Dict[str, Any]] = []

        if self.country_df is not None:
            try:
                clean = self.country_df.rename(columns=lambda c: c.strip())
                for _, row in clean.iterrows():
                    country_code = row.get("Country Code")
                    country_name = row.get("Short Name") or row.get("Long Name")
                    region = str(row.get("Region", "") or "")
                    income_group = str(row.get("Income Group", "") or "")
                    if not country_name:
                        continue

                    chunk_text = (
                        f"Paese: {country_name}. Regione: {region}. "
                        f"Fascia di reddito: {income_group}."
                    )
                    texts.append(chunk_text)
                    items.append({
                        "type": "country",
                        "code": country_code,
                        "name": country_name,
                        "region": region,
                        "income_group": income_group,
                    })
            except Exception as e:
                print(f"[WDIStore] Avviso: indicizzazione country fallita: {e}")

        if self.footnote_df is not None:
            try:
                clean = self.footnote_df.rename(columns=lambda c: c.strip())
                for _, row in clean.iterrows():
                    footnote_text = str(row.get("footnote", "") or row.get("Footnote", "") or "").strip()
                    if not footnote_text or footnote_text.lower() == "nan":
                        continue
                    country_code = row.get("Country Code", "")
                    series_code = row.get("Series Code", "")
                    year = row.get("year", "") or row.get("Year", "")
                    indicator_name = self.code_to_name.get(series_code, series_code)

                    # Contesto esplicito: senza questo, l'embedding di una
                    # nota tipo "Estimated value" perde ogni significato
                    # se isolata dal paese/indicatore/anno a cui si riferisce.
                    chunk_text = (
                        f"Nota sui dati per {country_code}, indicatore "
                        f"{indicator_name} ({series_code}), anno {year}: {footnote_text}"
                    )
                    texts.append(chunk_text)
                    items.append({
                        "type": "footnote",
                        "text": footnote_text,
                        "country_code": country_code,
                        "series_code": series_code,
                        "year": year,
                    })
            except Exception as e:
                print(f"[WDIStore] Avviso: indicizzazione footnote fallita: {e}")

        print(f"[WDIStore] Indicizzazione semantica: {len(texts)} chunk metadata "
              f"(country + footnote).")
        self.metadata_index.build(texts, items)

    # -----------------------------------------------------------------
    # API di retrieval — sostituiscono le versioni TF-IDF
    # -----------------------------------------------------------------
    def retrieve_docs(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Retrieval semantico sui chunk di documentazione indicatori.
        A differenza della versione TF-IDF, risultati multipli possono
        provenire dallo stesso indicatore (chunk diversi: definizione
        breve, estesa, metodologia...).
        """
        return self.indicator_index.search(query, top_n=top_n)

    def retrieve_metadata(
        self,
        query: str,
        top_n: int = 3,
        series_code: Optional[str] = None,
        country_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieval semantico su footnote/country, con filtro opzionale
        per series_code/country_code applicato sui metadata dei chunk.
        Pre-filtrare per codice esatto (quando disponibile) è più
        affidabile che fidarsi solo della similarità semantica, perché
        codici come 'SI.POV.DDAY' non hanno un significato "semantico"
        intrinseco su cui l'embedding possa ragionare bene.
        """
        def _filter(item: Dict[str, Any]) -> bool:
            if series_code and item.get("series_code") != series_code:
                return False
            if country_code and item.get("country_code") != country_code:
                return False
            return True

        use_filter = bool(series_code or country_code)
        return self.metadata_index.search(
            query, top_n=top_n, filter_fn=_filter if use_filter else None
        )

    def get_schema_info(self) -> Dict[str, Any]:
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
        """Lookup esatto/fuzzy per codice o nome — invariato rispetto alla
        versione precedente, complementare (non sostituito) dal retrieval
        semantico: qui serve precisione su un identificatore noto, non
        similarità di significato.
        """
        if query.upper() in self.code_to_name:
            code = query.upper()
            name = self.code_to_name[code]
            return {"code": code, "name": name, "match_type": "exact_code"}

        query_lower = query.lower()
        for name_key, code in self.name_to_code.items():
            if query_lower in name_key:
                return {
                    "code": code,
                    "name": self.code_to_name.get(code),
                    "match_type": "fuzzy_name",
                }

        return {
            "error": f"No indicator found matching '{query}'",
            "suggestion": "Prova retrieve_docs() per la ricerca semantica.",
        }


# =====================================================================
# SANDBOX — invariata rispetto alla versione precedente
# =====================================================================
def run_python_sandbox(code: str, df: pd.DataFrame) -> str:
    safe_globals = {
        "pd": pd,
        "np": np,
        "df": df.copy(),
        "__builtins__": {
            "len": len, "min": min, "max": max, "sum": sum, "round": round,
            "sorted": sorted, "list": list, "dict": dict, "float": float,
            "int": int, "str": str, "abs": abs, "pow": pow, "divmod": divmod,
            "mean": lambda x: float(np.mean(x)),
            "std": lambda x: float(np.std(x)),
            "median": lambda x: float(np.median(x)),
            "var": lambda x: float(np.var(x)),
            "isinstance": isinstance, "type": type, "bool": bool,
            "enumerate": enumerate, "zip": zip, "map": map,
            "filter": filter, "range": range,
        },
    }
    local: Dict[str, Any] = {}
    try:
        exec(code, safe_globals, local)
        if "result" in local:
            return json.dumps(local["result"], default=str, ensure_ascii=False)
        return "Error: executed code did not assign 'result' variable."
    except Exception as e:
        return f"Execution error: {str(e)}"


SYSTEM_PROMPT = """You are a Data Analyst Agent specialized in World Development Indicator (WDI) analysis.

**CRITICAL INSTRUCTIONS:**
- YOU are responsible for using the tools. NEVER suggest to the user "use tool X to...". 
- ALWAYS complete analysis autonomously by calling tools.
- DO NOT provide partial/incomplete answers that delegate work to the user.
- If you mention a tool, you MUST have already called it or are about to call it.

You have four tools, organized in a 2-phase strategy:

**PHASE 1 — Determine and verify indicator(s)**
- retrieve_docs(query): semantic search over indicator documentation chunks (definitions, methodology,
  source). Results may include multiple chunks from the same indicator (short definition, long definition,
  methodology) — read them together to form a complete picture.
- retrieve_metadata(query, series_code=None, country_code=None): semantic search over country metadata and
  data-quality footnotes. Pass series_code/country_code when known to pre-filter results — this is more
  reliable than relying on semantic similarity alone for exact codes.
- inspect_schema(query): exact/fuzzy lookup of indicator code <-> name, or general DataFrame schema when
  called without arguments.

**PHASE 2 — Retrieve and compute data**
- run_python(code): execute pandas/numpy code on the DataFrame `df`. ALWAYS filter by Series Code (e.g.
  'SI.POV.DDAY'), never by Indicator Name. Assign the final answer to a variable named `result`.

**Decision Rules:**
1. Question about definition/methodology/source → call retrieve_docs()
2. Ambiguous concept (e.g. "poverty", "GDP") → call retrieve_docs() to find candidates, then inspect_schema() to verify exact code
3. Question requires specific data values/calculations → call run_python() with verified Series Code
4. Question about data quality/comparability → call retrieve_metadata() 
5. If run_python() result is empty/NaN → adjust filters and retry, never abandon
6. If unsure, use a tool — silence is not acceptable

**Forbidden:**
- Telling the user "you can use the run_python() tool to..."
- Suggesting steps the user should take
- Leaving analysis incomplete
- Guessing data values when tools could retrieve them

Your responsibility is to provide final, actionable answers by using tools autonomously.
"""


class DataAnalystAgent:
    def __init__(self):
        self.store = WDIStore()
        self.llm = OpenRouter(
            api_key=OPENROUTER_API_KEY, max_tokens=1024, context_window=4096, model=OPENROUTER_MODEL
        )

        self.inspect_schema_tool = FunctionTool.from_defaults(
            fn=self._inspect_schema_wrapper,
            name="inspect_schema",
            description="Get DataFrame schema (columns, dtypes, year range) OR lookup indicator by exact "
                        "code/fuzzy name. Input: optional query string (e.g. 'SI.POV.DDAY' or 'poverty').",
        )
        self.retrieve_docs_tool = FunctionTool.from_defaults(
            fn=self._retrieve_docs_wrapper,
            name="retrieve_docs",
            description="Semantic search (sentence embeddings + FAISS) over indicator documentation. "
                        "Input: a natural language query (e.g. 'child mortality rate'). Returns top "
                        "matching documentation chunks with indicator_code, field_type and similarity score.",
        )
        self.retrieve_metadata_tool = FunctionTool.from_defaults(
            fn=self._retrieve_metadata_wrapper,
            name="retrieve_metadata",
            description="Semantic search over country metadata and data-quality footnotes. Input: a query "
                        "string, optionally with 'series_code' and/or 'country_code' to pre-filter results "
                        "(format: 'query | series_code=XX.XXX.XXX | country_code=XXX').",
        )
        self.run_python_tool = FunctionTool.from_defaults(
            fn=self._run_python_wrapper,
            name="run_python",
            description="Execute Python on DataFrame `df`. Use Series Code (not Indicator Name) to filter. "
                        "Assign result to variable `result`.",
        )

        self.agent = ReActAgent(
            name="DataAnalystAgent",
            tools=[
                self.inspect_schema_tool,
                self.retrieve_docs_tool,
                self.retrieve_metadata_tool,
                self.run_python_tool,
            ],
            llm=self.llm,
            verbose=True,
            max_iterations=8,
            system_prompt=SYSTEM_PROMPT,
        )

    def _inspect_schema_wrapper(self, query: str = "") -> str:
        result = self.store.lookup_indicator(query) if query.strip() else self.store.get_schema_info()
        return json.dumps(result, ensure_ascii=False)

    def _retrieve_docs_wrapper(self, query: str) -> str:
        return json.dumps(self.store.retrieve_docs(query, top_n=5), ensure_ascii=False)

    def _retrieve_metadata_wrapper(self, query: str) -> str:
        # Parsing semplice del formato "query | series_code=XX | country_code=XX"
        # per permettere all'LLM di passare filtri opzionali in un singolo input testuale.
        series_code = None
        country_code = None
        parts = [p.strip() for p in query.split("|")]
        base_query = parts[0]
        for part in parts[1:]:
            if part.startswith("series_code="):
                series_code = part.split("=", 1)[1].strip() or None
            elif part.startswith("country_code="):
                country_code = part.split("=", 1)[1].strip() or None

        results = self.store.retrieve_metadata(
            base_query, top_n=3, series_code=series_code, country_code=country_code
        )
        return json.dumps(results, ensure_ascii=False)

    def _run_python_wrapper(self, code: str) -> str:
        return run_python_sandbox(code, self.store.df)

    async def aquery(self, question: str) -> str:
        try:
            response = await self.agent.run(
                user_msg=question, max_iterations=8, early_stopping_method="generate"
            )
            answer = str(response)
            # Post-process: remove suggestions to use tools
            answer = self._remove_tool_suggestions(answer)
            return answer
        except Exception as e:
            msg = str(e)
            if "Max iterations" in msg or "max_iterations" in msg:
                response = await self.agent.run(user_msg=question, early_stopping_method="generate")
                answer = str(response)
                answer = self._remove_tool_suggestions(answer)
                return answer
            raise

    def _remove_tool_suggestions(self, text: str) -> str:
        """Remove suggestions like 'you can use the run_python() tool' or 'please use...'"""
        lines = text.split('\n')
        filtered = []
        for line in lines:
            # Remove lines that are pure suggestions to use tools
            lower = line.lower()
            if any(pattern in lower for pattern in [
                'you can use',
                'please use',
                'use the run_python()',
                'use the retrieve_',
                'use the inspect_',
                'for the exact figure',
                'for exact data',
                'please run',
                'you can run',
            ]):
                # Check if it's a pure suggestion (not a description of what we found)
                if not any(word in lower for word in ['found', 'retrieved', 'got', 'shows', 'indicates']):
                    continue
            filtered.append(line)
        return '\n'.join(filtered).strip()

    def query(self, question: str, timeout: int = 30) -> str:
        return asyncio.run(self.aquery(question))

    def answer(self, question: str, timeout: int = 30) -> str:
        return self.query(question, timeout=timeout)

    def format_answer(self, answer: Any) -> str:
        if isinstance(answer, str):
            return answer
        try:
            return str(answer)
        except Exception:
            return json.dumps(answer, ensure_ascii=False)


def interactive():
    agent = DataAnalystAgent()
    print("Data Analyst Agent — RAG semantico (sentence-transformers + FAISS). Type 'exit' to quit.")
    while True:
        q = input("Question> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("Thinking... (this may take a few seconds)")
        try:
            print(agent.query(q))
        except Exception as e:
            print(f"Agent error: {e}")


if __name__ == "__main__":
    interactive()