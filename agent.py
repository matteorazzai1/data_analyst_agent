import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_CSV = DATA_DIR / "sample_wdi.csv"
INDICATORS_JSON = DATA_DIR / "indicators.json"

DEFAULT_INDICATOR_CODES = [
    "NY.GDP.MKTP.CD",
    "SP.POP.TOTL",
    "EN.ATM.CO2E.PC",
    "SL.UEM.TOTL.ZS",
    "SI.POV.DDAY",
]


class WDIStore:
    def __init__(self):
        self.indicators = self._load_indicators()
        self.df = self._load_data()
        self.vectorizer, self.tfidf = self._build_doc_index()

    def _load_indicators(self) -> List[Dict[str, Any]]:
        with open(INDICATORS_JSON, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(SAMPLE_CSV)
        df["year"] = df["year"].astype(int)
        return df

    def _build_doc_index(self):
        texts = []
        for indicator in self.indicators:
            text = " ".join(
                [indicator["name"], indicator["source_note"], indicator.get("topics", "")]
            )
            texts.append(text)

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform(texts)
        return vectorizer, tfidf

    def retrieve_docs(self, query: str, top_n: int = 3) -> List[Dict[str, Any]]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf).flatten()
        ranked = scores.argsort()[::-1][:top_n]
        results = []
        for i in ranked:
            indicator = self.indicators[i].copy()
            indicator["score"] = float(scores[i])
            results.append(indicator)
        return results

    def get_indicator(self, code: str) -> Optional[Dict[str, Any]]:
        for indicator in self.indicators:
            if indicator["indicator_code"] == code:
                return indicator
        return None

    def print_data_summary(self) -> None:
        print("Dataset summary")
        print(self.df["indicator_code"].value_counts())


class QueryPlanner:
    DATA_TERMS = [
        "what", "compare", "trend", "highest", "lowest", "average", "mean", "total", "difference",
        "growth", "increase", "decrease", "year", "value", "under", "above", "between", "change",
        "versus", "vs", "rank", "top", "bottom",
    ]
    DOCS_TERMS = [
        "definition", "meaning", "indicator", "methodology", "source", "caveat", "note", "notes",
        "comparability", "comparable", "revision", "quality", "coverage", "metadata", "interpretation",
        "why", "explain", "explanation", "use", "definition",
    ]

    def classify(self, query: str) -> str:
        text = query.lower()
        data_score = sum(term in text for term in self.DATA_TERMS)
        docs_score = sum(term in text for term in self.DOCS_TERMS)

        if docs_score and not data_score:
            return "docs"
        if data_score and not docs_score:
            return "data"
        if docs_score and data_score:
            return "both"
        return "both"


class CodeGenerator:
    def __init__(self, store: WDIStore):
        self.store = store

    def _openai_generate(self, question: str) -> Optional[str]:
        try:
            import openai

            openai.api_key = OPENAI_API_KEY
            system = (
                "You are a Python developer who answers data questions by generating a small Python snippet. "
                "The only allowed variables are df (a pandas DataFrame), pd, and np. "
                "Return only valid Python code that assigns the final answer to a variable named result. "
                "Do not import modules or print anything."
            )
            prompt = (
                f"DataFrame df has columns: country, country_code, year, indicator_code, indicator_name, value. "
                f"Write Python code to answer the question: {question}\n"
                "The code should compute a meaningful value and assign it to result. "
                "If the question is too vague for the available sample data, assign a descriptive string to result."
            )
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0,
            )
            text = response.choices[0].message.content
            return text
        except Exception:
            return None

    def _heuristic_code(self, question: str) -> str:
        q = question.lower()
        if "renewable" in q or "renewables" in q:
            return "result = 'La domanda richiede un indicatore non disponibile nel campione WDI fornito.'\n"

        if "gdp" in q or "ny.gdp.mktp.cd" in q or "gross domestic product" in q:
            indicator_code = "NY.GDP.MKTP.CD"
        elif "population" in q or "sp.pop.totl" in q:
            indicator_code = "SP.POP.TOTL"
        elif "co2" in q or "emissions" in q or "en.atm.co2e.pc" in q:
            indicator_code = "EN.ATM.CO2E.PC"
        elif "unemploy" in q or "sl.uem.totl.zs" in q:
            indicator_code = "SL.UEM.TOTL.ZS"
        elif "poverty" in q or "si.pov.dday" in q:
            indicator_code = "SI.POV.DDAY"
        else:
            indicator_code = None

        countries = [c.lower() for c in self.store.df["country"].unique()]
        selected_countries = [c for c in countries if c in q]
        country_filter = ""
        if selected_countries:
            quoted = ", ".join(repr(c.title()) for c in selected_countries)
            country_filter = f" & df['country'].isin([{quoted}])"

        if "compare" in q or "vs" in q or "versus" in q or "between" in q:
            code = (
                "indicator_code = '{}'\n"
                "subset = df[(df['indicator_code'] == indicator_code ){}]\n"
                "result = subset.sort_values(['year', 'country']).head(20).to_dict(orient='records')\n"
            ).format(indicator_code, country_filter)
            return code

        match = re.search(r"(\d{4})", question)
        if match and indicator_code:
            year = match.group(1)
            code = (
                "indicator_code = '{}'\n"
                "subset = df[(df['indicator_code'] == indicator_code ) & (df['year'] == {}){}]\n"
                "result = subset[['country', 'year', 'value']].to_dict(orient='records')\n"
            ).format(indicator_code, year, country_filter)
            return code

        if indicator_code:
            code = (
                "indicator_code = '{}'\n"
                "subset = df[(df['indicator_code'] == indicator_code ){}]\n"
                "result = subset.groupby('country')['value'].mean().round(3).to_dict()\n"
            ).format(indicator_code, country_filter)
            return code

        return "result = 'Non riesco a costruire una query chiara con i dati disponibili.'\n"

    def synthesize(self, question: str) -> str:
        if OPENAI_API_KEY:
            generated = self._openai_generate(question)
            if generated:
                return generated
        return self._heuristic_code(question)


class Sandbox:
    @staticmethod
    def run(code: str, df: pd.DataFrame) -> Any:
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
        exec(code, safe_globals, local)
        if "result" not in local:
            raise ValueError("Il codice eseguito non ha assegnato 'result'.")
        return local["result"]


class DataAnalystAgent:
    def __init__(self):
        self.store = WDIStore()
        self.planner = QueryPlanner()
        self.generator = CodeGenerator(self.store)

    def answer(self, question: str) -> Dict[str, Any]:
        plan = self.planner.classify(question)
        docs = []
        data_result = None
        code = None
        errors = []

        if plan in ("docs", "both"):
            docs = self.store.retrieve_docs(question, top_n=3)

        if plan in ("data", "both"):
            code = self.generator.synthesize(question)
            try:
                data_result = Sandbox.run(code, self.store.df)
            except Exception as exc:
                errors.append(str(exc))
                data_result = None

        return {
            "question": question,
            "plan": plan,
            "docs": docs,
            "data_result": data_result,
            "code": code,
            "errors": errors,
        }

    def format_answer(self, result: Dict[str, Any]) -> str:
        lines = [f"Question: {result['question']}", f"Plan: {result['plan']}\n"]
        if result["docs"]:
            lines.append("Document retrieval:")
            for doc in result["docs"]:
                lines.append(f"- {doc['indicator_code']}: {doc['name']}")
                lines.append(f"  Note: {doc['source_note']}")
            lines.append("")
        if result["data_result"] is not None:
            lines.append("Data result:")
            lines.append(str(result["data_result"]))
            lines.append("")
        if result["errors"]:
            lines.append("Errors:")
            lines.extend(result["errors"])
            lines.append("")
        if result["code"]:
            lines.append("Executed code:")
            lines.append(result["code"].strip())
        return "\n".join(lines)


def interactive_loop():
    agent = DataAnalystAgent()
    print("Data Analyst Agent — WDI sample. Digita 'exit' per uscire.")
    while True:
        question = input("Domanda> ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        answer = agent.answer(question)
        print(agent.format_answer(answer))
        print("---")


def main():
    interactive_loop()


if __name__ == "__main__":
    main()
