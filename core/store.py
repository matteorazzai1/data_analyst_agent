import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer

from .semantic_index import SemanticIndex

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WDI_FOLDER = DATA_DIR / "WDI_CSV_2026_04_09"
WDI_MAIN_CSV = WDI_FOLDER / "WDICSV.csv"
WDI_SERIES_CSV = WDI_FOLDER / "WDISeries.csv"
WDI_COUNTRY_CSV = WDI_FOLDER / "WDICountry.csv"
WDI_FOOTNOTE_CSV = WDI_FOLDER / "WDIfootnote.csv"
WDI_COUNTRY_SERIES_CSV = WDI_FOLDER / "WDIcountry-series.csv"
WDI_SERIES_TIME_CSV = WDI_FOLDER / "WDIseries-time.csv"

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


class WDIStore:
    FIELD_LABELS = {
        "short_definition": "Definizione breve",
        "long_definition": "Definizione estesa",
        "methodology": "Metodologia statistica",
        "development_relevance": "Rilevanza per lo sviluppo",
    }

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
                    continue

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

    def retrieve_docs(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        return self.indicator_index.search(query, top_n=top_n)

    def retrieve_metadata(
        self,
        query: str,
        top_n: int = 3,
        series_code: Optional[str] = None,
        country_code: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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
