import json
from llama_index.core.tools import FunctionTool
from core.store import WDIStore


def create_retrieve_metadata_tool(store: WDIStore) -> FunctionTool:
    """Create the retrieve_metadata tool."""
    
    def retrieve_metadata(query: str) -> str:
        # Parsing semplice del formato "query | series_code=XX | country_code=XX"
        series_code = None
        country_code = None
        parts = [p.strip() for p in query.split("|")]
        base_query = parts[0]
        for part in parts[1:]:
            if part.startswith("series_code="):
                series_code = part.split("=", 1)[1].strip() or None
            elif part.startswith("country_code="):
                country_code = part.split("=", 1)[1].strip() or None

        results = store.retrieve_metadata(
            base_query, top_n=3, series_code=series_code, country_code=country_code
        )
        return json.dumps(results, ensure_ascii=False)
    
    return FunctionTool.from_defaults(
        fn=retrieve_metadata,
        name="retrieve_metadata",
        description="Semantic search over country metadata and data-quality footnotes. Input: a query "
                    "string, optionally with 'series_code' and/or 'country_code' to pre-filter results "
                    "(format: 'query | series_code=XX.XXX.XXX | country_code=XXX').",
    )
