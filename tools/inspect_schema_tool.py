import json
from llama_index.core.tools import FunctionTool
from core.store import WDIStore


def create_inspect_schema_tool(store: WDIStore) -> FunctionTool:
    """Create the inspect_schema tool."""
    
    def inspect_schema(query: str = "") -> str:
        result = store.lookup_indicator(query) if query.strip() else store.get_schema_info()
        return json.dumps(result, ensure_ascii=False)
    
    return FunctionTool.from_defaults(
        fn=inspect_schema,
        name="inspect_schema",
        description="Get DataFrame schema (columns, dtypes, year range) OR lookup indicator by exact "
                    "code/fuzzy name. Input: optional query string (e.g. 'SI.POV.DDAY' or 'poverty').",
    )
