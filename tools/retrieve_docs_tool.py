import json
from llama_index.core.tools import FunctionTool
from core.store import WDIStore


def create_retrieve_docs_tool(store: WDIStore) -> FunctionTool:
    """Create the retrieve_docs tool."""
    
    def retrieve_docs(query: str) -> str:
        return json.dumps(store.retrieve_docs(query, top_n=5), ensure_ascii=False)
    
    return FunctionTool.from_defaults(
        fn=retrieve_docs,
        name="retrieve_docs",
        description="Semantic search (sentence embeddings + FAISS) over indicator documentation. "
                    "Input: a natural language query (e.g. 'child mortality rate'). Returns top "
                    "matching documentation chunks with indicator_code, field_type and similarity score.",
    )
