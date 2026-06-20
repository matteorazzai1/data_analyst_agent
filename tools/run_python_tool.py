import pandas as pd
from llama_index.core.tools import FunctionTool
from core.sandbox import run_python_sandbox


def create_run_python_tool(df: pd.DataFrame) -> FunctionTool:
    """Create the run_python tool."""
    
    def run_python(code: str) -> str:
        return run_python_sandbox(code, df)
    
    return FunctionTool.from_defaults(
        fn=run_python,
        name="run_python",
        description="Execute Python on DataFrame `df`. Use Series Code (not Indicator Name) to filter. "
                    "Assign result to variable `result`.",
    )
