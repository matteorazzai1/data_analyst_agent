import json
from typing import Any, Dict
import pandas as pd
import numpy as np


def run_python_sandbox(code: str, df: pd.DataFrame) -> str:
    """Execute Python code in a restricted sandbox with limited builtins."""
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
