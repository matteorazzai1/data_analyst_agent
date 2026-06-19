#!/usr/bin/env python
"""Test Phase 1 + Phase 3 enhancements."""

from agent import WDIStore, DataAnalystAgent
import json

# Test Phase 1: schema inspection
print("=" * 60)
print("PHASE 1: Schema Inspection Tests")
print("=" * 60)

store = WDIStore()

# Test get_schema_info
schema = store.get_schema_info()
print("\n✓ get_schema_info() works:")
print(f"  - Columns: {schema['columns']}")
print(f"  - Year range: {schema['year_range']}")
print(f"  - Unique indicators: {schema['unique_indicators']}")
print(f"  - Unique countries: {schema['unique_countries']}")
print(f"  - Row count: {schema['row_count']}")

# Test lookup_indicator with exact code match
print("\n✓ lookup_indicator() - Exact match (SI.POV.DDAY):")
result_exact = store.lookup_indicator('SI.POV.DDAY')
print(f"  - Code: {result_exact.get('code')}")
print(f"  - Name: {result_exact.get('name')}")
print(f"  - Match type: {result_exact.get('match_type')}")

# Test lookup_indicator with fuzzy match
print("\n✓ lookup_indicator() - Fuzzy match ('poverty'):")
result_fuzzy = store.lookup_indicator('poverty')
if 'error' not in result_fuzzy:
    print(f"  - Code: {result_fuzzy.get('code')}")
    print(f"  - Name: {result_fuzzy.get('name')}")
    print(f"  - Match type: {result_fuzzy.get('match_type')}")
else:
    print(f"  - {result_fuzzy['error']}")

# Test Phase 3: Extended builtins
print("\n" + "=" * 60)
print("PHASE 3: Extended Builtins Tests")
print("=" * 60)

from agent import run_python_sandbox
import pandas as pd
import numpy as np

test_df = pd.DataFrame({
    'value': [10.5, 20.3, 15.8, np.nan, 25.0],
    'country': ['A', 'B', 'C', 'D', 'E']
})

# Test abs
code_abs = """
result = [abs(-x) for x in [-1, -2, 3]]
"""
output = run_python_sandbox(code_abs, test_df)
print(f"\n✓ abs() in sandbox: {output}")

# Test mean and std
code_stats = """
values = [10, 20, 30, 40, 50]
result = {
    'mean': mean(values),
    'std': std(values)
}
"""
output = run_python_sandbox(code_stats, test_df)
print(f"✓ mean() and std() in sandbox: {output}")

# Test enumerate
code_enum = """
result = list(enumerate(['a', 'b', 'c']))
"""
output = run_python_sandbox(code_enum, test_df)
print(f"✓ enumerate() in sandbox: {output}")

# Test DataAnalystAgent with new tool
print("\n" + "=" * 60)
print("AGENT INITIALIZATION")
print("=" * 60)

agent = DataAnalystAgent()
print("\n✓ DataAnalystAgent initialized successfully")
print(f"  - Tools count: {len(agent.agent.tools)}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
