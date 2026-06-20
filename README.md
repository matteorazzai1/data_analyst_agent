# DataAnalystAgent — World Development Indicator Chatbot

A conversational AI agent that answers questions about **World Development Indicators (WDI)** by combining semantic retrieval (RAG) on indicator documentation with autonomous code execution on tabular data.

## 🎯 Features

- **Semantic Retrieval (RAG)**: Uses sentence-transformers + FAISS to search indicator definitions, methodologies, and sources
- **Code Execution**: Executes pandas/numpy code in a restricted sandbox to analyze WDI data
- **Autonomous Reasoning**: LlamaIndex ReActAgent with explicit 2-phase strategy (Retrieve → Verify → Execute)
- **Web Interface**: Gradio-based chatbot with example questions and conversation history
- **CLI Mode**: Command-line interface for batch processing and testing

## 📋 Requirements

- Python 3.9+
- API key for OpenRouter (or compatible LLM service)
- WDI dataset CSV files (provided in `data/WDI_CSV_2026_04_09/`)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone <repo-url>
cd data_analyst_agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:
```env
# Required: OpenRouter API key
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Model selection (default: openrouter/owl-alpha)
OPENROUTER_MODEL=openrouter/owl-alpha

# Optional: Embedding model (default: all-MiniLM-L6-v2)
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

## 📁 Project Structure

```
data_analyst_agent/
│
├── core/                          # Core data processing
│   ├── __init__.py
│   ├── semantic_index.py          # FAISS vector index + sentence-transformers
│   ├── store.py                   # WDIStore: loads & prepares WDI data
│   └── sandbox.py                 # Python code execution sandbox
│
├── tools/                         # ReActAgent tools (modular)
│   ├── __init__.py
│   ├── inspect_schema_tool.py     # Schema lookup & indicator fuzzy/exact match
│   ├── retrieve_docs_tool.py      # Semantic search on indicator documentation
│   ├── retrieve_metadata_tool.py  # Semantic search on country/footnote metadata
│   └── run_python_tool.py         # Code execution in restricted environment
│
├── examples/
│   └── run_examples.py            # CLI example runner (testing & batch mode)
│
├── agent.py                       # DataAnalystAgent + ReActAgent orchestration
├── main.py                        # Gradio web interface
├── doc.py                         # Generate DOCX documentation
│
├── data/
│   └── WDI_CSV_2026_04_09/        # World Development Indicators dataset
│       ├── WDICSV.csv             # Main data (countries × indicators × years)
│       ├── WDISeries.csv          # Indicator metadata (definitions, methodology)
│       ├── WDICountry.csv         # Country/region metadata
│       ├── WDIfootnote.csv        # Data quality notes
│       └── ...
│
├── .env                           # Environment variables (not in version control)
├── .gitignore
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

### Core Modules Explained

#### `core/semantic_index.py`
- **SemanticIndex**: Generic wrapper around FAISS (IndexFlatIP) + sentence-transformers
- Uses L2-normalized embeddings for cosine similarity
- Supports optional filter functions for pre/post-filtering results

#### `core/store.py`
- **WDIStore**: Loads all WDI CSV files and prepares them for querying
- Transforms main data from wide (columns = years) → long format (one row per value)
- Builds two separate FAISS indices:
  - `indicator_index`: Indicator definitions, methodologies, sources (chunked per field)
  - `metadata_index`: Country metadata and data quality footnotes
- Provides APIs: `retrieve_docs()`, `retrieve_metadata()`, `lookup_indicator()`, `get_schema_info()`

#### `core/sandbox.py`
- **run_python_sandbox()**: Executes user-provided code in a restricted environment
- Whitelist of safe builtins (no `import`, `eval`, file I/O)
- Available: pandas, numpy, basic math, iteration functions

#### `tools/`
Each tool file exports a factory function `create_*_tool()` that returns a `FunctionTool`:
- **inspect_schema_tool**: Lookup indicator by code/name or get DataFrame schema
- **retrieve_docs_tool**: Semantic search on indicator documentation
- **retrieve_metadata_tool**: Semantic search on country/footnote data with optional filters
- **run_python_tool**: Execute code on the DataFrame

#### `agent.py`
- **DataAnalystAgent**: Main orchestrator class
  - Initializes WDIStore, LLM (OpenRouter), and 4 tools
  - Creates ReActAgent with explicit system prompt
  - Implements 2-phase strategy: Retrieve/Verify → Execute
  - Post-processes responses to remove tool suggestions
- **SYSTEM_PROMPT**: Declares rules and decision logic for autonomous execution
- **EXAMPLES**: List of example questions for Gradio/CLI

## 💬 Usage

### Option 1: Web Interface (Gradio)

```bash
python main.py
```

Then open `http://127.0.0.1:7860` in your browser.

**Features:**
- Chat interface with conversation history
- 6 example questions in clickable buttons (auto-fill input)
- "About" section with architecture details
- Clear button to reset conversation
- Support for Enter key to submit

### Option 2: Command-Line Interface (CLI)

```bash
python examples/run_examples.py
```

Edit `EXAMPLES` in `examples/run_examples.py` to test different questions.

### Option 3: Interactive Mode

```bash
python agent.py
```

Then type questions at the prompt. Type `exit` or `quit` to stop.

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | (required) | Your OpenRouter API key |
| `OPENROUTER_MODEL` | `openrouter/owl-alpha` | LLM model to use |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model for semantic search |

### Tuning Parameters (in `agent.py`)

- **`max_iterations`**: ReActAgent iteration limit (default: 8)
  - Increase if questions require more tool calls
  - Decrease to reduce latency
  
- **Tool top_n values**:
  - `retrieve_docs()`: Returns top 5 chunks (in `tools/retrieve_docs_tool.py`)
  - `retrieve_metadata()`: Returns top 3 chunks (in `tools/retrieve_metadata_tool.py`)

## 📊 Example Questions

Try asking:

- **Definition**: "What does the indicator SI.POV.DDAY measure?"
- **Data**: "What was the GDP of India in 2020?"
- **Comparison**: "Compare CO2 emissions per capita between the United States and China in 2020."
- **Quality**: "Is GDP data comparable across countries in WDI?"
- **Trend**: "What was the trend in unemployment rate for Brazil between 2010 and 2020?"
- **Complex**: "Show me the share of renewable energy in GDP."

## 🏗️ Architecture

### 2-Phase Strategy

The agent follows an explicit 2-phase reasoning strategy (declared in `SYSTEM_PROMPT`):

**Phase 1: Determine and Verify Indicator**
- User asks a conceptual question → call `retrieve_docs()` for semantically similar definitions
- Ambiguous term (e.g., "poverty") → use `retrieve_docs()` to find candidates, then `inspect_schema()` to verify exact code
- Question about data quality → call `retrieve_metadata()`
- Indicator code already known → call `inspect_schema()` for quick lookup

**Phase 2: Retrieve and Compute**
- Question requires specific values → call `run_python()` with the verified Series Code
- Filter by country/year if relevant
- If result is empty/NaN → adjust filters and retry

### Semantic Indexing Strategy

Two separate FAISS indices avoid noise:

| Index | Source | Chunk Strategy | Why |
|-------|--------|-----------------|-----|
| `indicator_index` | WDISeries.csv | One chunk per field (short def, long def, methodology, relevance) | Prevents long definitions from dominating similarity scores |
| `metadata_index` | WDICountry.csv + WDIfootnote.csv | One chunk per country + one per footnote, enriched with context | Short, focused on data availability and quality |

### Tool Selection Strategy

The system prompt provides explicit decision rules to prevent the agent from choosing tools randomly:

```
1. Definition/methodology → retrieve_docs()
2. Ambiguous concept → retrieve_docs() then inspect_schema()
3. Data values/calculations → run_python() with verified Series Code
4. Data quality/comparability → retrieve_metadata()
5. If unsure → use a tool (silence is not acceptable)
```

## 📚 Documentation

Generate a comprehensive DOCX documentation file:

```bash
pip install python-docx
python doc.py
```

This creates `DataAnalystAgent_Documentazione.docx` with:
- Full architecture diagrams
- Design decisions and trade-offs
- Chunking strategies
- Future improvements
- Spec-Driven Development notes

## 🔒 Sandbox Security

The Python execution sandbox restricts access to:
- **No imports**: No `import` statement allowed
- **No eval/exec**: No `eval()`, `exec()`, or dynamic code generation
- **No file I/O**: No `open()` or file system operations
- **Whitelist-based**: Only safe builtins available

**Available functions:**
- Math: `len`, `min`, `max`, `sum`, `abs`, `pow`, `round`, `divmod`
- Stats: `mean`, `std`, `median`, `var` (via numpy)
- Collections: `list`, `dict`, `sorted`, `enumerate`, `zip`, `map`, `filter`
- Type checks: `isinstance`, `type`, `bool`
- Modules: `pandas` (as `pd`), `numpy` (as `np`), `DataFrame` (as `df`)

## 🚦 Troubleshooting

### Issue: "Missing OpenRouter API key"
**Solution**: Make sure `.env` file exists in the project root with `OPENROUTER_API_KEY=...`

### Issue: "Required dataset file not found"
**Solution**: Check that WDI CSV files exist in `data/WDI_CSV_2026_04_09/`. Download if needed from World Bank.

### Issue: Slow initialization
**Cause**: First run loads and embeds ~5,000 chunks. This is normal.
**Solution**: Subsequent runs reuse the indices (they stay in memory during the session).

### Issue: Agent suggests "use the tool X"
**Expected**: The system prompt forbids this. If it happens, the post-processing filter removes these lines.

### Issue: Empty or NaN results from run_python()
**Solution**: The agent implements retry logic to adjust filters (country/year) before giving up.

## 📖 Further Reading

- **LlamaIndex**: https://docs.llamaindex.ai/
- **FAISS**: https://github.com/facebookresearch/faiss
- **sentence-transformers**: https://www.sbert.net/
- **OpenRouter**: https://openrouter.ai/
- **World Development Indicators**: https://datatopics.worldbank.org/world-development-indicators/

## 📄 License

[Add your license here]

## 👤 Author

[Your name/organization]

## 🙏 Acknowledgments

- Dataset: World Bank World Development Indicators
- Architecture: LlamaIndex, FAISS, sentence-transformers
- Interface: Gradio

---

**Last Updated**: June 2026  
**Version**: 1.0 (Modular + Gradio Interface)
