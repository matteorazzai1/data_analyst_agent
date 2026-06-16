# Data Analyst Agent

Progetto take-home: agente AI per rispondere a domande in linguaggio naturale su dataset tabellari e documentazione tecnica.

## Obiettivo

Questo repo dimostra due implementazioni di un agente che combina:

- esecuzione di codice su dati tabellari in sandbox (prototipo semplice)
- retrieval da un corpus di documentazione tecnica su indicatori WDI (prototipo semplice)
- **ReActAgent con World Bank API live** (OpenRouter, production-ready)

## Come usare

### Versione semplice (prototipo offline)

1. Creare un ambiente Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Eseguire l'agente con esempi:

```powershell
python examples\run_examples.py
```

3. Interazione manuale:

```powershell
python agent.py
```

### Versione OpenRouter (full WDI API, Gradio UI)

1. Setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configurare la API key:

Crea un file `.env`:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Oppure esporta direttamente:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

3. Avviare l'agente con Gradio:

```powershell
python agent_openrouter.py
```

Apri browser a `http://localhost:7860`

## Dataset

Il repository include:

- **Versione semplice**: campione rappresentativo in `data/sample_wdi.csv` (offline, eseguibile da zero)
- **Versione OpenRouter**: accesso live a World Bank API (`https://api.worldbank.org/v2`)
  - Cache locale di indicatori e paesi per performance
  - Dati in real-time per query specifiche

## OpenAI vs OpenRouter

- **Versione semplice** (`agent.py`): supporta OpenAI se `OPENAI_API_KEY` è disponibile, altrimenti usa euristiche
- **Versione OpenRouter** (`agent_openrouter.py`): richiede `OPENROUTER_API_KEY`, usa owl-alpha model e ReActAgent da llama_index
