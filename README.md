# Data Analyst Agent

Progetto take-home: agente AI per rispondere a domande in linguaggio naturale su dataset tabellari e documentazione tecnica.

## Obiettivo

Questo repo dimostra un agente che combina:

- esecuzione di codice su dati tabellari in sandbox
- retrieval da un corpus di documentazione tecnica su indicatori WDI
- decisione automatica su quando usare dati, documentazione o entrambi

## Come usare

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

## OpenAI (opzionale)

Se vuoi usare un modello OpenAI per la comprensione delle query e la generazione di codice, esporta `OPENAI_API_KEY`:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Se non è disponibile, l'agente userà una modalità fallback basata su euristiche.

## Dati

Il repository include un campione rappresentativo di WDI in `data/sample_wdi.csv` e documentazione indicatori in `data/indicators.json`.

Con più tempo, il codice può essere esteso per scaricare *tutti* i dati WDI dal World Bank API.
