# Design del Data Analyst Agent

## Versioni disponibili

Il repo contiene due implementazioni complementari:

1. **`agent.py`** — Prototipo leggero, standalone
   - Dataset offline di esempio (5 indicatori, 4 paesi, 2 anni)
   - Retrieval TF-IDF su documentazione
   - Generazione di codice euristico o OpenAI
   - Eseguibile da zero senza dipendenze esterne

2. **`agent_openrouter.py`** — Agente ReAct con World Bank API live
   - ReActAgent da llama_index
   - World Bank API per dati in tempo reale (~1600 indicatori)
   - Gradio UI per chat interattiva
   - Richiede OpenRouter API key

---

## Architettura (versione semplice — `agent.py`)

Il progetto è organizzato in tre componenti principali:

### 1. `WDIStore`
- Carica il dataset tabellare da `sample_wdi.csv`
- Carica la documentazione da `indicators.json`
- Costruisce un indice TF-IDF per il retrieval documentale
- Fornisce metodi per ricerca indicatori e estrazione dati

### 2. `QueryPlanner`
- Classifica la domanda in quattro categorie: `data`, `docs`, `both`, `unknown`
- Usa regole lessicali (parole chiave come "trend", "definition", "compare", etc.)
- La classificazione guida il workflow dell'agente

### 3. `CodeGenerator` + `Sandbox`
- Genera codice Python per rispondere a query sui dati
- Se `OPENAI_API_KEY` è disponibile, chiede a GPT-3.5-turbo
- Altrimenti, usa euristiche (pattern matching su nomi indicatori, anni, paesi)
- Esegue il codice in una sandbox Python minimale
- Cattura eccezioni e ritorna messaggi di errore espliciti

### Flusso

```
Query → QueryPlanner.classify()
  ↓
  ├─ "docs" → WDIStore.retrieve_docs() → Risultati documentali
  ├─ "data" → CodeGenerator.synthesize() → Sandbox.run() → Risultati numerici
  ├─ "both" → Recupera sia documenti che dati
  └─ "unknown" → Messaggio di fallback
```

---

## Architettura (versione OpenRouter — `agent_openrouter.py`)

### 1. ReActAgent (llama_index)
- Orchestrazione centralizzata con il modello owl-alpha (OpenRouter)
- Tool calling automatico basato sul reasoning dell'agente
- Verbose mode per visualizzare il pensiero dell'agente

### 2. Tools (function tools da llama_index)
- `search_indicators` — cercare indicatori WDI per parola chiave
- `search_countries` — cercare paesi per nome/regione
- `get_indicator_data` — recuperare dati per un indicatore e paese (con anno opzionale)
- `compare_countries` — confronto cross-country di un indicatore
- `get_country_info` — informazioni di base su un paese (regione, income level, capitale)

### 3. World Bank API
- Endpoint base: `https://api.worldbank.org/v2`
- Cache locale in `data/wb_indicators.json` e `data/wb_countries.json`
- Fallback a cache se API non disponibile (offline resilience)
- Limiti rate: gestito dalla cache e da logica di requestin efficiente

### 4. Gradio UI
- Chat interface per dialogo naturale
- Supporta multi-turn conversations
- Visualizza il reasoning in verbose mode
- Pulizia delle risposte (rimozione di tag di thinking)

### Flusso

```
Query (Gradio) → ReActAgent.aquery()
  ↓
  ReActAgent (owl-alpha)
    ├─ Reasoning: "Che tool mi serve?"
    ├─ Tool call 1: search_indicators("GDP")
    ├─ Tool call 2: get_indicator_data(...)
    ├─ Reasoning: "Ora ho i dati, formulo risposta"
    └─ Risposta testuale
  ↓
clean_response() → Gradio output
```

---

## Scelte principali e motivazioni

### 1. Uso di WDI

Ho scelto il **World Development Indicators (WDI)** della World Bank perché:
- Struttura tabellare chiara (paese, anno, indicatore, valore)
- ~1600 indicatori socio-economici (GDP, popolazione, CO2, disoccupazione, povertà, etc.)
- Documentazione tecnica ricca di caveat e note metodologiche
- API pubblica senza autenticazione
- Dati storici 50+ anni

**Prototipo semplice**: subset rappresentativo per garantire eseguibilità offline.

**Versione OpenRouter**: accesso live alla API ufficiale, dati real-time.

### 2. QueryPlanner vs ReActAgent

**Prototipo semplice**: routing manuale basato su regole lessicali.
- Vantaggi: veloce, prevedibile, non richiede LLM esterno
- Svantaggi: fragile su formulazioni diverse

**Versione OpenRouter**: orchestrazione automatica con LLM.
- Vantaggi: flessibile, si adatta a nuovi pattern di query
- Svantaggi: latenza maggiore, richiede API key

### 3. Retrieval documentale

**Prototipo semplice** usa TF-IDF (scikit-learn):
- Veloce e leggero
- Funziona bene con testi brevi

**Versione OpenRouter** usa tool calling:
- ReActAgent decide quando cercare
- Più intelligente su query complesse

### 4. Sandbox Python vs Tool Calling

**Prototipo semplice**:
- Genera codice Python e lo esegue
- Controllo fine sui calcoli

**Versione OpenRouter**:
- Delega a World Bank API
- Più scalabile

---

## Debolezze conosciute

### Versione semplice (`agent.py`)

1. **Corpus limitato** — Solo 5 indicatori e 4 paesi nel sample
2. **Classificazione fragile** — Regole lessicali non generalizzano bene
3. **Codice generato errato** — Nessuna validazione pre-execution
4. **RAG semplice** — TF-IDF ha basso recall su query semanticamente diverse
5. **Range temporale ristretto** — Solo 2010 e 2020, difficile calcolare trend

### Versione OpenRouter (`agent_openrouter.py`)

1. **Latenza API** — N round trip a World Bank API + LLM calls
2. **Rate limiting** — World Bank API ha limiti di concorrenza
3. **Tool calling errors** — owl-alpha potrebbe generare parametri errati
4. **Context window ridotto** — 4096 token, query lunghe potrebbero fallire
5. **No state management** — Ogni conversazione è indipendente
6. **No fact checking** — Accetta dati da API senza validazione

---

## Cosa farei con più tempo

### Versione semplice
- Download full WDI dataset (~1600 indicatori)
- Miglior RAG con embedding models (sentence-transformers)
- Query planning con LLM per classificare meglio
- Unit test e benchmark

### Versione OpenRouter
- Switch a modello più capace (GPT-4, Claude 3)
- State management e persistent sessions
- Retry logic con exponential backoff
- Batch queries e aggregazione
- Fact checking e validation
- UI avanzata con visualizzazioni (grafici, mappe)

---

## Esempi forniti

### `examples/run_examples.py` (prototipo semplice)
Mostra 6 query diverse coprendo data/docs/mixed/unsupported cases

### Versione OpenRouter (live chat)
Avvia Gradio UI con chat interface e tool calling trasparente

---

## Tabella comparativa

| Aspetto | `agent.py` | `agent_openrouter.py` |
|---------|-----------|----------------------|
| **Setup** | Immediato | Richiede OpenRouter API |
| **Latency** | ms | s (API calls) |
| **Data** | Sample offline | Full WDI live |
| **Flexibility** | Regole fisse | LLM reasoning |
| **Produzione** | Prototipo | Semi-production |
| **UI** | CLI | Gradio |
| **Costo** | Gratis | $ (OpenRouter token) |
