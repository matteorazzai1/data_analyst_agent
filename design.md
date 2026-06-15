# Design del Data Analyst Agent

## Obiettivo del progetto

L'agente deve rispondere a domande su dati tabellari e su documentazione tecnica, scegliendo autonomamente se usare i dati, i documenti o entrambi.

## Architettura

Il progetto è organizzato in tre componenti principali:

1. `WDIStore`
   - contiene il dataset tabellare (campione WDI) e l'indice di documento.
   - fornisce caricamento dati e retrieval su documentazione indicatori.

2. `QueryPlanner`
   - classifica la domanda in quattro casi: `data`, `docs`, `both`, `unknown`.
   - usa regole lessicali e una modalità opzionale OpenAI per affinare la decisione.

3. `Agent`
   - combina retrieval e codice sandbox.
   - se necessario, genera Python con OpenAI per rispondere a query sui dati.
   - esegue il codice in un ambiente minimale e restituisce i risultati.

## Scelte principali e motivazioni

### 1. Uso di WDI

Ho scelto il World Development Indicators (WDI) per la sua struttura tabellare con indicatori socio-economici e per la documentazione tecnica ricca di caveat.

Per garantire che il progetto sia eseguibile offline, il repository include un subset rappresentativo di WDI:
- 5 indicatori chiave
- 4 paesi
- 2 anni

Questo permette di mostrare l'architettura dell'agente senza dipendere da download massivi.

### 2. Retrieval documentale

Il retrieval usa:
- `scikit-learn` con TF-IDF
- ricerca sulle descrizioni e note degli indicatori

Motivazione:
- sufficiente per un prototipo leggero
- evita dipendenze su servizi esterni o modelli embedding pesanti

### 3. Esecuzione di codice

La componente di calcolo usa:
- `pandas` su un `DataFrame` strutturato
- un sandbox Python che espone solo `pd`, `np` e `df`

Se è presente `OPENAI_API_KEY`, l'agente prova a generare codice Python dal testo della domanda. Altrimenti, usa una modalità fallback basata su euristiche.

### 4. Decisione operativa

L'agente valuta la domanda con regole lessicali e segnali di ambiguità.

Esempi:
- domande su trend, valori, confronti → `data`
- domande su definizioni, significato, comparabilità → `docs`
- domande miste o vaghe → `both`

## Debolezze conosciute

1. **Corpus limitato**
   - il dataset è un subset e la documentazione è ridotta.
   - è un prototipo, non un indice completo di WDI.

2. **Classificazione della query**
   - le regole lessicali possono sbagliare su formulazioni complesse.
   - con più tempo userei un classificatore ML o un LLM dedicato.

3. **Esecuzione Python generata**
   - se si usa OpenAI, il codice può essere errato o sintatticamente sbagliato.
   - per mitigare la rottura, l'agente cattura le eccezioni e torna con un messaggio esplicito.

4. **RAG semplice**
   - la ricerca è basata solo su TF-IDF e non su un modello di comprensione profonda.
   - non c'è un meccanismo di verità aumentata o fact-checking avanzato.

## Cosa farei con più tempo

- estendere il dataset a tutto WDI usando il bulk download API
- costruire un indice di retrieval documentale su un corpus completo di note e metodologie
- aggiungere un vero componente di `query planning` con un LLM che decide il workflow
- supportare operatori semantici più sofisticati via `langchain` o `Haystack`
- aggiungere test end-to-end, validation su codice generato e monitoraggio di errori

## Esempi forniti

Il file `examples/run_examples.py` mostra:
- query dati dirette
- query documentali
- query miste
- query non risolvibili con il dataset disponibile
