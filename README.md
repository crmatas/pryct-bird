# SKYTERBIRD

An ornithology assistant that answers questions using a ChromaDB RAG knowledge
base and LangGraph. It includes a FastAPI service for saving and retrieving
search history.

## Tools

### Main assistant (`asistent.py`)

- `save_user_search`: saves a user's question through the local API.
- `list_user_searches`: retrieves the user's latest 100 searches. Type
  `historial` or `history` in the assistant terminal to display them.

### Additional agent (`multytool_agent.py`)

- `search_bird_encyclopedia`: searches the local bird encyclopedia.
- `check_sighting_weather`: retrieves current weather for a location using
  Open-Meteo.
- `register_sighting`: saves a sighting to `bitacora_avistamientos.txt` when
  explicitly requested by the user.

## Project structure

```text
.
├── asistent.py                 # Main assistant and LangGraph workflow
├── main.py                     # FastAPI service and SQLite search history
├── multytool_agent.py          # Additional bird, weather, and sighting tools
├── index.py                    # Indexes documents in ChromaDB
├── requirements.txt            # Python dependencies
├── documentos_pajaros/         # Source documents (.md, .txt, or .pdf)
├── chroma_pajaros_db/          # Persistent vector index
└── search_history.sqlite3      # Search history; created when the API starts
```

## Installation

From the project directory, create and activate a virtual environment, then
install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The assistant requires Ollama and the configured models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

If the ChromaDB index has not been created yet, add documents to
`documentos_pajaros/` and build it with:

```bash
python index.py
```

## Running the API and assistant

Open two terminals in the project directory and activate the virtual
environment in both.

**Terminal 1: start FastAPI**

```bash
fastapi run --reload main.py 
uvicorn main:app --reload
```

Interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

**Terminal 2: start the assistant**

```bash
python asistent.py
```

Ask a bird-related question to search the knowledge base, or type `historial`
to display saved searches. Type `exit` to quit.

## Connecting to the API from the terminal

Save a search:

```bash
curl -X POST http://127.0.0.1:8000/searches \
  -H "Content-Type: application/json" \
  -d '{"user_id":"local-user","query":"Habitat of the Australian magpie"}'
```

Retrieve a user's search history:

```bash
curl "http://127.0.0.1:8000/searches?user_id=local-user&limit=100"
```

By default, the assistant uses `http://localhost:8000` and the user ID
`local-user`. Set `SEARCH_API_URL` or `SEARCH_USER_ID` to change them. Set
`SEARCH_HISTORY_DB` to change the SQLite database location.
