# GraphRAG Pipeline Server

A Flask app exposing three endpoints for a GraphRAG pipeline:

- `POST /extract-graph` — heuristic entity/relationship extraction from raw text (no external LLM call; uses regex-based proper-noun detection + pattern matching for FOUNDED/DEVELOPED/INTEGRATED_INTO/HIRED/AUTHORED relations).
- `POST /graph-query` — multi-hop BFS reasoning over a supplied entity/relationship graph to answer a natural-language question, returning `answer`, `reasoning_path`, and `hops`.
- `POST /community-summary` — template-based natural-language summary of a connected sub-community.

Tested locally — all three endpoints return correct results for the sample LangChain/Harrison Chase/OpenAI scenario (2-hop reasoning) and a second Anthropic/Dario Amodei scenario (hire + authorship).

## Files
- `app.py` — the full server (single file, only depends on Flask).
- `requirements.txt` — `Flask`, `gunicorn`.
- `Procfile` — `web: gunicorn -w 2 -b 0.0.0.0:$PORT app:app` (for Render/Railway/Heroku-style platforms).

The app reads the `PORT` env var (falls back to 8080), so it works on any platform that injects `PORT`.

## Deploying it yourself (pick one)

**Glitch (no login needed for anonymous projects):**
1. Go to glitch.com → New Project → "Import from GitHub" or choose a Python/Flask starter.
2. Replace the starter's files with `app.py` and `requirements.txt` from this folder.
3. Glitch auto-installs requirements and starts the app; grab the `https://<project-name>.glitch.me` URL it gives you.

**Render.com / Railway.app (free tier, needs a GitHub-connected account):**
1. Push these files to a GitHub repo.
2. Create a new Web Service pointing at the repo — both platforms auto-detect the `Procfile` and `requirements.txt`.
3. Deploy; use the URL Render/Railway gives you as the grader's base URL.

**PythonAnywhere (free tier, needs signup):**
1. Upload `app.py`, create a new Flask web app pointing at it.
2. Reload; your base URL will be `https://<username>.pythonanywhere.com`.

**Run locally / on your own server:**
```bash
pip install -r requirements.txt
python app.py          # dev server on :8080
# or, for production:
gunicorn -w 2 -b 0.0.0.0:8080 app:app
```

## Quick test once deployed

```bash
BASE=https://your-deployed-url

curl -s -X POST $BASE/extract-graph -H "Content-Type: application/json" -d '{
  "chunk_id": "C001",
  "text": "LangChain was created by Harrison Chase. LangChain integrates with OpenAI. Sam Altman founded OpenAI."
}'

curl -s -X POST $BASE/graph-query -H "Content-Type: application/json" -d '{
  "question": "Who created the framework that integrates with OpenAI?",
  "graph": {
    "entities": [
      {"name": "LangChain", "type": "Framework"},
      {"name": "Harrison Chase", "type": "Person"},
      {"name": "OpenAI", "type": "Organization"}
    ],
    "relationships": [
      {"source": "Harrison Chase", "target": "LangChain", "relation": "CREATED"},
      {"source": "LangChain", "target": "OpenAI", "relation": "INTEGRATED_INTO"}
    ]
  }
}'

curl -s -X POST $BASE/community-summary -H "Content-Type: application/json" -d '{
  "community_id": "COM_001",
  "entities": ["LangChain", "Harrison Chase", "OpenAI"],
  "relationships": [
    {"source": "Harrison Chase", "target": "LangChain", "relation": "CREATED"},
    {"source": "LangChain", "target": "OpenAI", "relation": "INTEGRATED_INTO"}
  ]
}'
```

Once you have a public base URL, that's what you paste into the grader's "GraphRAG Base URL" field (it appends `/extract-graph`, `/graph-query`, `/community-summary` automatically).
