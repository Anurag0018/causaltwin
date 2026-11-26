# Frontend — CausalTwin Console

Next.js (App Router) + React Flow (`@xyflow/react`) console for the CausalTwin
backend. One page: the discovered causal graph, a what-if simulator, and the
recommendation engine with its explanation card.

## Run it

```bash
# 1. backend (from the repo root, venv activated)
cd backend && uvicorn app.main:app --port 8000

# 2. frontend
cd frontend && npm install && npm run dev     # http://localhost:3000
```

Production build: `npm run build && npm start`. Or both services together from
the repo root: `docker compose up --build` (UI on :3000, API on :8000).

## How it talks to the API

The browser calls the same-origin `/api/*` path; `next.config.js` rewrites it
server-side to the backend (`BACKEND_URL`, default `http://localhost:8000`;
docker-compose sets it to the backend service). No CORS setup is needed in
this mode. To call the backend directly instead, set
`NEXT_PUBLIC_API_URL=http://localhost:8000` — the backend allows CORS from
`localhost:3000`/`3001` for that mode.

## Layout of the app

- `app/page.tsx` — the console: slim header (backend status, graph version),
  graph canvas with a Discovered / Ground-truth toggle, a 380px sidebar
  (Discovery · What-if · Recommendation · Root cause) and a status strip with
  the last discovery's metrics.
- `components/CausalGraph.tsx` — React Flow canvas with custom nodes.
- `components/TwinNode.tsx` — node card: name, role tag (lever / KPI / state /
  exogenous / event) and unit; the KPI node carries the accent border.
- `components/InterventionPanel.tsx` — what-if panel (`POST /intervene`) and
  recommendation panel (`POST /recommend`).
- `components/RecommendationCard.tsx` — the recommendation as a designed card:
  smallest sufficient change, predicted effect with CI, causal path, the
  negligible alternative, refutation status — plus the explanation prose.
- `components/GraphVersionPicker.tsx` — selector over `GET /graph/versions`.
- `lib/api.ts` — typed API client (shapes mirror `backend/app/main.py`).
- `lib/graph.ts` — topological layering that turns API edges into positioned
  React Flow nodes/edges with lag labels.

Design: Inter (via `next/font`), one accent colour (indigo-600), 1px borders
instead of shadows, tabular figures for all numbers. Root-cause ranking is
evaluated offline in the engine and is not exposed by the API yet — the
sidebar reports its measured benchmark result rather than a live panel.
