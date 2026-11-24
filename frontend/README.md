# Frontend — Phase 6 (placeholder)

The twin UI is **Phase 6** of the blueprint, built after the engine
(Phases 1–5) produces numbers worth visualising. Nothing here is built yet.

## Planned stack

Next.js + Tailwind CSS + **React Flow** for the interactive causal graph.

```bash
npx create-next-app@latest . --typescript --tailwind --app
npm install reactflow
```

## Planned pages / components

- `app/page.tsx` — twin dashboard: graph canvas + KPI header
- `components/CausalGraph.tsx` — React Flow DAG: nodes coloured by role
  (lever / state / KPI / event), edges labelled with lag + effect size,
  root-cause heat colouring
- `components/InterventionPanel.tsx` — sliders for actionable levers;
  calls `POST /intervene`, renders before/after trajectories
- `components/RecommendationCard.tsx` — output of `POST /recommend` rendered
  with the explanation card from `POST /explain` (effect, CI, causal path,
  graph version, refutation status)
- `components/GraphVersionPicker.tsx` — version selector + edge-diff view
  over `GET /graph/versions` and `GET /graph/diff`

API base: the FastAPI backend on `:8000` (proxy `/api/*` to it in
`next.config.js`, the same pattern as the NeuroRoute console — no CORS).
