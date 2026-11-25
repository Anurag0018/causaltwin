# CausalTwin — Causal Digital Twin & Simulator

**SNH-AI-01 · SpectoV National Hackathon AI Track** — *AI That Discovers Why a System Is Failing*

Most AI systems predict **what** will happen. CausalTwin answers **why** — and what to change.

A conventional ML model can say *"machine temperature indicates 82% failure probability"*, but it
cannot distinguish **"Sensor X is correlated with the failure"** from **"changing Sensor X would
actually prevent the failure."** CausalTwin builds a Structural Causal Model (SCM) of a dynamic
system from time-series **and event** data, so it can:

- discover causal relationships (not correlations),
- simulate `do(X = x')` what-if interventions and predict downstream effects,
- attribute anomalies to upstream root causes,
- recommend the **smallest intervention** predicted to prevent or mitigate a failure —
  with a confidence and a plain-language explanation for every output.

This repo is the **Phase 0/1 starter scaffold** for a weekend build (see the weekend plan below).
**Phase 1 has started:** the NASA CMAPSS loader (`load_cmapss()` in `backend/app/data_loader.py`,
with RUL computation, capped RUL, constant-sensor handling, an ADF `stationarity_report()` and
per-unit `prepare_series()`) is implemented and tested.
The full design is the aligned v3 blueprint: *CausalTwin Engineering Blueprint & Implementation
Roadmap (Aligned to SNH-AI-01)*.

---

## Step Zero — lock the environment BEFORE writing code

The causal stack (Tigramite / DoWhy / EconML on numpy, scipy, scikit-learn) is extremely
version-sensitive. The pins in `backend/requirements.txt` were **verified together** — resolver,
clean install, and a runtime smoke test (PCMCI run, LinearDML fit, DoWhy `identify_effect`).
Do not bump any of them in isolation; in particular **dowhy is pinned to 0.13 on purpose**
(0.14 requires numpy>2.0 and cannot coexist with the numpy<2.0 stack).

> **Mac note:** the default `python3` on the dev Mac is **3.9**, which is too old
> (Tigramite 5.2.x needs Python ≥ 3.10). Check `python3.11 --version` **first**; if it's
> missing, install Python 3.11 before anything else. Always create the venv with
> `python3.11` explicitly — never bare `python3`.

```bash
cd causaltwin/backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py        # must print: SMOKE TEST PASSED
pip freeze > requirements-lock.txt  # commit this lock file; CI/Docker build from it
```

Run the pre-flight in the **same terminal** you'll work in (exports and venv activation
live only in that terminal):

```bash
which python && python --version    # expect: .../.venv/bin/python, Python 3.11.x
```

Docker alternative:

```bash
docker compose up backend           # python:3.11-slim, uvicorn on :8000
curl localhost:8000/health
```

Run the tests (module form, so the project interpreter and import path are right):

```bash
cd backend && python -m pytest        # from the activated venv
```

---

## Weekend plan (Sat 28 – Sun 29 Nov 2026)

| When | Phase | Done when |
|---|---|---|
| Sat morning | **Step Zero** | venv on Python 3.11, smoke test passes, `requirements-lock.txt` committed |
| Sat | **Phase 1 — Synthetic SCM & data/event pipeline** | `data_loader.generate()` (already working in the scaffold) extended: CMAPSS loader ✅ implemented (with RUL + preprocessing helpers), ADF report wired into preprocessing ✅ (`stationarity_report()` / `prepare_series()`) |
| Sun | **Phase 2 — Causal discovery** | PCMCI on the synthetic SCM prints its first **SHD + Edge F1** vs the known ground truth (`POST /discover`) |
| Later | Phase 3 DML + refutation → Phase 4 counterfactual/RCA/recommend → Phase 5 benchmark vs correlational LightGBM → Phase 6 React Flow UI | per blueprint |

The scaffold already contains working starter versions of every engine module
(discovery, estimator, simulator, root-cause, updater, explain) so the weekend
starts at Phase 1 code, not at setup.

## Repo map

```
causaltwin/
├── docker-compose.yml          # backend service (+ frontend placeholder, Phase 6)
├── backend/
│   ├── Dockerfile              # python:3.11-slim
│   ├── requirements.txt        # verified pins — see Step Zero
│   ├── app/
│   │   ├── main.py             # FastAPI: /health /discover /intervene /graph/versions /explain
│   │   ├── schemas.py          # Pydantic v2: NodeSpec, EdgeSpec, EventSpec, TwinGraph, ...
│   │   ├── data_loader.py      # synthetic SCM generator (known ground-truth DAG) + event stream + CMAPSS loader + ADF/preprocessing helpers
│   │   ├── discovery.py        # Tigramite PCMCI wrapper + SHD / Edge-F1 metrics
│   │   ├── estimator.py        # EconML LinearDML for treatment→KPI pairs (Phase 3)
│   │   ├── simulator.py        # structural equations + do(X) shift propagation (Phase 4)
│   │   ├── root_cause.py       # residual z-score root-cause ranking
│   │   ├── updater.py          # graph version registry + edge diff (Stage 0)
│   │   └── explain.py          # confidence / explanation cards (§3.5)
│   ├── scripts/smoke_test.py   # Step Zero verification
│   └── tests/test_scm_engine.py
├── frontend/                   # Phase 6 placeholder — see frontend/README.md
├── notebooks/                  # Phase 5 benchmark notebook — see notebooks/README.md
└── data/                       # local datasets (gitignored)
```

## Honest-claims rule (from the blueprint)

Interventional accuracy (SHD, ATE error, intervention RMSE) is claimed **only** from the
synthetic ground-truth SCM, where the true DAG and true effects are known. NASA CMAPSS is
observational run-to-failure data — on that tier CausalTwin demonstrates discovery,
graph updating and root-cause tracing, not intervention accuracy.

## Phase-1 additions (not in the scaffold's requirements)

`xgboost`, `polars` and `duckdb` are deliberately **not** pinned yet. Add them in Phase 1
only after re-running the full smoke test with them included, and regenerate
`requirements-lock.txt` — never add a package to this stack unverified.
