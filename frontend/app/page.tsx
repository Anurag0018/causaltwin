"use client";

import { useCallback, useEffect, useState } from "react";
import CausalGraph from "@/components/CausalGraph";
import { RecommendationPanel, WhatIfPanel, primaryBtn } from "@/components/InterventionPanel";
import { api, type DiscoverResponse } from "@/lib/api";

function Section({
  label,
  children,
  last = false,
}: {
  label: string;
  children: React.ReactNode;
  last?: boolean;
}) {
  return (
    <section className={last ? "px-5 py-5" : "border-b border-gray-200 px-5 py-5"}>
      <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.09em] text-gray-500">{label}</h2>
      {children}
    </section>
  );
}

function Metric({ k, v }: { k: string; v: string }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-[#fafafa] px-3 py-2">
      <div className="tnum text-[15px] font-semibold leading-tight text-gray-900">{v}</div>
      <div className="mt-0.5 text-[10.5px] font-medium uppercase tracking-[0.07em] text-gray-500">{k}</div>
    </div>
  );
}

export default function Page() {
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const [stack, setStack] = useState<Record<string, string> | null>(null);
  const [graphVersion, setGraphVersion] = useState<number | null>(null);
  const [discovery, setDiscovery] = useState<DiscoverResponse | null>(null);
  const [showTruth, setShowTruth] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nSteps, setNSteps] = useState(2000);

  const checkHealth = useCallback(async () => {
    try {
      const h = await api.health();
      setBackendUp(true);
      setStack(h.versions);
    } catch {
      setBackendUp(false);
    }
    try {
      const v = await api.graphVersions();
      setGraphVersion(v.versions.length ? v.versions[v.versions.length - 1] : null);
    } catch {
      /* registry is informational only */
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  async function runDiscovery() {
    setBusy(true);
    setError(null);
    try {
      setDiscovery(await api.discover({ n_steps: nSteps, tau_max: 2, pc_alpha: 0.001, seed: 0 }));
    } catch (e) {
      setError(`Discovery failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  const edges = discovery ? (showTruth ? discovery.ground_truth_edges : discovery.discovered_edges) : null;
  const m = discovery?.metrics;

  return (
    <div className="flex h-screen flex-col bg-[#fafafa]">
      {/* ── Header ─────────────────────────────────────────── */}
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-gray-200 bg-white px-5">
        <div className="flex items-baseline gap-2">
          <span className="text-[15px] font-semibold tracking-tight text-gray-900">CausalTwin</span>
          <span className="text-[13px] text-gray-400">Console</span>
        </div>
        <div className="ml-auto flex items-center gap-4">
          {graphVersion !== null && (
            <span className="tnum rounded-md border border-gray-200 bg-[#fafafa] px-2 py-1 text-[11px] font-medium text-gray-600">
              Graph v{graphVersion}
            </span>
          )}
          <span className="flex items-center gap-2 text-xs text-gray-600">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                backendUp === null ? "bg-gray-300" : backendUp ? "bg-emerald-500" : "bg-red-500"
              }`}
            />
            {backendUp === null
              ? "Checking backend"
              : backendUp
                ? `Backend connected${stack ? ` · Tigramite ${stack.tigramite}` : ""}`
                : "Backend not reachable"}
          </span>
        </div>
      </header>

      {/* ── Body ───────────────────────────────────────────── */}
      <div className="flex min-h-0 flex-1">
        <main className="relative flex min-w-0 flex-1 flex-col bg-white">
          {backendUp === false && (
            <div className="border-b border-red-200 bg-red-50 px-5 py-3 text-[13px] leading-relaxed text-red-800">
              <span className="font-medium">Backend not reachable.</span> Start it from the activated venv with{" "}
              <code className="rounded border border-red-200 bg-white px-1.5 py-0.5 font-mono text-xs">
                uvicorn app.main:app --port 8000
              </code>{" "}
              in <span className="font-mono text-xs">backend/</span>, then reload this page.
            </div>
          )}
          {error && (
            <div className="border-b border-red-200 bg-red-50 px-5 py-2.5 text-[13px] text-red-800">{error}</div>
          )}

          <div className="flex items-center justify-between px-5 pb-1 pt-4">
            <h1 className="text-xl font-semibold tracking-tight text-gray-900">
              {showTruth ? "Causal graph — ground truth" : "Causal graph"}
            </h1>
            <div className="flex overflow-hidden rounded-md border border-gray-300 text-xs font-medium">
              {(["Discovered", "Ground truth"] as const).map((name) => {
                const active = (name === "Ground truth") === showTruth;
                return (
                  <button
                    key={name}
                    disabled={!discovery}
                    onClick={() => setShowTruth(name === "Ground truth")}
                    className={`px-3 py-1.5 transition disabled:cursor-not-allowed disabled:opacity-40 ${
                      active ? "bg-gray-900 text-white" : "bg-white text-gray-600 hover:bg-gray-50"
                    }`}
                  >
                    {name}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="min-h-0 flex-1">
            <CausalGraph
              edges={edges}
              loading={busy}
              graphKey={`${showTruth ? "truth" : "disc"}-${discovery?.discovered_edges.length ?? 0}-${discovery?.n_steps ?? 0}`}
            />
          </div>
        </main>

        {/* ── Sidebar ──────────────────────────────────────── */}
        <aside className="hidden w-[380px] shrink-0 overflow-y-auto border-l border-gray-200 bg-white lg:block">
          <Section label="Discovery">
            <div className="space-y-3">
              <div>
                <div className="flex items-baseline justify-between text-[13px] text-gray-700">
                  <span>Series length</span>
                  <span className="tnum font-medium text-gray-900">{nSteps.toLocaleString()} steps</span>
                </div>
                <input
                  type="range"
                  min={600}
                  max={2000}
                  step={200}
                  value={nSteps}
                  onChange={(e) => setNSteps(Number(e.target.value))}
                  className="mt-1.5 w-full accent-indigo-600"
                />
              </div>
              <button className={primaryBtn} onClick={runDiscovery} disabled={busy || backendUp === false}>
                {busy && <span className="spinner spinner-sm" aria-hidden />}
                {busy ? "Running PCMCI" : "Run discovery"}
              </button>
              <p className="text-xs leading-relaxed text-gray-500">
                PCMCI at tau_max 2, pc_alpha 0.001 — the best configuration from the tuning sweep (Edge F1 0.73 at
                n=2,000 vs 0.60 for the baseline settings).
              </p>
              {m && (
                <div className="grid grid-cols-2 gap-2 pt-1">
                  <Metric k="Precision" v={m.precision.toFixed(2)} />
                  <Metric k="Recall" v={m.recall.toFixed(2)} />
                  <Metric k="Edge F1" v={m.edge_f1.toFixed(2)} />
                  <Metric k="SHD" v={String(m.shd_style)} />
                </div>
              )}
            </div>
          </Section>

          <Section label="What-if">
            <WhatIfPanel onError={setError} />
          </Section>

          <Section label="Recommendation">
            <RecommendationPanel onError={setError} />
          </Section>

          <Section label="Root cause" last>
            <p className="text-[13px] leading-relaxed text-gray-600">
              Per-incident root-cause ranking runs in the engine and is evaluated offline; it is not exposed by the
              API yet, so there is no live panel to show.
            </p>
            <div className="mt-3 grid grid-cols-2 gap-2">
              <Metric k="Top-1 accuracy" v="1.00" />
              <Metric k="Top-3 accuracy" v="1.00" />
            </div>
            <p className="mt-2 text-xs leading-relaxed text-gray-500">
              Across 12 injected faults (4 nodes × 3 seeds), residual z-score ranking with a clean-reference fit —
              measured on the synthetic SCM.
            </p>
          </Section>
        </aside>
      </div>

      {/* ── Status strip ───────────────────────────────────── */}
      <footer className="flex h-9 shrink-0 items-center justify-between border-t border-gray-200 bg-white px-5 text-xs text-gray-500">
        <span className="tnum">
          {discovery
            ? `Last discovery: n=${discovery.n_steps.toLocaleString()} · ${discovery.discovered_edges.length} edges found · Edge F1 ${discovery.metrics.edge_f1.toFixed(2)} · ${discovery.n_events} events`
            : "No discovery run in this session"}
        </span>
        <span className="hidden md:block">
          Interventional numbers are measured on the synthetic ground-truth SCM; CMAPSS results are structure-only.
        </span>
      </footer>
    </div>
  );
}
