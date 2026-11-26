"use client";

import { useState } from "react";
import { api, type InterveneResponse, type RecommendResponse } from "@/lib/api";
import RecommendationCard from "./RecommendationCard";

export const primaryBtn =
  "inline-flex items-center gap-2 rounded-md bg-indigo-600 px-3.5 py-[7px] text-[13px] font-medium text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-gray-300";
const sliderLabel = "flex items-baseline justify-between text-[13px] text-gray-700";

export function WhatIfPanel({ onError }: { onError: (msg: string | null) => void }) {
  const [delta, setDelta] = useState(80);
  const [result, setResult] = useState<InterveneResponse | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    onError(null);
    try {
      setResult(await api.intervene({ node: "fan_speed_nc", delta, target: "failure_risk" }));
    } catch (e) {
      onError(`What-if failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      <div>
        <div className={sliderLabel}>
          <span>
            Sustained change to <span className="font-medium text-gray-900">fan speed</span>
          </span>
          <span className="tnum font-medium text-gray-900">
            {delta > 0 ? "+" : ""}
            {delta} rpm
          </span>
        </div>
        <input
          type="range"
          min={-150}
          max={150}
          step={10}
          value={delta}
          onChange={(e) => setDelta(Number(e.target.value))}
          className="mt-1.5 w-full accent-indigo-600"
        />
      </div>
      <button className={primaryBtn} onClick={run} disabled={busy}>
        {busy && <span className="spinner spinner-sm" aria-hidden />}
        {busy ? "Simulating" : "Simulate intervention"}
      </button>
      {result && (
        <div className="rounded-lg border border-gray-200 bg-[#fafafa] px-3.5 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-gray-500">
            Predicted failure risk shift
          </div>
          <div className="tnum mt-1 text-xl font-semibold text-gray-900">
            {result.target_shift > 0 ? "+" : ""}
            {result.target_shift.toFixed(2)}{" "}
            <span className="text-[13px] font-normal text-gray-500">points, steady state</span>
          </div>
          <p className="mt-1 text-xs leading-relaxed text-gray-500">
            Propagated through the fitted structural equations, not a correlation lookup.
          </p>
        </div>
      )}
    </div>
  );
}

export function RecommendationPanel({ onError }: { onError: (msg: string | null) => void }) {
  const [target, setTarget] = useState(10);
  const [rec, setRec] = useState<RecommendResponse | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    onError(null);
    try {
      setRec(await api.recommend({ target_reduction_points: target, lever: "fan_speed_nc" }));
    } catch (e) {
      onError(`Recommendation failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      <div>
        <div className={sliderLabel}>
          <span>Target failure risk reduction</span>
          <span className="tnum font-medium text-gray-900">{target.toFixed(0)} points</span>
        </div>
        <input
          type="range"
          min={5}
          max={20}
          step={1}
          value={target}
          onChange={(e) => setTarget(Number(e.target.value))}
          className="mt-1.5 w-full accent-indigo-600"
        />
      </div>
      <button className={primaryBtn} onClick={run} disabled={busy}>
        {busy && <span className="spinner spinner-sm" aria-hidden />}
        {busy ? "Searching" : "Recommend smallest change"}
      </button>
      {busy && <p className="text-xs text-gray-400">Searching the lever grid and bootstrapping the confidence interval…</p>}
      {rec && <RecommendationCard rec={rec} />}
    </div>
  );
}
