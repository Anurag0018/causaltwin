// Typed client for the CausalTwin FastAPI backend.
//
// Default base is the same-origin "/api" path, which next.config.js rewrites
// to the backend (http://localhost:8000) server-side. Set
// NEXT_PUBLIC_API_URL to an absolute backend URL to call it directly instead
// (the backend allows CORS from localhost:3000/3001 for that mode).

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api";

export interface EdgeOut {
  source: string;
  target: string;
  lag: number;
  p_value?: number;
}

export interface DiscoverResponse {
  n_steps: number;
  tau_max: number;
  discovered_edges: EdgeOut[];
  ground_truth_edges: EdgeOut[];
  metrics: {
    true_positives: number;
    false_positives: number;
    false_negatives: number;
    precision: number;
    recall: number;
    edge_f1: number;
    shd_style: number;
  };
  n_events: number;
}

export interface InterveneResponse {
  node: string;
  delta: number;
  target: string;
  target_shift: number;
  all_shifts: Record<string, number>;
}

export interface Candidate {
  param: string;
  delta: number;
  predicted_reduction_points: number;
  meets_target: boolean;
}

export interface RecommendResponse {
  param: string;
  direction: string;
  pct: number;
  kpi: string;
  effect_pct: number;
  ci_95: [number, number] | null;
  path: string;
  negligible_param: string | null;
  graph_version: number;
  refutation_status: string;
  delta: number;
  lever_baseline: number;
  target_baseline: number;
  target_reduction_points: number;
  predicted_reduction_points: number;
  negligible_candidate: { param: string; delta: number; predicted_reduction_points: number };
  candidates: Candidate[];
  explanation: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  versions: Record<string, string>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  discover: (body: { n_steps: number; tau_max: number; pc_alpha: number; seed: number }) =>
    request<DiscoverResponse>("/discover", { method: "POST", body: JSON.stringify(body) }),
  intervene: (body: { node: string; delta: number; target: string }) =>
    request<InterveneResponse>("/intervene", { method: "POST", body: JSON.stringify(body) }),
  recommend: (body: { target_reduction_points: number; lever: string }) =>
    request<RecommendResponse>("/recommend", { method: "POST", body: JSON.stringify(body) }),
  graphVersions: () => request<{ versions: number[] }>("/graph/versions"),
  graphDiff: (fromVersion: number, toVersion: number) =>
    request<{ from_version: number; to_version: number; added: EdgeOut[]; removed: EdgeOut[]; unchanged_count: number }>(
      `/graph/diff?from_version=${fromVersion}&to_version=${toVersion}`
    ),
};
