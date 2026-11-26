import { MarkerType, type Edge, type Node } from "@xyflow/react";
import type { EdgeOut } from "./api";

// Node metadata mirrors the ground-truth SCM registered in
// backend/app/main.py (roles / units are static properties of the system).
export const NODE_META: Record<string, { label: string; role: string; unit?: string }> = {
  load: { label: "Load", role: "exogenous" },
  fan_speed_nc: { label: "Fan speed", role: "actionable_lever", unit: "rpm" },
  coolant_temp: { label: "Coolant temp", role: "intermediate_state", unit: "°C" },
  vibration: { label: "Vibration", role: "intermediate_state", unit: "mm/s" },
  hpc_outlet_temp: { label: "HPC outlet temp", role: "intermediate_state", unit: "°C" },
  oil_pressure: { label: "Oil pressure", role: "intermediate_state", unit: "psi" },
  power_draw: { label: "Power draw", role: "intermediate_state", unit: "kW" },
  failure_risk: { label: "Failure risk", role: "target_kpi", unit: "0–100 index" },
  maintenance_event: { label: "Maintenance", role: "event", unit: "binary" },
};

export const ROLE_TAG: Record<string, string> = {
  exogenous: "Exogenous",
  actionable_lever: "Lever",
  intermediate_state: "State",
  target_kpi: "KPI",
  event: "Event",
};

export type TwinNodeData = {
  name: string;
  tag: string;
  unit?: string;
  role: string;
};

export type TwinNode = Node<TwinNodeData, "twin">;

/** Topological layering: each node's layer is the longest path from any
 *  source, so causes sit left of effects and same-depth nodes stack.
 *
 *  The layering MUST be cycle-safe. PCMCI returns lagged edges; collapsed
 *  to node pairs they can form directed cycles (e.g. vibration ->
 *  maintenance_event at lag 1 and maintenance_event -> vibration at
 *  lag 1). A naive longest-path relaxation never converges on a cycle —
 *  every pass pushes the cycle members (and everything downstream of
 *  them) one layer further right, which once produced a 5,136px-wide
 *  graph whose nodes all landed outside the fitted viewport. Here the
 *  longest path is computed by memoised DFS and any edge pointing back
 *  to a node already on the DFS stack is ignored for layering (it is
 *  still drawn — it simply routes as a feedback edge). */
export function layoutGraph(edges: EdgeOut[]): { nodes: TwinNode[]; edges: Edge[] } {
  const ids = new Set<string>();
  edges.forEach((e) => {
    ids.add(e.source);
    ids.add(e.target);
  });

  const preds = new Map<string, string[]>();
  ids.forEach((id) => preds.set(id, []));
  edges.forEach((e) => {
    if (e.source !== e.target) preds.get(e.target)!.push(e.source);
  });

  const layer = new Map<string, number>();
  const inStack = new Set<string>();
  const computeLayer = (id: string): number => {
    const memo = layer.get(id);
    if (memo !== undefined) return memo;
    inStack.add(id);
    let best = 0;
    for (const p of [...new Set(preds.get(id) ?? [])].sort()) {
      if (inStack.has(p)) continue; // back-edge of a cycle: not a layering step
      best = Math.max(best, computeLayer(p) + 1);
    }
    inStack.delete(id);
    layer.set(id, best);
    return best;
  };
  [...ids].sort().forEach(computeLayer);

  const byLayer = new Map<number, string[]>();
  layer.forEach((l, id) => {
    const arr = byLayer.get(l) ?? [];
    arr.push(id);
    byLayer.set(l, arr);
  });

  const X_GAP = 280;
  const Y_GAP = 116;
  const nodes: TwinNode[] = [];
  [...byLayer.entries()]
    .sort((a, b) => a[0] - b[0])
    .forEach(([l, layerIds]) => {
      layerIds.sort();
      const totalH = (layerIds.length - 1) * Y_GAP;
      layerIds.forEach((id, i) => {
        const meta = NODE_META[id] ?? { label: id, role: "intermediate_state" };
        nodes.push({
          id,
          type: "twin",
          position: { x: l * X_GAP, y: i * Y_GAP - totalH / 2 },
          // Lets fitView frame the graph correctly on the first paint,
          // before React Flow has measured the rendered cards (176px is
          // the fixed card width set in TwinNode).
          initialWidth: 176,
          initialHeight: 56,
          data: { name: meta.label, tag: ROLE_TAG[meta.role] ?? meta.role, unit: meta.unit, role: meta.role },
        });
      });
    });

  const flowEdges: Edge[] = edges.map((e, i) => ({
    id: `${e.source}->${e.target}@${e.lag}-${i}`,
    source: e.source,
    target: e.target,
    type: "smoothstep",
    label: `lag ${e.lag}`,
    labelStyle: { fontSize: 10, fill: "#6b7280", fontWeight: 500 },
    labelBgStyle: { fill: "#ffffff", stroke: "#e5e7eb", strokeWidth: 1 },
    labelBgPadding: [6, 3] as [number, number],
    labelBgBorderRadius: 999,
    style: { stroke: "#c3c9d1", strokeWidth: 1.4 },
    markerEnd: { type: MarkerType.ArrowClosed, color: "#9aa3af", width: 13, height: 13 },
  }));

  return { nodes, edges: flowEdges };
}
