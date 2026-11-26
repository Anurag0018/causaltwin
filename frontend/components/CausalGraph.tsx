"use client";

import { useMemo } from "react";
import { Background, BackgroundVariant, Controls, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { EdgeOut } from "@/lib/api";
import { layoutGraph } from "@/lib/graph";
import TwinNode from "./TwinNode";

const nodeTypes = { twin: TwinNode };

export default function CausalGraph({
  edges,
  loading,
  graphKey,
}: {
  edges: EdgeOut[] | null;
  loading: boolean;
  graphKey: string;
}) {
  const laid = useMemo(() => (edges && edges.length ? layoutGraph(edges) : null), [edges]);

  if (loading && !laid) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
        <span className="spinner" aria-hidden />
        <p className="text-[13px] text-gray-500">Running PCMCI discovery…</p>
      </div>
    );
  }

  if (!laid) {
    return (
      <div className="flex h-full items-center justify-center px-8 text-center">
        <div>
          <p className="text-sm font-medium text-gray-700">No causal graph yet</p>
          <p className="mx-auto mt-1 max-w-xs text-[13px] leading-relaxed text-gray-500">
            Run discovery to build the causal graph from the synthetic machine data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <ReactFlow
      key={graphKey}
      nodes={laid.nodes}
      edges={laid.edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.16 }}
      nodesDraggable
      nodesConnectable={false}
      elementsSelectable={false}
      minZoom={0.4}
      maxZoom={1.6}
    >
      <Background variant={BackgroundVariant.Dots} color="#e3e6ea" gap={22} size={1.2} />
      <Controls showInteractive={false} position="bottom-left" />
    </ReactFlow>
  );
}
