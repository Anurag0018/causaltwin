"use client";

import { useEffect, useMemo } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Panel,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { EdgeOut } from "@/lib/api";
import { layoutGraph } from "@/lib/graph";
import TwinNode from "./TwinNode";

const nodeTypes = { twin: TwinNode };

const FIT_VIEW_OPTIONS = { padding: 0.15, maxZoom: 1.1 } as const;

function GraphCanvas({ edges, graphKey }: { edges: EdgeOut[]; graphKey: string }) {
  const laid = useMemo(() => layoutGraph(edges), [edges]);
  const { fitView } = useReactFlow();

  // Refit explicitly whenever a new graph arrives. The `fitView` prop
  // alone only frames the initial mount; nodes that load afterwards (or
  // whose measured size lands a frame later) can otherwise leave the
  // graph framed on empty canvas. Nodes carry initialWidth/Height, so
  // this rAF call can already compute the correct bounds.
  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      fitView({ ...FIT_VIEW_OPTIONS, duration: 250 });
    });
    return () => cancelAnimationFrame(raf);
  }, [laid, fitView]);

  return (
    <ReactFlow
      key={graphKey}
      nodes={laid.nodes}
      edges={laid.edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={FIT_VIEW_OPTIONS}
      nodesDraggable
      nodesConnectable={false}
      elementsSelectable={false}
      minZoom={0.25}
      maxZoom={1.6}
    >
      <Background variant={BackgroundVariant.Dots} color="#e3e6ea" gap={22} size={1.2} />
      <Controls showInteractive={false} position="bottom-left" />
      <Panel position="bottom-right">
        <button
          type="button"
          onClick={() => fitView({ ...FIT_VIEW_OPTIONS, duration: 250 })}
          className="rounded-lg border border-gray-200 bg-white px-2.5 py-1.5 text-xs font-medium text-gray-600 transition hover:bg-gray-50"
        >
          Fit view
        </button>
      </Panel>
    </ReactFlow>
  );
}

export default function CausalGraph({
  edges,
  loading,
  graphKey,
}: {
  edges: EdgeOut[] | null;
  loading: boolean;
  graphKey: string;
}) {
  if (loading && !edges?.length) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
        <span className="spinner" aria-hidden />
        <p className="text-[13px] text-gray-500">Running PCMCI discovery…</p>
      </div>
    );
  }

  if (!edges?.length) {
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
    <ReactFlowProvider>
      <GraphCanvas edges={edges} graphKey={graphKey} />
    </ReactFlowProvider>
  );
}
