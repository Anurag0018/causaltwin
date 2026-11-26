"use client";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { TwinNode } from "@/lib/graph";

const handleStyle = {
  width: 6,
  height: 6,
  background: "#d1d5db",
  border: "none",
} as const;

export default function TwinNode({ data }: NodeProps<TwinNode>) {
  const isKpi = data.role === "target_kpi";
  const isEvent = data.role === "event";
  return (
    <div
      className="twin-node"
      style={{
        width: 176,
        background: "#ffffff",
        border: `${isKpi ? 1.5 : 1}px ${isEvent ? "dashed" : "solid"} ${isKpi ? "#4f46e5" : "#d8dce2"}`,
        borderRadius: 8,
        padding: "9px 12px 8px",
      }}
    >
      <Handle type="target" position={Position.Left} style={handleStyle} />
      <div style={{ fontSize: 13, fontWeight: 550, color: "#111827", lineHeight: 1.3 }}>{data.name}</div>
      <div
        style={{
          marginTop: 3,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
        }}
      >
        <span
          style={{
            fontSize: 9.5,
            fontWeight: 600,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: isKpi ? "#4f46e5" : "#9ca3af",
          }}
        >
          {data.tag}
        </span>
        {data.unit && <span style={{ fontSize: 10.5, color: "#9ca3af" }}>{data.unit}</span>}
      </div>
      <Handle type="source" position={Position.Right} style={handleStyle} />
    </div>
  );
}
