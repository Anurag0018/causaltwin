"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export default function GraphVersionPicker() {
  const [versions, setVersions] = useState<number[] | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .graphVersions()
      .then((r) => {
        if (cancelled) return;
        setVersions(r.versions);
        setSelected(r.versions.length ? r.versions[r.versions.length - 1] : null);
      })
      .catch(() => {
        if (!cancelled) setError("Version registry not reachable.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return <p className="text-sm text-gray-500">{error}</p>;
  if (!versions) return <p className="text-sm text-gray-400">Loading versions…</p>;
  if (!versions.length) return <p className="text-sm text-gray-500">No graph versions registered yet.</p>;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <label htmlFor="graph-version" className="text-sm text-gray-700">
          Graph version
        </label>
        <select
          id="graph-version"
          className="rounded-md border border-gray-300 bg-white px-2 py-1 text-sm text-gray-900"
          value={selected ?? ""}
          onChange={(e) => setSelected(Number(e.target.value))}
        >
          {versions.map((v) => (
            <option key={v} value={v}>
              v{v}
            </option>
          ))}
        </select>
      </div>
      <p className="text-xs leading-relaxed text-gray-500">
        {versions.length === 1
          ? "Only the ground-truth graph (v1) is registered. New versions appear here when discovery results are promoted into the registry."
          : `${versions.length} versions registered. Edge diffs between versions are available from the backend at /graph/diff.`}
      </p>
    </div>
  );
}
