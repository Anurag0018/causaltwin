import type { RecommendResponse } from "@/lib/api";

function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-[5px]">
      <dt className="shrink-0 text-[11px] font-semibold uppercase tracking-[0.08em] text-gray-500">{k}</dt>
      <dd className="tnum text-right text-[13px] font-medium text-gray-900">{children}</dd>
    </div>
  );
}

export default function RecommendationCard({ rec }: { rec: RecommendResponse }) {
  return (
    <div className="overflow-hidden rounded-lg border border-gray-200">
      <div className="border-b border-gray-200 bg-[#fafafa] px-3.5 py-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-gray-500">
          Smallest sufficient change
        </div>
        <div className="mt-1 text-[15px] font-semibold leading-snug text-gray-900">
          {rec.direction} fan speed by {rec.delta.toFixed(0)} rpm{" "}
          <span className="font-normal text-gray-500">(+{rec.pct.toFixed(2)}%)</span>
        </div>
        <div className="tnum mt-0.5 text-[13px] text-gray-600">
          Predicted effect: −{rec.predicted_reduction_points.toFixed(2)} points failure risk (−
          {rec.effect_pct.toFixed(2)}%)
          {rec.ci_95 && (
            <>
              {" "}
              · 95% CI {rec.ci_95[0].toFixed(2)}% to {rec.ci_95[1].toFixed(2)}%
            </>
          )}
        </div>
      </div>
      <div className="px-3.5 py-2.5">
        <p className="text-[13px] leading-relaxed text-gray-700">{rec.explanation}</p>
        <dl className="mt-2 divide-y divide-gray-100 border-t border-gray-100">
          <Row k="Causal path">{rec.path.replaceAll(" drives ", " → ")}</Row>
          <Row k="Tried, negligible">
            {rec.negligible_candidate.param} +{rec.negligible_candidate.delta.toFixed(0)} →{" "}
            {rec.negligible_candidate.predicted_reduction_points.toFixed(2)} pts
          </Row>
          <Row k="Confidence">
            Refutation {rec.refutation_status} · graph v{rec.graph_version}
          </Row>
        </dl>
      </div>
    </div>
  );
}
