"""Pydantic v2 schemas for the CausalTwin SCM graph, events and API payloads."""
from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class NodeSpec(BaseModel):
    id: str
    role: str = Field(description="actionable_lever | intermediate_state | target_kpi | event | exogenous")
    unit: Optional[str] = None
    bounds: Optional[Tuple[float, float]] = None


class EdgeSpec(BaseModel):
    source: str
    target: str
    lag: int = 1
    ate: Optional[float] = None
    ci_95: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    refutation_passed: Optional[bool] = None


class EventSpec(BaseModel):
    ts: int
    type: str = Field(description="maintenance | alarm | operator_action | config_change")
    node: str
    is_treatment: bool = False


class TwinGraph(BaseModel):
    twin_id: str = "synthetic_machine_scm_v1"
    graph_version: int = 1
    max_lag: int = 2
    nodes: List[NodeSpec] = []
    edges: List[EdgeSpec] = []
    events: List[EventSpec] = []


class InterventionRequest(BaseModel):
    node: str = "fan_speed_nc"
    delta: float = Field(description="Change applied to the node, in that node's units")
    target: str = "failure_risk"
    n_steps: int = 1200
    seed: int = 0


class RecommendationResponse(BaseModel):
    param: str
    pct: float
    kpi: str
    effect_pct: float
    ci_95: Optional[Tuple[float, float]] = None
    path: str = ""
    negligible_param: Optional[str] = None
    graph_version: int = 1
    refutation_status: str = "passed"
    explanation: str = ""
