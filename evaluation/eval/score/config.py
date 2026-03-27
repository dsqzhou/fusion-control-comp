# Copyright @2025 ENN Energy(enn.cn)
# 初赛评估常量：零分阈值、5 个子任务权重与指标列表。

from typing import Dict, List, Any

METRIC_IP = "Ip"
METRIC_POS = "pos"
METRIC_LCFS = "LCFS"

EPSILON_MAX: Dict[str, float] = {
    METRIC_IP: 0.05,
    METRIC_POS: 3.0,
    METRIC_LCFS: 3.0,
}

TASK_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "A1": {"total": 25, "metrics": {METRIC_IP: 8, METRIC_POS: 8, METRIC_LCFS: 9}},
    "A2": {"total": 15, "metrics": {METRIC_IP: 7, METRIC_POS: 4, METRIC_LCFS: 4}},
    "B1": {"total": 30, "metrics": {METRIC_IP: 10, METRIC_POS: 10, METRIC_LCFS: 10}},
    "B2": {"total": 15, "metrics": {METRIC_IP: 7, METRIC_POS: 4, METRIC_LCFS: 4}},
    "B3": {"total": 15, "metrics": {METRIC_IP: 4, METRIC_POS: 7, METRIC_LCFS: 4}},
}

TASK_IDS: List[str] = ["A1", "A2", "B1", "B2", "B3"]

CONFIGURATION_TASKS: Dict[str, List[str]] = {
    "A": ["A1", "A2"],
    "B": ["B1", "B2", "B3"],
}

TOTAL_MAX_SCORE: float = 100.0
TOTAL_TIME_MS: float = 500.0
TOTAL_STEPS: int = 500


def get_epsilon_max(metric: str) -> float:
    return EPSILON_MAX.get(metric, 3.0)


def get_task_metrics_and_weights(task_id: str) -> Dict[str, float]:
    t = TASK_WEIGHTS.get(task_id)
    if not t:
        return {}
    return dict(t["metrics"])
