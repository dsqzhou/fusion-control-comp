#!/usr/bin/env python3
# coding=utf-8
"""
评测入口：读取 target（标准答案）与 infer_result（选手结果），按 README 规范输出 JSON，详情写 log。
用法: python evaluate.py <target_path> <infer_result_path>
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# 同目录 score 包
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from score import config, evaluate, per_metric_scores, total_score_with_tie_break

TASK_IDS = ["A1", "A2", "B1", "B2", "B3"]


def _ensure_np(obj):
    """JSON 反序列化后把 list 转成 numpy 供 score 使用；ref 里可能是 list。"""
    if isinstance(obj, list):
        return np.asarray(obj, dtype=np.float64)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "lcfs_per_step" and isinstance(v, list):
                # 保持为 list of arrays，供 score 逐步 LCFS 使用
                out[k] = [np.asarray(x, dtype=np.float64) for x in v]
            else:
                out[k] = _ensure_np(v)
        return out
    return obj


def _setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


def main() -> None:
    output = {
        "errorMsg": "user input is wrong, please check !",
        "score": 0,
        "scoreJson": {},
        "success": False,
    }

    log_dir = EVAL_DIR / "logs"
    logger = _setup_logging(log_dir)

    try:
        if len(sys.argv) < 3:
            output["errorMsg"] = "Arguments missing. Usage: python evaluate.py <target_path> <infer_result_path>"
            logger.warning(output["errorMsg"])
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return

        std_path = sys.argv[1]
        user_path = sys.argv[2]

        if not os.path.isfile(std_path):
            output["errorMsg"] = f"Target file not found: {std_path}"
            logger.warning(output["errorMsg"])
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return
        if not os.path.isfile(user_path):
            output["errorMsg"] = f"Infer result file not found: {user_path}"
            logger.warning(output["errorMsg"])
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return

        with open(std_path, "r", encoding="utf-8") as f:
            target_raw = json.load(f)
        with open(user_path, "r", encoding="utf-8") as f:
            infer_raw = json.load(f)

        # 校验键
        for tid in TASK_IDS:
            if tid not in target_raw:
                output["errorMsg"] = f"Target missing task: {tid}"
                logger.warning(output["errorMsg"])
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return
            if tid not in infer_raw:
                output["errorMsg"] = f"Infer result missing task: {tid}"
                logger.warning(output["errorMsg"])
                print(json.dumps(output, ensure_ascii=False, indent=2))
                return

        per_task_results = []
        task_scores_dict = {}
        task_epsilons_dict = {}
        task_metric_scores = {}
        for tid in TASK_IDS:
            ref = _ensure_np(target_raw[tid])
            traj_raw = infer_raw[tid]
            trajectory = _ensure_np(traj_raw["trajectory"])
            timeout = bool(traj_raw.get("timeout", False))

            result = evaluate(
                trajectory,
                ref,
                task_ids=[tid],
                timeouts={tid: timeout},
            )
            res0 = result["per_task_results"][0]
            per_task_results.append(res0)
            task_scores_dict[tid] = result["task_scores"][tid]
            task_epsilons_dict[tid] = {
                k: float(v) for k, v in result["task_epsilons"][tid].items()
            }

            w = config.get_task_metrics_and_weights(tid)
            ms = per_metric_scores(res0.per_step_epsilons, w, config.EPSILON_MAX, timeout)
            task_metric_scores[tid] = {
                "Ip": round(ms.get("Ip", 0.0), 6),
                "pos": round(ms.get("pos", 0.0), 6),
                "lcfs_points": round(ms.get("LCFS", 0.0), 6),
            }

        total_score, tie_key = total_score_with_tie_break(per_task_results)
        final_score = round(total_score, 6)

        # score1..score5 与 TASK_IDS 顺序一致：A1, A2, B1, B2, B3
        part_scores = {
            f"score{i}": round(float(task_scores_dict[tid]), 6)
            for i, tid in enumerate(TASK_IDS, start=1)
        }

        # 详情写 log
        detail = {
            "total_score": total_score,
            "tie_break_key": list(tie_key),
            "task_scores": task_scores_dict,
            "task_epsilons": task_epsilons_dict,
            "task_metric_scores": task_metric_scores,
        }
        logger.info("Evaluation detail: %s", json.dumps(detail, ensure_ascii=False, indent=2))

        score_json = {"score": final_score, **part_scores, "task_subscores": task_metric_scores}
        success_output = {
            "score": final_score,
            **part_scores,
            "scoreJson": score_json,
            "errorMsg": "",
            "success": True,
        }
        print(json.dumps(success_output, ensure_ascii=False, indent=2))

    except Exception as e:
        output["errorMsg"] = str(e)
        logger.exception("Evaluation failed")
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
