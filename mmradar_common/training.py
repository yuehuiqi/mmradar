from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def record_periodic_metrics(work_dir, epoch, metrics, project=None, dataset=None):
    metrics_dir = Path(work_dir) / "periodic_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "epoch": int(epoch),
        "project": project,
        "dataset": dataset,
        "metrics": _json_value(metrics),
    }
    epoch_path = metrics_dir / f"epoch_{int(epoch):03d}.json"
    epoch_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    history_path = metrics_dir / "metrics_history.json"
    if history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            history = []
    else:
        history = []
    history = [item for item in history if int(item.get("epoch", -1)) != int(epoch)]
    history.append(record)
    history.sort(key=lambda item: int(item["epoch"]))
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return epoch_path
