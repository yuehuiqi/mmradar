#!/usr/bin/env python3
"""Generate a detailed Markdown report from the MMRadar suite metrics JSON."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_CORE_METRICS = [
    "center_ap_0.5m",
    "center_ap_1m",
    "center_ap_2m",
    "center_ap_4m",
    "center_f1_1m",
    "center_precision_1m",
    "center_recall_1m",
    "bev_iou_ap_0.25",
    "bev_iou_ap_0.5",
    "bev_iou_f1_0.5",
    "3d_iou_ap_0.25",
    "3d_iou_ap_0.5",
    "3d_iou_f1_0.5",
    "mean_best_center_distance",
    "mean_best_bev_iou",
    "mean_best_3d_iou",
    "pred_objects",
]

DEFAULT_HISTORY_CORE = [
    "center_ap_1m",
    "center_ap_2m",
    "center_f1_1m",
    "center_recall_1m",
    "center_precision_1m",
    "bev_iou_ap_0.5",
    "bev_iou_f1_0.5",
    "3d_iou_ap_0.25",
    "3d_iou_ap_0.5",
    "mean_best_center_distance",
    "mean_best_bev_iou",
    "mean_best_3d_iou",
]

PERCENT_LIKE_PREFIXES = (
    "center_ap_",
    "center_f1_",
    "center_precision_",
    "center_recall_",
    "bev_iou_ap_",
    "bev_iou_f1_",
    "bev_iou_precision_",
    "bev_iou_recall_",
    "3d_iou_ap_",
    "3d_iou_f1_",
    "3d_iou_precision_",
    "3d_iou_recall_",
    "precision_",
    "recall_",
)
PERCENT_LIKE_EXACT = {"mean_best_bev_iou", "mean_best_3d_iou"}
COUNT_EXACT = {
    "frames",
    "frames_with_predictions",
    "frames_without_predictions",
    "gt_objects",
    "pred_objects",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a full Markdown report for one MMRadar full-suite run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing ALL_MODELS_METRICS.json and suite_status.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path. Default: <run-dir>/MMAUD_V1_ALL_METRICS_REPORT.md",
    )
    return parser.parse_args()


def metric_order_key(metric: str) -> tuple[int, str]:
    group = 99
    if metric.startswith("recall/"):
        group = 0
    elif metric in COUNT_EXACT:
        group = 1
    elif metric.startswith("score_"):
        group = 2
    elif metric in {
        "mean_best_center_distance",
        "rmse_best_center_distance",
        "median_best_center_distance",
        "p90_best_center_distance",
        "p95_best_center_distance",
        "max_best_center_distance",
    }:
        group = 3
    elif metric in {"mean_best_bev_iou", "mean_best_3d_iou"}:
        group = 4
    elif metric.startswith("mean_abs_center_error"):
        group = 5
    elif metric.startswith("mean_abs_size_error"):
        group = 6
    elif metric.startswith("mean_abs_yaw_error"):
        group = 7
    elif (
        metric.startswith("center_")
        or metric.startswith("precision_")
        or metric.startswith("recall_")
    ):
        group = 8
    elif metric.startswith("bev_iou_"):
        group = 9
    elif metric.startswith("3d_iou_"):
        group = 10
    return group, metric


def escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def wsl_to_windows(path: str | None) -> str:
    if not path:
        return ""
    path = str(path)
    if path.startswith("/mnt/e/"):
        return "E:\\" + path[len("/mnt/e/") :].replace("/", "\\")
    return path


def seconds_to_hms(sec: Any) -> str:
    if sec is None:
        return "—"
    sec = int(round(float(sec)))
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    seconds = sec % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def is_percent_metric(metric: str) -> bool:
    return (
        metric.startswith(PERCENT_LIKE_PREFIXES)
        or metric.startswith("recall/roi_")
        or metric.startswith("recall/rcnn_")
        or metric in PERCENT_LIKE_EXACT
    )


def is_count_metric(metric: str) -> bool:
    return metric in COUNT_EXACT or any(x in metric for x in ["_tp_", "_fp_", "_fn_"])


def fmt_raw(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        abs_value = abs(value)
        if abs(value - round(value)) < 1e-10 and abs_value >= 10:
            return str(int(round(value)))
        if abs_value >= 1000:
            return f"{value:.2f}"
        if abs_value >= 100:
            return f"{value:.3f}"
        if abs_value >= 10:
            return f"{value:.4f}"
        if value == 0:
            return "0"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return escape_md(value)


def fmt_core(metric: str, value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        if is_count_metric(metric):
            return str(int(round(value)))
        if is_percent_metric(metric):
            return f"{100.0 * value:.2f}%"
        if "distance" in metric or "error" in metric:
            return f"{value:.3f}"
        if metric.startswith("score_"):
            return f"{value:.3f}"
        return fmt_raw(value)
    return escape_md(value)


def markdown_table(headers: list[Any], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(escape_md(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_md(cell) for cell in row) + " |")
    return "\n".join(lines)


def final_metrics(data: dict[str, Any], model: str) -> dict[str, Any]:
    return data[model].get("final", {}).get("metrics", {}) or {}


def history(data: dict[str, Any], model: str) -> list[dict[str, Any]]:
    return data[model].get("history") or []


def history_metric(entry: dict[str, Any], metric: str) -> Any:
    return (entry.get("metrics") or {}).get(metric)


def best_epoch(
    data: dict[str, Any], model: str, metric: str, larger_is_better: bool = True
) -> tuple[Any, Any]:
    values: list[tuple[float, Any]] = []
    for entry in history(data, model):
        value = history_metric(entry, metric)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            values.append((float(value), entry.get("epoch")))
    if not values:
        return "—", "—"
    values.sort(key=lambda item: item[0], reverse=larger_is_better)
    best_value, epoch = values[0]
    return epoch, best_value


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    metrics_path = run_dir / "ALL_MODELS_METRICS.json"
    status_path = run_dir / "suite_status.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {}
    )
    return data, status, metrics_path, status_path


def build_metric_union(data: dict[str, Any], models: list[str]) -> list[str]:
    seen: set[str] = set()
    metric_union: list[str] = []
    for model in models:
        for metric in final_metrics(data, model):
            if metric not in seen:
                seen.add(metric)
                metric_union.append(metric)
    return sorted(metric_union, key=metric_order_key)


def collect_time_range(
    data: dict[str, Any], status: dict[str, Any], models: list[str]
) -> tuple[list[datetime], list[datetime], float]:
    starts: list[datetime] = []
    ends: list[datetime] = []
    elapsed_sum = 0.0
    for model in models:
        model_status = status.get(model, data[model].get("status", {})) or {}
        elapsed = model_status.get("elapsed_sec")
        if elapsed is not None:
            elapsed_sum += float(elapsed)
        started_at = model_status.get("started_at")
        if not started_at:
            continue
        try:
            started_dt = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        starts.append(started_dt)
        if elapsed is not None:
            ends.append(started_dt + timedelta(seconds=float(elapsed)))
    return starts, ends, elapsed_sum


def generate_report(run_dir: Path, output: Path) -> None:
    data, status, metrics_path, status_path = load_run(run_dir)
    models = list(data.keys())
    metric_union = build_metric_union(data, models)
    starts, ends, elapsed_sum = collect_time_range(data, status, models)

    lines: list[str] = []
    lines.append("# MMAUD 七模型训练实验结果汇总（mmaud_v1）")
    lines.append("")
    lines.append(
        "这份报告由 `environment/full_runs/mmaud/mmaud_v1/ALL_MODELS_METRICS.json` "
        "提取生成，包含 7 个模型的最终指标，以及每 5 个 epoch 的完整验证指标。"
    )
    lines.append("")
    lines.append("## 1. 运行概况")
    lines.append("")
    lines.append("- 数据集：`mmaud`")
    lines.append("- Run tag：`mmaud_v1`")
    lines.append(f"- 模型数量：`{len(models)}`")
    lines.append("- 验证频率：每 `5` 个 epoch 一次")
    lines.append("- 最终 epoch：`80`")
    lines.append(f"- 指标 JSON：`{metrics_path}`")
    lines.append(f"- 本报告：`{output}`")
    if starts and ends:
        lines.append(f"- 首个模型开始时间：`{min(starts).strftime('%Y-%m-%d %H:%M:%S')}`")
        lines.append(f"- 最后一个模型结束时间：`{max(ends).strftime('%Y-%m-%d %H:%M:%S')}`")
        wall_seconds = (max(ends) - min(starts)).total_seconds()
        lines.append(
            f"- 7 个模型训练耗时合计：`{seconds_to_hms(elapsed_sum)}`；"
            f"顺序连续运行墙钟时间约 `{seconds_to_hms(wall_seconds)}`"
        )
    lines.append("")
    lines.append(
        "说明：完整指标表中的值保持原始数值；前面的核心对比表为了直观，"
        "把 AP / Precision / Recall / F1 / IoU 类 0-1 指标显示为百分比。"
    )
    lines.append("")

    status_rows: list[list[Any]] = []
    for model in models:
        model_status = status.get(model, data[model].get("status", {})) or {}
        status_rows.append(
            [
                model,
                model_status.get("state", "—"),
                model_status.get("attempt", "—"),
                model_status.get("epoch", data[model].get("final", {}).get("epoch", "—")),
                seconds_to_hms(model_status.get("elapsed_sec")),
                model_status.get("returncode", "—"),
                wsl_to_windows(
                    model_status.get("output_dir") or data[model].get("output_dir") or ""
                ),
                wsl_to_windows(model_status.get("log") or ""),
            ]
        )
    lines.append(
        markdown_table(
            ["模型", "状态", "尝试次数", "最终 epoch", "耗时", "返回码", "输出目录", "日志"],
            status_rows,
        )
    )
    lines.append("")

    lines.append("## 2. 最终核心指标对比")
    lines.append("")
    lines.append(
        "这张表适合先看整体。毫米波雷达目标框的 3D IoU 通常会比中心距离指标更苛刻，"
        "所以这里同时列出 Center / BEV IoU / 3D IoU 三组指标。"
    )
    lines.append("")
    core_rows = []
    for model in models:
        metrics = final_metrics(data, model)
        core_rows.append([model] + [fmt_core(metric, metrics.get(metric)) for metric in DEFAULT_CORE_METRICS])
    lines.append(markdown_table(["模型"] + DEFAULT_CORE_METRICS, core_rows))
    lines.append("")

    lines.append("## 3. 最终指标小结")
    lines.append("")
    rank_keys = [
        ("center_ap_1m", "Center AP@1m", True),
        ("center_ap_2m", "Center AP@2m", True),
        ("bev_iou_ap_0.5", "BEV AP@0.5", True),
        ("3d_iou_ap_0.25", "3D AP@0.25", True),
        ("3d_iou_ap_0.5", "3D AP@0.5", True),
        ("mean_best_center_distance", "平均中心误差", False),
        ("mean_best_bev_iou", "平均最佳 BEV IoU", True),
        ("mean_best_3d_iou", "平均最佳 3D IoU", True),
    ]
    rank_rows: list[list[Any]] = []
    for metric, label, larger_is_better in rank_keys:
        values = []
        for model in models:
            value = final_metrics(data, model).get(metric)
            if isinstance(value, (int, float)):
                values.append((float(value), model))
        values.sort(key=lambda item: item[0], reverse=larger_is_better)
        if not values:
            continue
        best_value, best_model = values[0]
        second_value, second_model = values[1] if len(values) > 1 else ("—", "—")
        rank_rows.append(
            [
                label,
                best_model,
                fmt_core(metric, best_value),
                second_model,
                fmt_core(metric, second_value),
            ]
        )
    lines.append(markdown_table(["指标", "最好模型", "最好值", "第二名", "第二名值"], rank_rows))
    lines.append("")
    lines.append(
        "从最终 epoch 看，`PFANet` 在中心距离类指标上最稳，`VoxelNeXt` 在 BEV AP@0.5 "
        "和 3D AP@0.5 上最高；`CenterPoint` 这次能正常训练和验证，但 3D IoU/近距离中心"
        "指标明显偏低，后续如果要深挖可以优先检查其 box size、z 轴、yaw 或 score 分布。"
    )
    lines.append("")

    lines.append("## 4. 每个模型的最佳 epoch（按核心指标）")
    lines.append("")
    best_rows = []
    best_specs = [
        ("center_ap_1m", "Center AP@1m", True),
        ("center_ap_2m", "Center AP@2m", True),
        ("bev_iou_ap_0.5", "BEV AP@0.5", True),
        ("3d_iou_ap_0.25", "3D AP@0.25", True),
        ("3d_iou_ap_0.5", "3D AP@0.5", True),
        ("mean_best_center_distance", "平均中心误差", False),
    ]
    for model in models:
        row = [model]
        for metric, _label, larger_is_better in best_specs:
            epoch, value = best_epoch(data, model, metric, larger_is_better)
            row.append(f"epoch {epoch} / {fmt_core(metric, value)}" if epoch != "—" else "—")
        best_rows.append(row)
    lines.append(
        markdown_table(
            [
                "模型",
                "Center AP@1m 最佳",
                "Center AP@2m 最佳",
                "BEV AP@0.5 最佳",
                "3D AP@0.25 最佳",
                "3D AP@0.5 最佳",
                "平均中心误差最低",
            ],
            best_rows,
        )
    )
    lines.append("")

    lines.append("## 5. 最终完整指标总表")
    lines.append("")
    lines.append(
        "下面是最终 epoch 的完整指标。PCDet 系列模型有 `recall/roi_*`、`recall/rcnn_*` "
        "这几个框架内置 recall；Det3D 系列的 CenterPoint / PillarNet-LTS 没有这些字段，"
        "所以显示为 `—`。"
    )
    lines.append("")
    full_rows = []
    for metric in metric_union:
        full_rows.append([metric] + [fmt_raw(final_metrics(data, model).get(metric)) for model in models])
    lines.append(markdown_table(["指标"] + models, full_rows))
    lines.append("")

    lines.append("## 6. 各模型每 5 个 epoch 的指标走势")
    lines.append("")
    lines.append(
        "每个模型先给一张核心走势表；完整的每 5 轮所有指标矩阵放在折叠块里，"
        "打开即可逐项核对。"
    )
    lines.append("")
    for index, model in enumerate(models, start=1):
        lines.append(f"### 6.{index} {model}")
        lines.append("")
        model_status = status.get(model, data[model].get("status", {})) or {}
        lines.append(
            f"- 输出目录：`{wsl_to_windows(model_status.get('output_dir') or data[model].get('output_dir') or '')}`"
        )
        lines.append(f"- 完整历史 JSON：`{wsl_to_windows(data[model].get('history_file') or '')}`")
        lines.append(f"- 日志：`{wsl_to_windows(model_status.get('log') or '')}`")
        lines.append("")
        hist = history(data, model)
        hist_rows = []
        for entry in hist:
            hist_rows.append(
                [entry.get("epoch", "—")]
                + [fmt_core(metric, history_metric(entry, metric)) for metric in DEFAULT_HISTORY_CORE]
            )
        lines.append(markdown_table(["epoch"] + DEFAULT_HISTORY_CORE, hist_rows))
        lines.append("")
        lines.append("<details>")
        lines.append(f"<summary>{model} 每 5 个 epoch 的完整指标矩阵（所有指标）</summary>")
        lines.append("")
        epochs = [entry.get("epoch", "—") for entry in hist]
        matrix_rows = []
        for metric in metric_union:
            values = [history_metric(entry, metric) for entry in hist]
            if all(value is None for value in values):
                continue
            matrix_rows.append([metric] + [fmt_raw(value) for value in values])
        lines.append(markdown_table(["指标"] + [f"epoch {epoch}" for epoch in epochs], matrix_rows))
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("## 7. 指标字段解释")
    lines.append("")
    explain_rows = [
        ["frames", "验证集帧数。"],
        ["frames_with_predictions / frames_without_predictions", "有/无预测框的帧数。"],
        ["gt_objects / pred_objects", "验证集 GT 数量 / 预测框数量。"],
        ["score_min / score_mean / score_median / score_max", "模型输出 score 的分布统计。"],
        [
            "center_*_0.5m / 1m / 2m / 4m",
            "按预测框中心点到 GT 中心点距离匹配得到的 TP/FP/FN、Precision、Recall、F1、AP。",
        ],
        [
            "bev_iou_*_0.1 / 0.25 / 0.3 / 0.5 / 0.7",
            "按旋转 BEV 框 IoU 阈值匹配得到的 TP/FP/FN、Precision、Recall、F1、AP。",
        ],
        [
            "3d_iou_*_0.1 / 0.25 / 0.3 / 0.5 / 0.7",
            "按 3D 框 IoU 阈值匹配得到的 TP/FP/FN、Precision、Recall、F1、AP。",
        ],
        [
            "mean_best_center_distance / rmse / median / p90 / p95 / max",
            "每个 GT 与最佳预测框之间中心距离的统计，越小越好。",
        ],
        [
            "mean_best_bev_iou / mean_best_3d_iou",
            "每个 GT 与最佳预测框之间的平均最佳 IoU，越大越好。",
        ],
        ["mean_abs_center_error_x/y/z", "GT 与最佳预测框中心坐标的平均绝对误差。"],
        ["mean_abs_size_error_dx/dy/dz", "GT 与最佳预测框尺寸的平均绝对误差。"],
        ["mean_abs_yaw_error_rad/deg", "GT 与最佳预测框朝向角平均绝对误差。"],
        ["recall/roi_*、recall/rcnn_*", "OpenPCDet 系列自带 recall 统计，Det3D 系列没有该字段。"],
    ]
    lines.append(markdown_table(["字段", "含义"], explain_rows))
    lines.append("")

    lines.append("## 8. 原始文件")
    lines.append("")
    lines.append(f"- 完整机器可读 JSON：`{metrics_path}`")
    lines.append(f"- 运行状态 JSON：`{status_path}`")
    lines.append(f"- 旧版简表：`{run_dir / 'ALL_MODELS_METRICS.md'}`")
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Metric fields: {len(metric_union)}")
    print(f"Models: {len(models)}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output or (run_dir / "MMAUD_V1_ALL_METRICS_REPORT.md")
    generate_report(run_dir, output.resolve())


if __name__ == "__main__":
    main()
