#!/usr/bin/env python3
"""
Generates a Markdown summary of ablation experiment results.
Usage:
  python generate_ablation_summary.py \
      --output_root /path/to/PFA-NET/output/mmradar_models \
      --tag mmaud_abl_v1 \
      --dataset MMAUD \
      --out_md /path/to/ablation_summary_mmaud.md
"""
import argparse, json, os, glob, pickle
from pathlib import Path
from datetime import datetime

MODELS_MMAUD = [
    ("pfanet_mmaud_full",                  "mmaud_all14_v1",  "PFA (baseline)"),
    ("dg_pfanet_mmaud_ablation_nograph",   "mmaud_abl_v1",    "DG-PFA w/o Graph"),
    ("dg_pfanet_mmaud_ablation_noaux",     "mmaud_abl_v1",    "DG-PFA w/o AuxAttn"),
    ("dg_pfanet_mmaud_full",               "mmaud_abl_v1",    "DG-PFA (full)"),
]

MODELS_AIQII = [
    ("pfanet_aiqii_full",                  "aiqii_abl_v1",    "PFA (baseline)"),
    ("dg_pfanet_aiqii_ablation_nograph",   "aiqii_abl_v1",    "DG-PFA w/o Graph"),
    ("dg_pfanet_aiqii_ablation_noaux",     "aiqii_abl_v1",    "DG-PFA w/o AuxAttn"),
    ("dg_pfanet_aiqii_full",               "aiqii_abl_v1",    "DG-PFA (full)"),
]

METRIC_KEYS = [
    "center_ap_0.5m", "center_ap_1m", "center_ap_2m", "center_ap_4m",
    "bev_iou_ap_0.25", "bev_iou_ap_0.5", "bev_iou_ap_0.7",
    "3d_iou_ap_0.25", "3d_iou_ap_0.3", "3d_iou_ap_0.5",
    "mean_best_center_distance",
]

def find_best_epoch(model_dir: Path):
    """Find best epoch by center_ap_2m from periodic_metrics json."""
    metrics_glob = list(model_dir.glob("periodic_metrics/*.json"))
    if not metrics_glob:
        metrics_glob = list(model_dir.glob("*/periodic_metrics/*.json"))

    best_epoch, best_val = -1, -1.0
    best_metrics = {}

    def try_entry(item):
        nonlocal best_epoch, best_val, best_metrics
        if not isinstance(item, dict):
            return
        ep = item.get("epoch", -1)
        m = item.get("metrics", item)  # some files wrap in "metrics" key, some don't
        val = m.get("center_ap_2m", m.get("center_ap_2.0m", -1.0))
        if val is None:
            val = -1.0
        if float(val) > best_val:
            best_val = float(val)
            best_epoch = ep
            best_metrics = m

    for jf in metrics_glob:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            # data may be a single dict or a list of dicts (metrics_history.json)
            items = data if isinstance(data, list) else [data]
            for item in items:
                try_entry(item)
        except Exception:
            pass

    return best_epoch, best_metrics


def get_model_row(output_root: Path, model_name: str, tag: str, display_name: str):
    model_dir = output_root / model_name / tag
    if not model_dir.exists():
        return display_name, None, None, f"NOT FOUND: {model_dir}"

    best_epoch, metrics = find_best_epoch(model_dir)
    if best_epoch == -1 or not metrics:
        return display_name, None, None, "No metrics found"

    # locate result.pkl for best epoch
    pkl_candidates = [
        model_dir / "periodic_eval" / f"epoch_{best_epoch:03d}" / "result.pkl",
        model_dir / "periodic_eval" / f"epoch_{best_epoch}" / "result.pkl",
        model_dir / "eval" / f"epoch_{best_epoch}" / "val" / "result.pkl",
    ]
    pkl_path = None
    for c in pkl_candidates:
        if c.exists():
            pkl_path = str(c)
            break

    return display_name, best_epoch, metrics, pkl_path


def fmt(v, precision=4):
    if v is None:
        return "–"
    try:
        return f"{float(v):.{precision}f}"
    except Exception:
        return str(v)


def generate_md(output_root: Path, dataset: str, out_md: Path):
    models = MODELS_MMAUD if dataset.upper() == "MMAUD" else MODELS_AIQII

    lines = [
        f"# DG-PFA 消融实验汇总 — {dataset}",
        f"",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 输出目录: `{output_root}`",
        f"",
        "## 实验配置",
        "",
        "| 变体 | VFE 模块 | 动态图 | 辅助注意力 |",
        "|------|----------|--------|------------|",
        "| PFA (baseline) | RadarPillarFeatureAttention | ✗ | ✗ |",
        "| DG-PFA w/o Graph | DynamicGraphRadarPillarFeatureAttention | ✗ (NUM_LAYERS=0) | ✓ |",
        "| DG-PFA w/o AuxAttn | DynamicGraphRadarPillarFeatureAttention | ✓ (NUM_LAYERS=2) | ✗ |",
        "| DG-PFA (full) | DynamicGraphRadarPillarFeatureAttention | ✓ (NUM_LAYERS=2) | ✓ |",
        "",
        "**训练参数:** 80 epochs, batch_size=4, workers=2, eval_interval=5",
        "",
        "---",
        "",
        "## 性能汇总",
        "",
    ]

    # Metrics table header
    header_cols = ["模型", "最优轮次", "Center-AP @0.5m", "Center-AP @1m", "Center-AP @2m",
                   "Center-AP @4m", "BEV-AP @0.25", "BEV-AP @0.5", "3D-AP @0.25", "3D-AP @0.5",
                   "MBD↓"]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    all_rows = []
    detail_blocks = []

    for model_name, tag, display_name in models:
        name, best_epoch, metrics, pkl_path = get_model_row(output_root, model_name, tag, display_name)
        if metrics is None:
            lines.append(f"| {name} | — | — | — | — | — | — | — | — | — | — |")
            detail_blocks.append(f"### {name}\n\n> ⚠️ 尚未完成或未找到结果: {pkl_path}\n")
            continue

        def m(k):
            # try multiple key variants
            for key in [k, k.replace(".", "_"), k.replace("_", "."), k.replace("ap_", "ap@")]:
                v = metrics.get(key)
                if v is not None:
                    return v
            return None

        row = [
            name,
            str(best_epoch),
            fmt(m("center_ap_0.5m") or m("center_ap_0.5")),
            fmt(m("center_ap_1m") or m("center_ap_1.0m")),
            fmt(m("center_ap_2m") or m("center_ap_2.0m")),
            fmt(m("center_ap_4m") or m("center_ap_4.0m")),
            fmt(m("bev_iou_ap_0.25")),
            fmt(m("bev_iou_ap_0.5")),
            fmt(m("3d_iou_ap_0.25")),
            fmt(m("3d_iou_ap_0.5")),
            fmt(m("mean_best_center_distance")),
        ]
        lines.append("| " + " | ".join(row) + " |")
        all_rows.append((name, best_epoch, metrics, pkl_path))

        detail_lines = [
            f"### {name}",
            f"",
            f"- **最优轮次:** epoch {best_epoch}",
            f"- **结果文件:** `{pkl_path}`",
            f"",
            f"#### 完整指标",
            f"",
            f"| 指标 | 值 |",
            f"|------|----|",
        ]
        for k in METRIC_KEYS:
            v = m(k)
            detail_lines.append(f"| {k} | {fmt(v)} |")
        detail_lines.append("")
        detail_blocks.append("\n".join(detail_lines))

    lines += [
        "",
        "---",
        "",
        "## 逐模型详情",
        "",
    ]
    lines += detail_blocks

    lines += [
        "---",
        "",
        "## result.pkl 路径（供融合实验使用）",
        "",
    ]
    for model_name, tag, display_name in models:
        name, best_epoch, metrics, pkl_path = get_model_row(output_root, model_name, tag, display_name)
        status = f"`{pkl_path}`" if pkl_path else "**未就绪**"
        lines.append(f"- **{name}** (epoch {best_epoch}): {status}")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] Written to {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--dataset", required=True, choices=["MMAUD", "AIQII", "mmaud", "aiqii"])
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    generate_md(
        output_root=Path(args.output_root),
        dataset=args.dataset.upper(),
        out_md=Path(args.out_md),
    )


if __name__ == "__main__":
    main()
