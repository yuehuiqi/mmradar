#!/usr/bin/env python3
"""
Generates a Markdown summary of fusion experiment results.
Usage:
  python generate_fusion_summary.py \
      --matrix_dir /path/to/fusion_mmaud/outputs \
      --dataset MMAUD \
      --out_md /path/to/fusion_summary_mmaud.md
"""
import argparse, csv, json, os
from pathlib import Path
from datetime import datetime


METRIC_DISPLAY = {
    "combo": "组合",
    "method": "方法",
    "frames": "帧数",
    "mean_best_center_distance": "MBD↓",
    "center_ap_0.5m": "CAP@0.5m",
    "center_ap_1m": "CAP@1m",
    "center_ap_2m": "CAP@2m",
    "center_ap_4m": "CAP@4m",
    "bev_iou_ap_0.25": "BEV@0.25",
    "bev_iou_ap_0.5": "BEV@0.5",
    "bev_iou_ap_0.7": "BEV@0.7",
    "3d_iou_ap_0.25": "3D@0.25",
    "3d_iou_ap_0.3": "3D@0.3",
    "3d_iou_ap_0.5": "3D@0.5",
}


def fmt(v, precision=4):
    if v is None or v == "":
        return "–"
    try:
        return f"{float(v):.{precision}f}"
    except Exception:
        return str(v)


def load_csv(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def generate_md(matrix_dir: Path, dataset: str, out_md: Path):
    # Find all matrix summary CSVs
    csv_files = sorted(matrix_dir.glob("*/matrix_summary.csv"))
    if not csv_files:
        csv_files = sorted(matrix_dir.glob("matrix_summary.csv"))

    lines = [
        f"# YOLO × 雷达融合实验汇总 — {dataset}",
        f"",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 数据目录: `{matrix_dir}`",
        f"",
        "## 融合方法说明",
        "",
        "| 方法 | 说明 |",
        "|------|------|",
        "| radar | 纯雷达基线（无融合）|",
        "| score_boost | YOLO置信度加权提升雷达得分 |",
        "| score_gate | YOLO置信度阈值门控（抑制弱检测）|",
        "| image_refine | 利用YOLO框精炼雷达3D位置 |",
        "| proposal_bayes | 贝叶斯融合（雷达+YOLO联合置信度）|",
        "| sample_score_boost | 采样式得分提升 |",
        "| sample_bayes | 采样式贝叶斯融合 |",
        "",
        "---",
        "",
    ]

    all_rows = []
    for csv_path in csv_files:
        suite_name = csv_path.parent.name
        rows = load_csv(csv_path)
        all_rows.extend(rows)

        lines.append(f"## 实验套件: `{suite_name}`")
        lines.append(f"")
        if not rows:
            lines.append("> 无结果")
            lines.append("")
            continue

        cols = list(rows[0].keys())
        display_cols = [METRIC_DISPLAY.get(c, c) for c in cols]
        lines.append("| " + " | ".join(display_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in rows:
            vals = []
            for c in cols:
                v = row.get(c, "")
                try:
                    float(v)
                    vals.append(fmt(v))
                except (ValueError, TypeError):
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    if all_rows:
        lines += [
            "---",
            "",
            "## 综合最优结果",
            "",
        ]
        # Find best by center_ap_2m and by 3d_iou_ap_0.5
        for metric_key, metric_name in [
            ("center_ap_2m", "Center-AP @2m"),
            ("3d_iou_ap_0.5", "3D-AP @0.5"),
            ("mean_best_center_distance", "MBD (越小越好)"),
        ]:
            reverse = (metric_key != "mean_best_center_distance")
            valid = [r for r in all_rows if r.get(metric_key, "") not in ("", None)]
            if valid:
                best = sorted(valid, key=lambda r: float(r[metric_key]), reverse=reverse)[0]
                lines.append(f"**最优 {metric_name}**: `{best.get('combo','?')}` / "
                             f"`{best.get('method','?')}` = **{fmt(best[metric_key])}**  ")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summary] Written to {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    generate_md(
        matrix_dir=Path(args.matrix_dir),
        dataset=args.dataset.upper(),
        out_md=Path(args.out_md),
    )


if __name__ == "__main__":
    main()
