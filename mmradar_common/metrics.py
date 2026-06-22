from __future__ import annotations

import numpy as np


def center_distance_metrics(det_annos, gt_annos, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """Greedy per-frame center matching for sparse single-class radar detection."""
    total_gt = 0
    total_pred = 0
    best_distances = []
    matched = {float(threshold): 0 for threshold in thresholds}

    for det, gt in zip(det_annos, gt_annos):
        gt_boxes = np.asarray(gt.get("gt_boxes_lidar", gt.get("gt_boxes", [])), dtype=np.float32)
        pred_boxes = np.asarray(det.get("boxes_lidar", []), dtype=np.float32)
        pred_scores = np.asarray(det.get("score", []), dtype=np.float32)

        gt_boxes = gt_boxes.reshape(-1, gt_boxes.shape[-1] if gt_boxes.size else 7)
        pred_boxes = pred_boxes.reshape(-1, pred_boxes.shape[-1] if pred_boxes.size else 7)
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        if not len(gt_boxes) or not len(pred_boxes):
            continue

        order = np.argsort(-pred_scores) if len(pred_scores) == len(pred_boxes) else np.arange(len(pred_boxes))
        distances = np.linalg.norm(gt_boxes[:, None, :3] - pred_boxes[None, order, :3], axis=2)
        best_distances.extend(distances.min(axis=1).tolist())

        for threshold in thresholds:
            used_gt: set[int] = set()
            for pred_column in range(distances.shape[1]):
                candidates = np.argsort(distances[:, pred_column])
                for gt_index in candidates:
                    gt_index = int(gt_index)
                    if gt_index not in used_gt and distances[gt_index, pred_column] <= threshold:
                        used_gt.add(gt_index)
                        break
            matched[float(threshold)] += len(used_gt)

    metrics = {
        "gt_objects": int(total_gt),
        "pred_objects": int(total_pred),
        "mean_best_center_distance": float(np.mean(best_distances)) if best_distances else float("inf"),
    }
    for threshold in thresholds:
        key = float(threshold)
        metrics[f"recall_{threshold:g}m"] = matched[key] / max(total_gt, 1)
        metrics[f"precision_{threshold:g}m"] = matched[key] / max(total_pred, 1)
    return metrics


def format_metrics(metrics):
    lines = [
        f"GT objects: {metrics['gt_objects']}",
        f"Predicted objects: {metrics['pred_objects']}",
        f"Mean best center distance: {metrics['mean_best_center_distance']:.4f} m",
    ]
    for threshold in (0.5, 1.0, 2.0, 4.0):
        lines.append(
            f"Center@{threshold:g}m: recall={metrics[f'recall_{threshold:g}m']:.4f}, "
            f"precision={metrics[f'precision_{threshold:g}m']:.4f}"
        )
    return "\n".join(lines)

