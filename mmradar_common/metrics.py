from __future__ import annotations

import math

import numpy as np


CENTER_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
IOU_THRESHOLDS = (0.1, 0.25, 0.3, 0.5, 0.7)


def _boxes(annotation):
    value = annotation.get(
        "boxes_lidar",
        annotation.get("gt_boxes_lidar", annotation.get("gt_boxes", [])),
    )
    return np.asarray(value, dtype=np.float32).reshape(-1, 7)


def _scores(annotation, count):
    value = annotation.get("score", annotation.get("scores", []))
    scores = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(scores) != count:
        return np.ones((count,), dtype=np.float32)
    return scores


def _names(annotation, count):
    value = annotation.get("name", annotation.get("names", []))
    names = np.asarray(value).astype(str).reshape(-1)
    if len(names) != count:
        return np.full((count,), "", dtype="<U1")
    return names


def _filter_annotation_by_name(annotation, class_name, prediction):
    boxes = _boxes(annotation)
    names = _names(annotation, len(boxes))
    mask = names == class_name
    filtered = {
        "name": names[mask],
        "boxes_lidar": boxes[mask],
        "gt_boxes_lidar": boxes[mask],
        "gt_boxes": boxes[mask],
    }
    if prediction:
        scores = _scores(annotation, len(boxes))
        filtered["score"] = scores[mask]
    return filtered


def _rectangle_corners(box):
    half_x, half_y = max(float(box[3]), 0.0) / 2, max(float(box[4]), 0.0) / 2
    corners = np.asarray(
        [[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y]],
        dtype=np.float64,
    )
    cosine, sine = math.cos(float(box[6])), math.sin(float(box[6]))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return corners @ rotation.T + np.asarray(box[:2], dtype=np.float64)


def _polygon_area(polygon):
    if len(polygon) < 3:
        return 0.0
    polygon = np.asarray(polygon, dtype=np.float64)
    return abs(
        np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
        - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
    ) * 0.5


def _inside(point, edge_start, edge_end):
    edge = edge_end - edge_start
    relative = point - edge_start
    return edge[0] * relative[1] - edge[1] * relative[0] >= -1e-9


def _line_intersection(start, end, clip_start, clip_end):
    direction = end - start
    clip_direction = clip_end - clip_start
    denominator = direction[0] * clip_direction[1] - direction[1] * clip_direction[0]
    if abs(denominator) < 1e-12:
        return end
    relative = clip_start - start
    factor = (
        relative[0] * clip_direction[1] - relative[1] * clip_direction[0]
    ) / denominator
    return start + factor * direction


def _intersection_area(first, second):
    output = [point for point in first]
    for index in range(len(second)):
        clip_start = second[index]
        clip_end = second[(index + 1) % len(second)]
        input_polygon = output
        output = []
        if not input_polygon:
            break
        start = input_polygon[-1]
        for end in input_polygon:
            end_inside = _inside(end, clip_start, clip_end)
            start_inside = _inside(start, clip_start, clip_end)
            if end_inside:
                if not start_inside:
                    output.append(_line_intersection(start, end, clip_start, clip_end))
                output.append(end)
            elif start_inside:
                output.append(_line_intersection(start, end, clip_start, clip_end))
            start = end
    return _polygon_area(output)


def _pairwise_geometry(pred_boxes, gt_boxes):
    distances = np.empty((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    bev_iou = np.zeros_like(distances)
    iou3d = np.zeros_like(distances)
    if not len(pred_boxes) or not len(gt_boxes):
        return distances, bev_iou, iou3d

    distances[:] = np.linalg.norm(
        pred_boxes[:, None, :3] - gt_boxes[None, :, :3], axis=2
    )
    pred_corners = [_rectangle_corners(box) for box in pred_boxes]
    gt_corners = [_rectangle_corners(box) for box in gt_boxes]

    for pred_index, pred_box in enumerate(pred_boxes):
        pred_area = max(float(pred_box[3] * pred_box[4]), 0.0)
        pred_volume = pred_area * max(float(pred_box[5]), 0.0)
        pred_bottom = float(pred_box[2] - pred_box[5] / 2)
        pred_top = float(pred_box[2] + pred_box[5] / 2)
        for gt_index, gt_box in enumerate(gt_boxes):
            gt_area = max(float(gt_box[3] * gt_box[4]), 0.0)
            intersection_area = _intersection_area(
                pred_corners[pred_index], gt_corners[gt_index]
            )
            union_area = pred_area + gt_area - intersection_area
            if union_area > 1e-9:
                bev_iou[pred_index, gt_index] = intersection_area / union_area

            gt_bottom = float(gt_box[2] - gt_box[5] / 2)
            gt_top = float(gt_box[2] + gt_box[5] / 2)
            overlap_height = max(0.0, min(pred_top, gt_top) - max(pred_bottom, gt_bottom))
            intersection_volume = intersection_area * overlap_height
            gt_volume = gt_area * max(float(gt_box[5]), 0.0)
            union_volume = pred_volume + gt_volume - intersection_volume
            if union_volume > 1e-9:
                iou3d[pred_index, gt_index] = intersection_volume / union_volume
    return distances, bev_iou, iou3d


def _average_precision(recall, precision):
    if not len(recall):
        return 0.0
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changed = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1]))


def _score_ordered_metrics(frame_values, frame_scores, total_gt, threshold, higher_is_better):
    predictions = []
    for frame_index, scores in enumerate(frame_scores):
        predictions.extend(
            (float(score), frame_index, pred_index)
            for pred_index, score in enumerate(scores)
        )
    predictions.sort(key=lambda item: item[0], reverse=True)
    used_gt = [set() for _ in frame_values]
    true_positives = []
    false_positives = []

    for _, frame_index, pred_index in predictions:
        values = frame_values[frame_index]
        candidates = []
        for gt_index in range(values.shape[1]):
            if gt_index in used_gt[frame_index]:
                continue
            value = float(values[pred_index, gt_index])
            passed = value >= threshold if higher_is_better else value <= threshold
            if passed:
                candidates.append((value, gt_index))
        if candidates:
            selected = (
                max(candidates, key=lambda item: item[0])
                if higher_is_better
                else min(candidates, key=lambda item: item[0])
            )
            used_gt[frame_index].add(selected[1])
            true_positives.append(1.0)
            false_positives.append(0.0)
        else:
            true_positives.append(0.0)
            false_positives.append(1.0)

    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recall_curve = cumulative_tp / max(total_gt, 1)
    precision_curve = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    tp = int(cumulative_tp[-1]) if len(cumulative_tp) else 0
    fp = int(cumulative_fp[-1]) if len(cumulative_fp) else 0
    fn = max(total_gt - tp, 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ap": _average_precision(recall_curve, precision_curve),
    }


def comprehensive_detection_metrics(
    det_annos,
    gt_annos,
    center_thresholds=CENTER_THRESHOLDS,
    iou_thresholds=IOU_THRESHOLDS,
):
    frame_distances = []
    frame_bev_iou = []
    frame_iou3d = []
    frame_scores = []
    nearest_distances = []
    nearest_bev_iou = []
    nearest_iou3d = []
    center_errors = []
    size_errors = []
    yaw_errors = []
    all_scores = []
    total_gt = 0
    total_pred = 0
    frames_with_predictions = 0

    for det, gt in zip(det_annos, gt_annos):
        pred_boxes = _boxes(det)
        gt_boxes = _boxes(gt)
        scores = _scores(det, len(pred_boxes))
        distances, bev_iou, iou3d = _pairwise_geometry(pred_boxes, gt_boxes)
        frame_distances.append(distances)
        frame_bev_iou.append(bev_iou)
        frame_iou3d.append(iou3d)
        frame_scores.append(scores)
        all_scores.extend(scores.tolist())
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        frames_with_predictions += int(len(pred_boxes) > 0)

        if len(pred_boxes) and len(gt_boxes):
            for gt_index in range(len(gt_boxes)):
                pred_index = int(np.argmin(distances[:, gt_index]))
                nearest_distances.append(float(distances[pred_index, gt_index]))
                nearest_bev_iou.append(float(bev_iou[pred_index, gt_index]))
                nearest_iou3d.append(float(iou3d[pred_index, gt_index]))
                center_errors.append(np.abs(pred_boxes[pred_index, :3] - gt_boxes[gt_index, :3]))
                size_errors.append(np.abs(pred_boxes[pred_index, 3:6] - gt_boxes[gt_index, 3:6]))
                yaw_delta = (
                    float(pred_boxes[pred_index, 6] - gt_boxes[gt_index, 6]) + math.pi
                ) % (2 * math.pi) - math.pi
                yaw_errors.append(abs(yaw_delta))

    metrics = {
        "frames": len(gt_annos),
        "frames_with_predictions": frames_with_predictions,
        "frames_without_predictions": len(gt_annos) - frames_with_predictions,
        "gt_objects": int(total_gt),
        "pred_objects": int(total_pred),
    }

    score_array = np.asarray(all_scores, dtype=np.float32)
    if len(score_array):
        metrics.update(
            score_min=float(score_array.min()),
            score_mean=float(score_array.mean()),
            score_median=float(np.median(score_array)),
            score_max=float(score_array.max()),
        )

    distance_array = np.asarray(nearest_distances, dtype=np.float32)
    if len(distance_array):
        metrics.update(
            mean_best_center_distance=float(distance_array.mean()),
            rmse_best_center_distance=float(np.sqrt(np.mean(distance_array ** 2))),
            median_best_center_distance=float(np.median(distance_array)),
            p90_best_center_distance=float(np.percentile(distance_array, 90)),
            p95_best_center_distance=float(np.percentile(distance_array, 95)),
            max_best_center_distance=float(distance_array.max()),
            mean_best_bev_iou=float(np.mean(nearest_bev_iou)),
            mean_best_3d_iou=float(np.mean(nearest_iou3d)),
        )
        mean_center_error = np.mean(center_errors, axis=0)
        mean_size_error = np.mean(size_errors, axis=0)
        for axis, value in zip("xyz", mean_center_error):
            metrics[f"mean_abs_center_error_{axis}"] = float(value)
        for axis, value in zip(("dx", "dy", "dz"), mean_size_error):
            metrics[f"mean_abs_size_error_{axis}"] = float(value)
        metrics["mean_abs_yaw_error_rad"] = float(np.mean(yaw_errors))
        metrics["mean_abs_yaw_error_deg"] = float(np.degrees(np.mean(yaw_errors)))
    else:
        metrics["mean_best_center_distance"] = float("inf")

    for threshold in center_thresholds:
        label = f"{threshold:g}m"
        values = _score_ordered_metrics(
            frame_distances, frame_scores, total_gt, threshold, higher_is_better=False
        )
        for name, value in values.items():
            metrics[f"center_{name}_{label}"] = value
        metrics[f"precision_{label}"] = values["precision"]
        metrics[f"recall_{label}"] = values["recall"]

    for prefix, frame_values in (("bev", frame_bev_iou), ("3d", frame_iou3d)):
        for threshold in iou_thresholds:
            label = f"{threshold:g}"
            values = _score_ordered_metrics(
                frame_values, frame_scores, total_gt, threshold, higher_is_better=True
            )
            for name, value in values.items():
                metrics[f"{prefix}_iou_{name}_{label}"] = value
    return metrics


def center_distance_metrics(det_annos, gt_annos, thresholds=CENTER_THRESHOLDS):
    """Backward-compatible entry point returning the full MMRadar metric set."""
    metrics = comprehensive_detection_metrics(
        det_annos, gt_annos, center_thresholds=thresholds
    )
    gt_class_names = []
    for gt in gt_annos:
        boxes = _boxes(gt)
        for name in _names(gt, len(boxes)):
            if name and name not in gt_class_names:
                gt_class_names.append(str(name))

    if len(gt_class_names) > 1:
        for class_name in gt_class_names:
            class_det_annos = [
                _filter_annotation_by_name(det, class_name, prediction=True)
                for det in det_annos
            ]
            class_gt_annos = [
                _filter_annotation_by_name(gt, class_name, prediction=False)
                for gt in gt_annos
            ]
            class_metrics = comprehensive_detection_metrics(
                class_det_annos,
                class_gt_annos,
                center_thresholds=thresholds,
            )
            for key, value in class_metrics.items():
                metrics[f"class/{class_name}/{key}"] = value
    return metrics


def format_metrics(metrics):
    lines = [
        f"Frames: {metrics.get('frames', 0)}",
        f"GT objects: {metrics['gt_objects']}",
        f"Predicted objects: {metrics['pred_objects']}",
        f"Frames without predictions: {metrics.get('frames_without_predictions', 0)}",
        f"Mean best center distance: {metrics['mean_best_center_distance']:.4f} m",
    ]
    for threshold in CENTER_THRESHOLDS:
        label = f"{threshold:g}m"
        lines.append(
            f"Center@{label}: precision={metrics[f'center_precision_{label}']:.4f}, "
            f"recall={metrics[f'center_recall_{label}']:.4f}, "
            f"F1={metrics[f'center_f1_{label}']:.4f}, AP={metrics[f'center_ap_{label}']:.4f}"
        )
    for prefix, title in (("bev", "BEV"), ("3d", "3D")):
        for threshold in IOU_THRESHOLDS:
            label = f"{threshold:g}"
            lines.append(
                f"{title} IoU@{label}: precision={metrics[f'{prefix}_iou_precision_{label}']:.4f}, "
                f"recall={metrics[f'{prefix}_iou_recall_{label}']:.4f}, "
                f"F1={metrics[f'{prefix}_iou_f1_{label}']:.4f}, "
                f"AP={metrics[f'{prefix}_iou_ap_{label}']:.4f}"
            )
    class_names = []
    for key in metrics:
        if key.startswith("class/"):
            parts = key.split("/", 2)
            if len(parts) >= 3 and parts[1] not in class_names:
                class_names.append(parts[1])
    for class_name in class_names:
        prefix = f"class/{class_name}/"
        lines.append(f"Class {class_name}:")
        lines.append(
            f"  Center@2m AP={metrics.get(prefix + 'center_ap_2m', 0.0):.4f}, "
            f"3D IoU@0.25 AP={metrics.get(prefix + '3d_iou_ap_0.25', 0.0):.4f}, "
            f"mean center distance={metrics.get(prefix + 'mean_best_center_distance', float('inf')):.4f} m"
        )
    return "\n".join(lines)
