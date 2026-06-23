from __future__ import annotations

from pathlib import Path

from .training import record_periodic_metrics


def build_periodic_eval_callback(
    *,
    style,
    eval_utils,
    cfg,
    args,
    test_loader,
    output_dir,
    logger,
    dist_train,
    project,
    dataset,
):
    periodic_root = Path(output_dir) / "periodic_eval"

    def evaluate(model, epoch):
        eval_model = model.module if dist_train and hasattr(model, "module") else model
        result_dir = periodic_root / f"epoch_{int(epoch):03d}"
        if style == "legacy":
            metrics = eval_utils.eval_one_epoch(
                cfg,
                eval_model,
                test_loader,
                epoch,
                logger,
                dist_test=dist_train,
                save_to_file=False,
                result_dir=result_dir,
            )
        else:
            metrics = eval_utils.eval_one_epoch(
                cfg,
                args,
                eval_model,
                test_loader,
                epoch,
                logger,
                dist_test=dist_train,
                result_dir=result_dir,
            )
        if cfg.LOCAL_RANK == 0:
            record_periodic_metrics(
                output_dir,
                epoch,
                metrics,
                project=project,
                dataset=dataset,
            )
        return metrics

    return evaluate
