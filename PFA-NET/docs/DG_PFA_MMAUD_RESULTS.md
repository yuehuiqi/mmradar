# DG-PFA MMAUD verified result

## Recommended artifact

- Baseline: fixed-seed original PFA, epoch 5.
- DG source: one low-learning-rate adaptation epoch.
- Final checkpoint: all 133 shared parameters restored from the baseline plus
  39 learned DG-only parameters.
- Inference setting: `MODEL.VFE.INFERENCE_FUSION_SCALE: 1.0`.
- DG-only state size: 64,820 values, about 1.34% over the 4,837,673-value
  baseline state.

This construction keeps the original PFA encoder, BEV backbone, and detection
head weights intact. Only the new graph/attention/bypass parameters differ.
Fusion scale `0.0` exactly reproduced the baseline metrics, providing an
end-to-end control for the sweep.

## Validation metrics (185 MMAUD frames)

| Metric | PFA control | DG-PFA | Delta |
|---|---:|---:|---:|
| Mean center distance (m, lower is better) | 1.1214 | **1.0996** | **-0.0218** |
| Center AP @ 0.5 m | 0.0424 | **0.0468** | **+0.0044** |
| Center AP @ 1 m | 0.2420 | **0.2582** | **+0.0162** |
| Center AP @ 2 m | 0.8804 | **0.8865** | **+0.0061** |
| BEV IoU AP @ 0.25 | **0.9395** | 0.9390 | -0.0005 |
| BEV IoU AP @ 0.3 | **0.9360** | 0.9354 | -0.0006 |
| BEV IoU AP @ 0.5 | 0.7213 | **0.7303** | **+0.0090** |
| 3D IoU AP @ 0.25 | 0.5896 | **0.6201** | **+0.0305** |
| 3D IoU AP @ 0.3 | 0.4406 | **0.4565** | **+0.0159** |
| 3D IoU AP @ 0.5 | 0.1092 | **0.1195** | **+0.0103** |

The result improves center localization and the practically useful medium/strict
3D IoU thresholds. The approximately 0.0005 reduction at low BEV thresholds is
small relative to the gains at BEV@0.5 and 3D thresholds.

## Reproduction

The retained baseline and stage-2 checkpoints used in this run are:

```text
output/mmradar_models/pfanet_mmaud_full/pfa_fixedseed_control/
  ckpt/checkpoint_epoch_5.pth

output/mmradar_models/dg_pfanet_mmaud_finetune/dg_pfa_stage2_v1/
  ckpt/checkpoint_epoch_1.pth
  ckpt/checkpoint_epoch_1_dg_only.pth
```

Merge shared baseline weights with DG-only learned weights:

```bash
python tools/merge_dg_pfa_checkpoint.py \
  --baseline output/mmradar_models/pfanet_mmaud_full/pfa_fixedseed_control/ckpt/checkpoint_epoch_5.pth \
  --adapted output/mmradar_models/dg_pfanet_mmaud_finetune/dg_pfa_stage2_v1/ckpt/checkpoint_epoch_1.pth \
  --output output/mmradar_models/dg_pfanet_mmaud_finetune/dg_pfa_stage2_v1/ckpt/checkpoint_epoch_1_dg_only.pth
```

Run the deterministic fusion sweep:

```bash
cd tools
python eval_dg_pfa_fusion_sweep.py \
  --cfg_file cfgs/mmradar_models/dg_pfanet_mmaud_finetune.yaml \
  --ckpt ../output/mmradar_models/dg_pfanet_mmaud_finetune/dg_pfa_stage2_v1/ckpt/checkpoint_epoch_1_dg_only.pth \
  --output_dir ../output/mmradar_models/dg_pfanet_mmaud_finetune/dg_pfa_stage2_v1/fusion_sweep
```

The full machine-readable sweep is in
`fusion_sweep/fusion_sweep_metrics.json`.

## Dataset-specific limitation

The current MMAUD conversion exposes XYZ plus a synthetic constant intensity
channel. It has no real Doppler or RCS values. Therefore this experiment uses
the DG-Pillar geometric branch and keeps the implemented motion/RCS branches
disabled. They can be enabled later by setting their raw feature indices on a
dataset that actually provides those measurements.
