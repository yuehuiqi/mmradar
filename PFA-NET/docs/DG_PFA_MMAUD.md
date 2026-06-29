# DG-PFA on MMAUD

This experiment adds the radar branch proposed in the thesis while preserving
the original implementation:

- `RadarPillarFeatureAttention`: unchanged PFA/RPFA baseline.
- `DynamicGraphRadarPillarFeatureAttention`: proposed opt-in encoder.
- `PillarVFE`: unchanged PointPillars baseline.

## Architecture

The proposed encoder contains four residual, identity-initialized additions:

1. masked mean pooling complements the original max-pooled pillar descriptor;
2. dynamic k-NN EdgeConv connects sparse occupied pillars inside each frame;
3. full-frame mean/max context supplies long-distance information;
4. separate motion, intensity, and structural attention branches generate a
   joint feature gate;
5. a learned per-pillar, per-channel bypass gate blends the enhanced descriptor
   with the untouched original RPFA max-pooled feature.

For MMAUD, graph and gate residuals are bounded to half strength. The optional
full-frame context and absolute-range prior are disabled because the provided
train/validation sequences have a pronounced elevation distribution shift;
keeping the graph local prevents the encoder from memorizing sequence-level
height shortcuts. Both capabilities remain available for later datasets.

All graph operations are isolated by `voxel_coords[:, 0]`, so nodes from
different samples in a batch can never become neighbors. Actual per-pillar mean
XYZ is used for the graph. This matters on MMAUD because the configured pillar
height spans the full Z range and therefore voxel-coordinate Z is constant.
Construction of the additional VFE modules also preserves and restores the
PyTorch RNG state, ensuring that a fixed-seed baseline and DG-PFA run initialize
the unchanged BEV backbone and detection head identically.

## MMAUD feature limitation

The current MMAUD conversion contains only XYZ. Its fourth `intensity` channel
is a synthetic constant `1.0`; Doppler and real RCS were not present in the
published folder-format point cloud. The MMAUD configuration therefore enables
the structural density/spread/range branch and sets both optional feature
indices to `-1`.

For a later dataset with real radar channels, set:

```yaml
MODEL:
  VFE:
    AUX_ATTENTION:
      MOTION_FEATURE_INDEX: 4
      INTENSITY_FEATURE_INDEX: 5
      MOTION_SCALE: 1.0
      INTENSITY_SCALE: 1.0
```

The indices refer to the raw point feature columns.

## Commands

From WSL:

```bash
cd /mnt/e/Scholar/mmradarDetect/PFA-NET/tools

# Functional smoke run
~/miniforge3/envs/PFANet/bin/python train.py \
  --cfg_file cfgs/mmradar_models/dg_pfanet_mmaud_smoke.yaml \
  --workers 0 --extra_tag dg_pfa_smoke --fix_random_seed

# Full run
~/miniforge3/envs/PFANet/bin/python train.py \
  --cfg_file cfgs/mmradar_models/dg_pfanet_mmaud_full.yaml \
  --workers 4 --extra_tag dg_pfa_mmaud_v1 --fix_random_seed \
  --eval_interval 5 --ckpt_save_interval 5

# Recommended stage-2 adaptation from a trained PFA checkpoint
~/miniforge3/envs/PFANet/bin/python train.py \
  --cfg_file cfgs/mmradar_models/dg_pfanet_mmaud_finetune.yaml \
  --pretrained_model \
    ../output/mmradar_models/pfanet_mmaud_full/<baseline-tag>/ckpt/checkpoint_epoch_5.pth \
  --workers 4 --extra_tag dg_pfa_finetune_v1 --fix_random_seed \
  --eval_interval 1 --ckpt_save_interval 1
```

Compare against the retained baseline with
`cfgs/mmradar_models/pfanet_mmaud_full.yaml`. For MMAUD, prioritize center AP,
mean center distance, and 3D IoU AP. The full DG-PFA configuration deliberately
inherits the original training schedule, detection head, loss weights, and data
pipeline unchanged, so the result is a clean encoder ablation.

For MMAUD, the recommended workflow is staged adaptation. Train or select the
retained PFA baseline first, load it with `--pretrained_model`, and then use the
five-epoch fine-tune configuration. The added branches are identity-initialized,
so stage 2 begins with exactly the baseline predictions and introduces graph
context gradually at one tenth of the original peak learning rate.

For a strict post-adaptation ablation, `tools/merge_dg_pfa_checkpoint.py` can
restore every shared parameter from the baseline while retaining only learned
DG-specific parameters from the adapted checkpoint. The
`INFERENCE_FUSION_SCALE` setting then scans from `0.0` (exact PFA bypass) to
`1.0` (full learned graph injection).

The verified MMAUD comparison and selected checkpoint are documented in
`docs/DG_PFA_MMAUD_RESULTS.md`.
