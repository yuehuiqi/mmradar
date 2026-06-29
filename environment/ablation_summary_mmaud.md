# DG-PFA 消融实验汇总 — MMAUD

> 生成时间: 2026-06-30 00:01:21
> 输出目录: `E:\Scholar\mmradarDetect\PFA-NET\output\mmradar_models`

## 实验配置

| 变体 | VFE 模块 | 动态图 | 辅助注意力 |
|------|----------|--------|------------|
| PFA (baseline) | RadarPillarFeatureAttention | ✗ | ✗ |
| DG-PFA w/o Graph | DynamicGraphRadarPillarFeatureAttention | ✗ (NUM_LAYERS=0) | ✓ |
| DG-PFA w/o AuxAttn | DynamicGraphRadarPillarFeatureAttention | ✓ (NUM_LAYERS=2) | ✗ |
| DG-PFA (full) | DynamicGraphRadarPillarFeatureAttention | ✓ (NUM_LAYERS=2) | ✓ |

**训练参数:** 80 epochs, batch_size=4, workers=2, eval_interval=5

---

## 性能汇总

| 模型 | 最优轮次 | Center-AP @0.5m | Center-AP @1m | Center-AP @2m | Center-AP @4m | BEV-AP @0.25 | BEV-AP @0.5 | 3D-AP @0.25 | 3D-AP @0.5 | MBD↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PFA (baseline) | 15 | 0.0478 | 0.3265 | 0.9535 | 1.0000 | 0.9167 | 0.6387 | 0.6789 | 0.1600 | 1.0400 |
| DG-PFA w/o Graph | — | — | — | — | — | — | — | — | — | — |
| DG-PFA w/o AuxAttn | — | — | — | — | — | — | — | — | — | — |
| DG-PFA (full) | — | — | — | — | — | — | — | — | — | — |

---

## 逐模型详情

### PFA (baseline)

- **最优轮次:** epoch 15
- **结果文件:** `E:\Scholar\mmradarDetect\PFA-NET\output\mmradar_models\pfanet_mmaud_full\mmaud_all14_v1\periodic_eval\epoch_015\result.pkl`

#### 完整指标

| 指标 | 值 |
|------|----|
| center_ap_0.5m | 0.0478 |
| center_ap_1m | 0.3265 |
| center_ap_2m | 0.9535 |
| center_ap_4m | 1.0000 |
| bev_iou_ap_0.25 | 0.9167 |
| bev_iou_ap_0.5 | 0.6387 |
| bev_iou_ap_0.7 | 0.3466 |
| 3d_iou_ap_0.25 | 0.6789 |
| 3d_iou_ap_0.3 | 0.6077 |
| 3d_iou_ap_0.5 | 0.1600 |
| mean_best_center_distance | 1.0400 |

### DG-PFA w/o Graph

> ⚠️ 尚未完成或未找到结果: NOT FOUND: E:\Scholar\mmradarDetect\PFA-NET\output\mmradar_models\dg_pfanet_mmaud_ablation_nograph\mmaud_abl_v1

### DG-PFA w/o AuxAttn

> ⚠️ 尚未完成或未找到结果: NOT FOUND: E:\Scholar\mmradarDetect\PFA-NET\output\mmradar_models\dg_pfanet_mmaud_ablation_noaux\mmaud_abl_v1

### DG-PFA (full)

> ⚠️ 尚未完成或未找到结果: NOT FOUND: E:\Scholar\mmradarDetect\PFA-NET\output\mmradar_models\dg_pfanet_mmaud_full\mmaud_abl_v1

---

## result.pkl 路径（供融合实验使用）

- **PFA (baseline)** (epoch 15): `E:\Scholar\mmradarDetect\PFA-NET\output\mmradar_models\pfanet_mmaud_full\mmaud_all14_v1\periodic_eval\epoch_015\result.pkl`
- **DG-PFA w/o Graph** (epoch None): `NOT FOUND: E:\Scholar\mmradarDetect\PFA-NET\output\mmradar_models\dg_pfanet_mmaud_ablation_nograph\mmaud_abl_v1`
- **DG-PFA w/o AuxAttn** (epoch None): `NOT FOUND: E:\Scholar\mmradarDetect\PFA-NET\output\mmradar_models\dg_pfanet_mmaud_ablation_noaux\mmaud_abl_v1`
- **DG-PFA (full)** (epoch None): `NOT FOUND: E:\Scholar\mmradarDetect\PFA-NET\output\mmradar_models\dg_pfanet_mmaud_full\mmaud_abl_v1`