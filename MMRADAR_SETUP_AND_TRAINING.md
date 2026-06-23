# MMRadarDetect 环境配置与训练说明

更新时间：2026-06-22

## 1. 当前机器与基础环境

本次没有改动 Windows 侧已有 conda 环境。7 个项目均在 WSL `Ubuntu-24.04` 下使用 Miniforge 独立环境。

- GPU：NVIDIA GeForce RTX 5070 Ti Laptop GPU
- Driver：595.79
- 显存：12227 MiB
- Compute Capability：12.0
- PyTorch：2.7.1+cu128
- CUDA Runtime：12.8
- spconv：2.3.6
- NumPy：1.23.5
- Numba：0.57.1

## 2. 项目与虚拟环境对应

| 项目 | WSL 环境 | 说明 |
|---|---:|---|
| OpenPCDet | `PointPillar` | 避开已有 Windows `openpcdet` 环境名 |
| CenterPoint | `CenterPoint` | Det3D 系 |
| PillarNet-LTS | `PillarNetLTS` | Det3D 系 |
| DSVT | `DSVT` | 动态 pillar / DSVT |
| InterFusion | `InterFusion` | 老 PCDet 分支，已做 PyTorch/spconv 兼容 |
| PFA-NET | `PFANet` | 老 PCDet 分支，已做 PyTorch/spconv 兼容 |
| VoxelNeXt | `VoxelNeXt` | VoxelNeXt 稀疏卷积分支 |

环境 Python 路径格式：

```bash
/home/yuehui/miniforge3/envs/<EnvName>/bin/python
```

## 3. 数据集配置

两套毫米波雷达数据集已生成 OpenPCDet/Det3D 可读的 info 文件。

| 数据集 | 数据根目录 | train | val |
|---|---|---:|---:|
| aiQiiDataset | `/mnt/e/Scholar/dataset/aiQiiDataset/radar_openpcdet` | 13200 | 2560 |
| MMAUD | `/mnt/e/Scholar/dataset/mmaud/mmaud_radar_camera_openpcdet` | 3834 | 185 |

同时生成 smoke 子集：

- train smoke：64 samples
- val smoke：32 samples

MMAUD 原始点坐标按 camera 风格读取，配置中统一转换为 lidar 风格：

```yaml
COORDINATE_TRANSFORM: camera_to_lidar
```

通用点云范围已统一为：

```text
[-16, -20, -8, 64, 20, 8]
```

这样 BEV 网格尺寸是 `160 × 80`，可以被 PointPillar/DSVT/VoxelNeXt 主干网络整除。

## 4. 关键兼容改动

本次已完成：

- 7 个项目均可 import `spconv.pytorch`。
- 逐项目编译了仓库 CUDA 扩展；CenterPoint / PillarNet-LTS 的 Det3D ops 也已编译。
- 修复 PyTorch 2.6+ checkpoint `weights_only=True` 默认行为。
- 修复 Python 3.10 `collections.Iterable` 兼容问题。
- 修复老 PCDet 分支 `spconv` 1.x API、`VoxelGenerator` API。
- 修复稀疏毫米波点云单 voxel 时 `squeeze()` 把特征挤成 1D 的问题。
- 修复 DSVT 缺 `torch_scatter` 时的 PyTorch 原生 fallback。
- 修复 VoxelNeXt 稀疏 voxel 数少于 `MAX_OBJ_PER_SAMPLE` 时评估解码报错。
- MMAUD smoke/full PCDet 配置关闭几何增强，避免极稀疏点云被旋转出范围。
- 修复 Det3D 评估接口对 `{token: prediction}` 字典和 `results` 返回格式的兼容。
- 七个项目统一为周期验证：正式训练每 5 epoch 验证一次，最终 epoch 必定验证。
- 指标扩展为中心距离、BEV IoU、3D IoU 的 Precision、Recall、F1、AP，以及中心/尺寸/yaw 误差和置信度统计。
- checkpoint 只保留关键节点：80 epoch 默认保存 8 个正式权重，任何项目硬上限不超过 10 个。

## 5. Smoke 训练结果

已完成 14 个 smoke 实验：7 个项目 × 2 个数据集。每个实验至少完成 1 epoch 训练。当前版本中 PCDet 与 Det3D 两类项目都会在 smoke 最终 epoch 后自动评估；下表耗时是首次接入版本的历史记录。

| 实验 | 状态 | 用时 |
|---|---:|---:|
| OpenPCDet-aiQii | OK | 23.92s |
| OpenPCDet-MMAUD | OK | 24.08s |
| InterFusion-aiQii | OK | 20.59s |
| InterFusion-MMAUD | OK | 21.40s |
| PFA-NET-aiQii | OK | 21.93s |
| PFA-NET-MMAUD | OK | 18.56s |
| DSVT-aiQii | OK | 23.37s |
| DSVT-MMAUD | OK | 25.33s |
| VoxelNeXt-aiQii | OK | 32.77s |
| VoxelNeXt-MMAUD | OK | 22.43s |
| CenterPoint-aiQii | OK | 21.42s |
| CenterPoint-MMAUD | OK | 16.53s |
| PillarNet-LTS-aiQii | OK | 39.08s |
| PillarNet-LTS-MMAUD | OK | 33.60s |

日志目录：

```text
/mnt/e/Scholar/mmradarDetect/environment/smoke_logs
```

Windows 对应路径：

```text
E:\Scholar\mmradarDetect\environment\smoke_logs
```

说明：1 epoch smoke 的指标只用于确认环境、数据流、loss、保存与评估流程正常，不代表模型已收敛。

## 6. 一键 smoke 复测

在 PowerShell 中运行：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_smoke_train_wsl.py
```

只跑某几个实验：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_smoke_train_wsl.py --only OpenPCDet_aiqii VoxelNeXt_mmaud
```

## 7. 一键连续训练七个模型

推荐通过统一脚本按数据集连续训练。脚本会依次运行：

```text
OpenPCDet → InterFusion → PFA-NET → DSVT → VoxelNeXt → CenterPoint → PillarNet-LTS
```

一键运行七个模型的 MMAUD 正式训练：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_full_suite_wsl.py `
  --dataset mmaud --run-tag mmaud_v1 --workers 4 --retries 2
```

一键运行七个模型的 aiQiiDataset 正式训练：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_full_suite_wsl.py `
  --dataset aiqii --run-tag aiqii_v1 --workers 4 --retries 2
```

参数说明：

- `--run-tag`：本批实验的唯一名称。恢复中断时必须使用原来的 tag。
- `--retries 2`：首次失败后最多自动重试 2 次；PCDet 自动读取最新 checkpoint，Det3D 自动添加最新 `epoch_*.pth` 继续训练。
- 同一模型多次失败后，默认记录失败并继续下一个模型。需要失败后立刻停止时添加 `--stop-on-failure`。
- 只运行部分模型时使用 `--only OpenPCDet DSVT CenterPoint`。可用名称为 `OpenPCDet`、`InterFusion`、`PFANet`、`DSVT`、`VoxelNeXt`、`CenterPoint`、`PillarNetLTS`。
- 使用 `--gpu 0` 指定 GPU，默认为 0。
- 正式开始前可添加 `--dry-run`，只检查七个命令、环境和输出路径，不启动训练。

如果终端、WSL 或电脑意外中断，重新执行完全相同的命令即可。脚本会：

1. 跳过已经有 epoch 80 最终指标的模型；
2. 从未完成模型的最近 checkpoint 恢复；
3. 继续后续模型；
4. 每个模型失败时保留独立 attempt 日志。

批量运行状态和汇总输出位于：

```text
E:\Scholar\mmradarDetect\environment\full_runs\<dataset>\<run-tag>\
├── suite_status.json
├── ALL_MODELS_METRICS.md
├── ALL_MODELS_METRICS.json
└── <Model>_attempt_<N>.log
```

`ALL_MODELS_METRICS.md` 是七模型最终关键指标对比表；`ALL_MODELS_METRICS.json` 在一个文件中保存七个模型的完整最终指标与每 5 epoch 的全部历史指标。每个模型自己的阶段指标也会位于其输出目录：

```text
periodic_metrics/
├── epoch_005.json
├── epoch_010.json
├── ...
├── epoch_080.json
└── metrics_history.json
```

每 5 epoch 会写出以下指标：

- Center@0.5/1/2/4m：TP、FP、FN、Precision、Recall、F1、AP；
- BEV IoU@0.1/0.25/0.3/0.5/0.7：Precision、Recall、F1、AP；
- 3D IoU@0.1/0.25/0.3/0.5/0.7：Precision、Recall、F1、AP；
- GT/预测数量、无预测帧数、置信度分布；
- 平均/RMSE/中位数/P90/P95/最大中心距离；
- x/y/z 中心误差、dx/dy/dz 尺寸误差、yaw 误差。

AP 基于模型配置中的 score threshold 和 NMS 后预测，比较不同模型或论文时必须保持数据划分、阈值和评价协议一致。

### 周期验证与 checkpoint 策略

- 正式训练默认 80 epoch，在 5、10、15……80 epoch 后验证。
- PCDet 系默认 `--ckpt_save_interval 10 --eval_interval 5 --max_ckpt_save_num 9`。80 epoch 保存 10、20……80 共 8 个正式 checkpoint；带训练中断用临时权重时也不超过 9 个。
- CenterPoint/PillarNet-LTS 使用 `workflow=[('train', 5), ('val', 1)]`，checkpoint 间隔为 10；80 epoch 保存 8 个，硬上限为 10 个。
- smoke 配置在最终第 1 epoch 后同样执行一次验证。
- 最终 epoch 即使不能被间隔整除，也会强制保存和验证。

## 8. 单模型正式训练命令

### OpenPCDet / InterFusion / PFA-NET

```bash
cd /mnt/e/Scholar/mmradarDetect/OpenPCDet/tools
/home/yuehui/miniforge3/envs/PointPillar/bin/python train.py \
  --cfg_file cfgs/mmradar_models/pointpillar_aiqii_full.yaml \
  --workers 4 --batch_size 2 --extra_tag full_aiqii
```

把项目和环境替换为：

| 项目 | env | aiQii full config | MMAUD full config |
|---|---|---|---|
| OpenPCDet | PointPillar | `cfgs/mmradar_models/pointpillar_aiqii_full.yaml` | `cfgs/mmradar_models/pointpillar_mmaud_full.yaml` |
| InterFusion | InterFusion | `cfgs/mmradar_models/pointpillar_aiqii_full.yaml` | `cfgs/mmradar_models/pointpillar_mmaud_full.yaml` |
| PFA-NET | PFANet | `cfgs/mmradar_models/pointpillar_aiqii_full.yaml` | `cfgs/mmradar_models/pointpillar_mmaud_full.yaml` |

### DSVT

```bash
cd /mnt/e/Scholar/mmradarDetect/DSVT/tools
/home/yuehui/miniforge3/envs/DSVT/bin/python train.py \
  --cfg_file cfgs/mmradar_models/dsvt_aiqii_full.yaml \
  --workers 4 --batch_size 1 --extra_tag full_aiqii
```

MMAUD 建议先用 batch size 2：

```bash
/home/yuehui/miniforge3/envs/DSVT/bin/python train.py \
  --cfg_file cfgs/mmradar_models/dsvt_mmaud_full.yaml \
  --workers 4 --batch_size 2 --extra_tag full_mmaud
```

### VoxelNeXt

```bash
cd /mnt/e/Scholar/mmradarDetect/VoxelNeXt/tools
/home/yuehui/miniforge3/envs/VoxelNeXt/bin/python train.py \
  --cfg_file cfgs/mmradar_models/voxelnext_aiqii_full.yaml \
  --workers 4 --batch_size 1 --extra_tag full_aiqii
```

MMAUD 建议先用 batch size 2：

```bash
/home/yuehui/miniforge3/envs/VoxelNeXt/bin/python train.py \
  --cfg_file cfgs/mmradar_models/voxelnext_mmaud_full.yaml \
  --workers 4 --batch_size 2 --extra_tag full_mmaud
```

### CenterPoint

```bash
cd /mnt/e/Scholar/mmradarDetect/CenterPoint
PYTHONPATH=/mnt/e/Scholar/mmradarDetect/CenterPoint \
/home/yuehui/miniforge3/envs/CenterPoint/bin/python tools/train.py \
  configs/mmradar/centerpoint_aiqii_full.py \
  --work_dir work_dirs/mmradar_centerpoint_aiqii_full --gpus 1
```

MMAUD：

```bash
PYTHONPATH=/mnt/e/Scholar/mmradarDetect/CenterPoint \
/home/yuehui/miniforge3/envs/CenterPoint/bin/python tools/train.py \
  configs/mmradar/centerpoint_mmaud_full.py \
  --work_dir work_dirs/mmradar_centerpoint_mmaud_full --gpus 1
```

### PillarNet-LTS

```bash
cd /mnt/e/Scholar/mmradarDetect/PillarNet-LTS
PYTHONPATH=/mnt/e/Scholar/mmradarDetect/PillarNet-LTS \
/home/yuehui/miniforge3/envs/PillarNetLTS/bin/python tools/train.py \
  configs/mmradar/pillarnet_aiqii_full.py \
  --work_dir work_dirs/mmradar_pillarnet_aiqii_full --gpus 1
```

MMAUD：

```bash
PYTHONPATH=/mnt/e/Scholar/mmradarDetect/PillarNet-LTS \
/home/yuehui/miniforge3/envs/PillarNetLTS/bin/python tools/train.py \
  configs/mmradar/pillarnet_mmaud_full.py \
  --work_dir work_dirs/mmradar_pillarnet_mmaud_full --gpus 1
```

上述 PCDet 命令现在默认每 5 epoch 验证、每 10 epoch 保存关键 checkpoint，无需再额外传验证参数。CenterPoint/PillarNet-LTS 的 full 配置也已内置五轮一次的 train/val 工作流。

## 9. 后续建议

1. smoke 已验证通路正常；正式训练时可直接观察 `periodic_metrics/metrics_history.json` 的五轮指标曲线。
2. MMAUD 极稀疏，建议 batch size 不低于 2；如果显存允许可以尝试 4。
3. aiQiiDataset 点数更多，可以先 batch size 2 起步。
4. 当前 full 配置默认 `NUM_EPOCHS/total_epochs=80`，可按需要调低或调高。
5. `output/`、`work_dirs/`、批量日志、模型权重和数据文件不进入 Git；提交源码前只提交配置、脚本、公共指标代码和文档。
6. 已经训练完成但没有周期验证的旧实验不会自动补出历史指标；可以单独评估其最终 checkpoint，或者用新 run tag 重新训练。
