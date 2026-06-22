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

## 5. Smoke 训练结果

已完成 14 个 smoke 实验：7 个项目 × 2 个数据集。每个实验至少完成 1 epoch 训练；PCDet 系同时完成自动评估。

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

## 7. 正式训练命令

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

## 8. 后续建议

1. smoke 已验证通路正常；正式训练建议先每个模型跑 5 epoch 看 loss 曲线，再拉长到 80 epoch。
2. MMAUD 极稀疏，建议 batch size 不低于 2；如果显存允许可以尝试 4。
3. aiQiiDataset 点数更多，可以先 batch size 2 起步。
4. 当前 full 配置默认 `NUM_EPOCHS/total_epochs=80`，可按需要调低或调高。
5. 如果要记录新版本，建议先添加 `.gitignore` 忽略 `build/`、`*.so`、`__pycache__/`、`output/`、`work_dirs/` 后再提交配置与代码改动。
