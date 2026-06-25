# MMRadarDetect 环境配置与训练说明

更新时间：2026-06-24

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

| 项目          |         WSL 环境 | 说明                                    |
| ------------- | ---------------: | --------------------------------------- |
| OpenPCDet     |  `PointPillar` | 避开已有 Windows`openpcdet` 环境名    |
| CenterPoint   |  `CenterPoint` | Det3D 系                                |
| PillarNet-LTS | `PillarNetLTS` | Det3D 系                                |
| DSVT          |         `DSVT` | 动态 pillar / DSVT                      |
| InterFusion   |  `InterFusion` | 老 PCDet 分支，已做 PyTorch/spconv 兼容 |
| PFA-NET       |       `PFANet` | 老 PCDet 分支，已做 PyTorch/spconv 兼容 |
| VoxelNeXt     |    `VoxelNeXt` | VoxelNeXt 稀疏卷积分支                  |

环境 Python 路径格式：

```bash
/home/yuehui/miniforge3/envs/<EnvName>/bin/python
```

## 3. 数据集配置

两套毫米波雷达数据集已生成 OpenPCDet/Det3D 可读的 info 文件。

| 数据集       | 数据根目录                                                    | train |  val |
| ------------ | ------------------------------------------------------------- | ----: | ---: |
| aiQiiDataset | `/mnt/e/Scholar/dataset/aiQiiDataset/radar_openpcdet`       | 11238 | 2560 |
| MMAUD        | `/mnt/e/Scholar/dataset/mmaud/mmaud_radar_camera_openpcdet` |  3834 |  185 |

aiQiiDataset 的原始 train info 中有 13200 帧。2026-06-23 检查正式训练失败原因时，发现其中一部分帧过于稀疏，增强和范围过滤后会在部分模型里变成空 voxel / 空 sparse feature；另外少数帧存在同一无人机框重复标注。当前版本对 aiQii 做了训练集侧的轻量清洗：

- train：从 13200 帧过滤为 11238 帧；
- val：保持 2560 帧不动，避免验证集被人为筛选；
- 过滤规则：训练帧至少需要 16 个范围内点、16 个 BEV 占用格、16 个 3D voxel；
- 重复框处理：aiQii 是单无人机场景，多框样本按中位数框合并为一个 `Drone` 框，共移除 11 个重复/近重复框；
- 原始 pkl 已备份到 `E:\Scholar\dataset\aiQiiDataset\radar_openpcdet\backup_before_aiqii_repair_20260623`。

aiQii 现在同时保留两套训练索引：

| 模式   | 类别                                                  | PCDet info                                                                    | Det3D/RadarNeXt info                                                                      |
| ------ | ----------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| 单类   | `Drone`                                             | `mmradar_infos_train.pkl` / `mmradar_infos_val.pkl`                       | `mmradar_det3d_infos_train.pkl` / `mmradar_det3d_infos_val.pkl`                       |
| 四分类 | `Air3S`、`Mini4pro`、`Mavic3Pro`、`jingling4` | `mmradar_infos_train_multiclass.pkl` / `mmradar_infos_val_multiclass.pkl` | `mmradar_det3d_infos_train_multiclass.pkl` / `mmradar_det3d_infos_val_multiclass.pkl` |

四分类版本沿用同一套稀疏帧过滤和重复框合并规则。当前四分类 train/val 目标分布为：

| split | Air3S | Mini4pro | Mavic3Pro | jingling4 |
| ----- | ----: | -------: | --------: | --------: |
| train |  3644 |     1523 |      2522 |      3549 |
| val   |   870 |      263 |       507 |       920 |

如需重新生成数据索引，在 PowerShell 中运行：

```powershell
wsl.exe -d Ubuntu-24.04 -- /home/yuehui/miniforge3/envs/PointPillar/bin/python /mnt/e/Scholar/mmradarDetect/environment/prepare_mmradar_datasets.py
```

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
- 修复 aiQii 极稀疏帧导致 PCDet scatter 用 `coords[:, 0].max()+1` 推断 batch size 错误的问题；现在优先使用 dataloader 写入的 `batch_dict['batch_size']`。
- 修复 VoxelNeXt sparse target assign / regression loss 在某个 batch 样本没有有效 sparse 输出时仍索引空张量的问题。
- aiQii 的 DSVT / VoxelNeXt 正式训练 batch size 调整为 2，避免 BatchNorm 在极端小样本上只看到单个特征。
- 批量运行脚本对子进程使用干净 Linux `PATH`，避免 WSL 继承 Windows PATH 后让 `gpustat` 或 Python entrypoint 找错环境。
- MMAUD smoke/full PCDet 配置关闭几何增强，避免极稀疏点云被旋转出范围。
- 修复 Det3D 评估接口对 `{token: prediction}` 字典和 `results` 返回格式的兼容。
- 七个项目统一为周期验证：正式训练每 5 epoch 验证一次，最终 epoch 必定验证。
- 指标扩展为中心距离、BEV IoU、3D IoU 的 Precision、Recall、F1、AP，以及中心/尺寸/yaw 误差和置信度统计。
- checkpoint 只保留关键节点：80 epoch 默认保存 8 个正式权重，任何项目硬上限不超过 10 个。

## 5. Smoke 训练结果

已完成 14 个 smoke 实验：7 个项目 × 2 个数据集。每个实验至少完成 1 epoch 训练。当前版本中 PCDet 与 Det3D 两类项目都会在 smoke 最终 epoch 后自动评估；下表耗时是首次接入版本的历史记录。

2026-06-23 修复 aiQii 数据和稀疏帧兼容后，又对 aiQii 重新做了验证：7 个项目的 aiQii smoke 均通过；OpenPCDet、DSVT、VoxelNeXt 还额外跑了全量 train + 全量 val 的 1 epoch 检查，均能正常训练并输出指标。

| 实验                | 状态 |   用时 |
| ------------------- | ---: | -----: |
| OpenPCDet-aiQii     |   OK | 23.92s |
| OpenPCDet-MMAUD     |   OK | 24.08s |
| InterFusion-aiQii   |   OK | 20.59s |
| InterFusion-MMAUD   |   OK | 21.40s |
| PFA-NET-aiQii       |   OK | 21.93s |
| PFA-NET-MMAUD       |   OK | 18.56s |
| DSVT-aiQii          |   OK | 23.37s |
| DSVT-MMAUD          |   OK | 25.33s |
| VoxelNeXt-aiQii     |   OK | 32.77s |
| VoxelNeXt-MMAUD     |   OK | 22.43s |
| CenterPoint-aiQii   |   OK | 21.42s |
| CenterPoint-MMAUD   |   OK | 16.53s |
| PillarNet-LTS-aiQii |   OK | 39.08s |
| PillarNet-LTS-MMAUD |   OK | 33.60s |

日志目录：

```text
/mnt/e/Scholar/mmradarDetect/environment/smoke_logs
```

Windows 对应路径：

```text
E:\Scholar\mmradarDetect\environment\smoke_logs
```

说明：1 epoch smoke 的指标只用于确认环境、数据流、loss、保存与评估流程正常，不代表模型已收敛。

## 6. 三个一键脚本的区别与 batch size

现在有三个批量脚本，定位不一样：

| 脚本                                              | 主要用途                          | 默认输出目录                   | 适合什么时候用                                                                        |
| ------------------------------------------------- | --------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------- |
| `environment/run_mmradar_smoke_train_wsl.py`    | 原始 7 项目的 1 epoch smoke 检查  | `environment/smoke_logs/`    | 改环境、改配置、改数据后，先快速确认能不能训练和验证                                  |
| `environment/run_mmradar_full_suite_wsl.py`     | 原始 7 项目的正式连续训练         | `environment/full_runs/`     | 正式复现 OpenPCDet、InterFusion、PFA-NET、DSVT、VoxelNeXt、CenterPoint、PillarNet-LTS |
| `environment/run_mmradar_extended_suite_wsl.py` | 扩展 7 模型的 smoke/full 连续训练 | `environment/extended_runs/` | 跑 PointRCNN、Part-A2、PV-RCNN、Voxel R-CNN、PV-RCNN++、RadarPillar、RadarNeXt        |

batch size 不需要再改配置文件，直接在命令里加参数：

| 参数                         | 作用                                                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `--batch-size 2`           | 覆盖当前脚本里所有模型的训练 batch size                                                                                         |
| `--pcdet-batch-size 2`     | 只覆盖 OpenPCDet 风格模型，包括 OpenPCDet/InterFusion/PFA-NET/DSVT/VoxelNeXt，以及扩展脚本里的 PointRCNN/PV-RCNN/RadarPillar 等 |
| `--det3d-batch-size 1`     | 只覆盖`run_mmradar_full_suite_wsl.py` / smoke 脚本中的 CenterPoint、PillarNet-LTS                                             |
| `--radarnext-batch-size 1` | 只覆盖扩展脚本中的 RadarNeXt                                                                                                    |

如果不传这些参数，就使用脚本和配置里的默认值。Det3D 类项目本身没有命令行 batch 参数，脚本会在运行目录里自动生成一个临时 config 来覆盖 `data.samples_per_gpu`，不需要手动维护。

smoke 复测示例：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_smoke_train_wsl.py
```

只看命令、不启动训练：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_smoke_train_wsl.py `
  --aiqii-classes multiclass --batch-size 2 --only OpenPCDet_aiqii CenterPoint_aiqii --dry-run
```

## 7. 一键连续训练命令

### 原始 7 项目：单类 aiQii，也就是四种无人机统一为 `Drone`

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_full_suite_wsl.py `
  --dataset aiqii --aiqii-classes single --run-tag aiqii_single_v1 `
  --workers 4 --batch-size 4 --retries 2
```

### 原始 7 项目：四分类 aiQii，分别检测 `Air3S`、`Mini4pro`、`Mavic3Pro`、`jingling4`

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_full_suite_wsl.py `
  --dataset aiqii --aiqii-classes multiclass --run-tag aiqii_multiclass_v1 `
  --workers 4 --batch-size 4 --retries 2
```

### 原始 7 项目：MMAUD，保持单类 `Drone` 不变

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_full_suite_wsl.py `
  --dataset mmaud --run-tag mmaud_v1 `
  --workers 4 --batch-size 4 --retries 2
```

### 扩展 7 模型：单类 aiQii

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_extended_suite_wsl.py `
  --dataset aiqii --aiqii-classes single --run-tag aiqii_ext_single_v1 `
  --mode full --workers 4 --batch-size 4 --radarnext-batch-size 4 --retries 2
```

### 扩展 7 模型：四分类 aiQii

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_extended_suite_wsl.py `
  --dataset aiqii --aiqii-classes multiclass --run-tag aiqii_ext_multiclass_v1 `
  --mode full --workers 4 --batch-size 4 --radarnext-batch-size 4 --retries 2
```

### 扩展 7 模型：MMAUD，保持单类 `Drone` 不变

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_extended_suite_wsl.py `
  --dataset mmaud --run-tag mmaud_ext_v1 `
  --mode full --workers 4 --batch-size 4 --radarnext-batch-size 4 --retries 2
```

参数说明：

- `--aiqii-classes single`：aiQii 四种无人机统一训练成一个 `Drone` 类，这是当前已训练结果使用的模式。
- `--aiqii-classes multiclass`：aiQii 保留四种无人机类别，指标 JSON 中会额外写入 `class/Air3S/...`、`class/Mini4pro/...` 等分类别指标。
- `--run-tag`：本批实验的唯一名称。恢复中断时必须使用原来的 tag。
- `--retries 2`：首次失败后最多自动重试 2 次。
- `--only`：只运行指定模型。例如原始脚本可用 `--only OpenPCDet DSVT CenterPoint`；扩展脚本可用 `--only PointRCNN RadarPillar RadarNeXt`。
- `--gpu 0`：指定 GPU，默认 0。
- `--dry-run`：只打印命令、环境和输出路径，不启动训练。

如果终端、WSL 或电脑意外中断，重新执行完全相同的命令即可。脚本会跳过已经完成的模型，未完成的模型会尽量从最近 checkpoint 继续。

输出目录：

```text
E:\Scholar\mmradarDetect\environment\full_runs\aiqii\<run-tag>\
E:\Scholar\mmradarDetect\environment\full_runs\aiqii_multiclass\<run-tag>\
E:\Scholar\mmradarDetect\environment\full_runs\mmaud\<run-tag>\

E:\Scholar\mmradarDetect\environment\extended_runs\aiqii\<mode>\<run-tag>\
E:\Scholar\mmradarDetect\environment\extended_runs\aiqii_multiclass\<mode>\<run-tag>\
E:\Scholar\mmradarDetect\environment\extended_runs\mmaud\<mode>\<run-tag>\
```

每 5 epoch 会写出中心距离、BEV IoU、3D IoU 的 Precision/Recall/F1/AP，以及中心/尺寸/yaw 误差。正式训练默认 80 epoch，checkpoint 默认不超过 10 个。

## 8. 单模型正式训练命令

下面命令都按 full 训练写，默认 80 epoch、每 5 epoch 验证、每 10 epoch 保存关键 checkpoint。`--extra_tag` / `--work_dir` 建议自己换成有辨识度的名字。

### 8.1 原始 7 项目：PCDet 风格模型

通用模板：

```bash
cd <PROJECT_ROOT>/tools
PYTHONPATH=/mnt/e/Scholar/mmradarDetect:<PROJECT_ROOT> \
<PYTHON> train.py \
  --cfg_file <CONFIG> \
  --workers 4 --batch_size 4 --extra_tag <RUN_TAG> \
  --ckpt_save_interval 10 --max_ckpt_save_num 9 --eval_interval 5 --max_waiting_mins 0
```

| 模型                    | PROJECT_ROOT                                 | PYTHON                                                  | aiQii 单类 CONFIG                                   | aiQii 四分类 CONFIG                                            | MMAUD CONFIG                                        |
| ----------------------- | -------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------- |
| OpenPCDet / PointPillar | `/mnt/e/Scholar/mmradarDetect/OpenPCDet`   | `/home/yuehui/miniforge3/envs/PointPillar/bin/python` | `cfgs/mmradar_models/pointpillar_aiqii_full.yaml` | `cfgs/mmradar_models/pointpillar_aiqii_multiclass_full.yaml` | `cfgs/mmradar_models/pointpillar_mmaud_full.yaml` |
| InterFusion             | `/mnt/e/Scholar/mmradarDetect/InterFusion` | `/home/yuehui/miniforge3/envs/InterFusion/bin/python` | `cfgs/mmradar_models/pointpillar_aiqii_full.yaml` | `cfgs/mmradar_models/pointpillar_aiqii_multiclass_full.yaml` | `cfgs/mmradar_models/pointpillar_mmaud_full.yaml` |
| PFA-NET                 | `/mnt/e/Scholar/mmradarDetect/PFA-NET`     | `/home/yuehui/miniforge3/envs/PFANet/bin/python`      | `cfgs/mmradar_models/pointpillar_aiqii_full.yaml` | `cfgs/mmradar_models/pointpillar_aiqii_multiclass_full.yaml` | `cfgs/mmradar_models/pointpillar_mmaud_full.yaml` |
| DSVT                    | `/mnt/e/Scholar/mmradarDetect/DSVT`        | `/home/yuehui/miniforge3/envs/DSVT/bin/python`        | `cfgs/mmradar_models/dsvt_aiqii_full.yaml`        | `cfgs/mmradar_models/dsvt_aiqii_multiclass_full.yaml`        | `cfgs/mmradar_models/dsvt_mmaud_full.yaml`        |
| VoxelNeXt               | `/mnt/e/Scholar/mmradarDetect/VoxelNeXt`   | `/home/yuehui/miniforge3/envs/VoxelNeXt/bin/python`   | `cfgs/mmradar_models/voxelnext_aiqii_full.yaml`   | `cfgs/mmradar_models/voxelnext_aiqii_multiclass_full.yaml`   | `cfgs/mmradar_models/voxelnext_mmaud_full.yaml`   |

### 8.2 原始 7 项目：Det3D 风格模型

CenterPoint：

```bash
cd /mnt/e/Scholar/mmradarDetect/CenterPoint
PYTHONPATH=/mnt/e/Scholar/mmradarDetect:/mnt/e/Scholar/mmradarDetect/CenterPoint \
/home/yuehui/miniforge3/envs/CenterPoint/bin/python tools/train.py \
  configs/mmradar/centerpoint_aiqii_full.py \
  --work_dir work_dirs/mmradar_centerpoint_aiqii_single_v1 --gpus 1
```

CenterPoint 四分类把 config 换成：

```text
configs/mmradar/centerpoint_aiqii_multiclass_full.py
```

CenterPoint MMAUD 把 config 换成：

```text
configs/mmradar/centerpoint_mmaud_full.py
```

PillarNet-LTS：

```bash
cd /mnt/e/Scholar/mmradarDetect/PillarNet-LTS
PYTHONPATH=/mnt/e/Scholar/mmradarDetect:/mnt/e/Scholar/mmradarDetect/PillarNet-LTS \
/home/yuehui/miniforge3/envs/PillarNetLTS/bin/python tools/train.py \
  configs/mmradar/pillarnet_aiqii_full.py \
  --work_dir work_dirs/mmradar_pillarnet_aiqii_single_v1 --gpus 1
```

PillarNet-LTS 四分类把 config 换成：

```text
configs/mmradar/pillarnet_aiqii_multiclass_full.py
```

PillarNet-LTS MMAUD 把 config 换成：

```text
configs/mmradar/pillarnet_mmaud_full.py
```

### 8.3 扩展 OpenPCDet 模型

这些模型都在 OpenPCDet 项目里跑，共用 `PointPillar` 环境：

```bash
cd /mnt/e/Scholar/mmradarDetect/OpenPCDet/tools
PYTHONPATH=/mnt/e/Scholar/mmradarDetect:/mnt/e/Scholar/mmradarDetect/OpenPCDet \
/home/yuehui/miniforge3/envs/PointPillar/bin/python train.py \
  --cfg_file <CONFIG> \
  --workers 4 --batch_size 4 --extra_tag <RUN_TAG> \
  --ckpt_save_interval 10 --max_ckpt_save_num 9 --eval_interval 5 --max_waiting_mins 0
```

| 模型        | aiQii 单类 CONFIG                                      | aiQii 四分类 CONFIG                                               | MMAUD CONFIG                                           |
| ----------- | ------------------------------------------------------ | ----------------------------------------------------------------- | ------------------------------------------------------ |
| PointRCNN   | `cfgs/mmradar_models/pointrcnn_aiqii_full.yaml`      | `cfgs/mmradar_models/pointrcnn_aiqii_multiclass_full.yaml`      | `cfgs/mmradar_models/pointrcnn_mmaud_full.yaml`      |
| Part-A2     | `cfgs/mmradar_models/parta2_aiqii_full.yaml`         | `cfgs/mmradar_models/parta2_aiqii_multiclass_full.yaml`         | `cfgs/mmradar_models/parta2_mmaud_full.yaml`         |
| PV-RCNN     | `cfgs/mmradar_models/pvrcnn_aiqii_full.yaml`         | `cfgs/mmradar_models/pvrcnn_aiqii_multiclass_full.yaml`         | `cfgs/mmradar_models/pvrcnn_mmaud_full.yaml`         |
| Voxel R-CNN | `cfgs/mmradar_models/voxelrcnn_aiqii_full.yaml`      | `cfgs/mmradar_models/voxelrcnn_aiqii_multiclass_full.yaml`      | `cfgs/mmradar_models/voxelrcnn_mmaud_full.yaml`      |
| PV-RCNN++   | `cfgs/mmradar_models/pvrcnnplusplus_aiqii_full.yaml` | `cfgs/mmradar_models/pvrcnnplusplus_aiqii_multiclass_full.yaml` | `cfgs/mmradar_models/pvrcnnplusplus_mmaud_full.yaml` |

### 8.4 RadarPillar

```bash
cd /mnt/e/Scholar/mmradarDetect/RadarPillar
LD_LIBRARY_PATH=/home/yuehui/miniforge3/envs/RadarPillar/lib/python3.10/site-packages/torch/lib:/home/yuehui/miniforge3/envs/RadarPillar/targets/x86_64-linux/lib:/home/yuehui/miniforge3/envs/RadarPillar/lib \
PYTHONPATH=/mnt/e/Scholar/mmradarDetect:/mnt/e/Scholar/mmradarDetect/RadarPillar \
/home/yuehui/miniforge3/envs/RadarPillar/bin/python tools/train.py \
  --cfg_file tools/cfgs/mmradar_models/radarpillar_aiqii_full.yaml \
  --workers 4 --batch_size 4 --extra_tag radarpillar_aiqii_single_v1 \
  --ckpt_save_interval 10 --max_ckpt_save_num 9 --max_waiting_mins 0
```

四分类把 config 换成：

```text
tools/cfgs/mmradar_models/radarpillar_aiqii_multiclass_full.yaml
```

MMAUD 把 config 换成：

```text
tools/cfgs/mmradar_models/radarpillar_mmaud_full.yaml
```

### 8.5 RadarNeXt

```bash
cd /mnt/e/Scholar/mmradarDetect/RadarNeXt
PYTHONPATH=/mnt/e/Scholar/mmradarDetect:/mnt/e/Scholar/mmradarDetect/RadarNeXt \
/home/yuehui/miniforge3/envs/RadarNeXt/bin/python tools/train.py \
  configs/mmradar/radarnext_centerpoint_aiqii_full.py \
  --work-dir work_dirs/radarnext_aiqii_single_v1 \
  --cfg-options train_dataloader.batch_size=1 train_dataloader.num_workers=4 val_dataloader.num_workers=4
```

四分类把 config 换成：

```text
configs/mmradar/radarnext_centerpoint_aiqii_multiclass_full.py
```

MMAUD 把 config 换成：

```text
configs/mmradar/radarnext_centerpoint_mmaud_full.py
```

## 9. 后续建议

1. smoke 已验证通路正常；正式训练时可直接观察 `periodic_metrics/metrics_history.json` 的五轮指标曲线。
2. MMAUD 极稀疏，建议 batch size 不低于 2；如果显存允许可以尝试 4。
3. aiQiiDataset 已按稀疏帧规则清洗训练集；当前一键脚本对所有 PCDet 系模型使用 batch size 2 起步。
4. 当前 full 配置默认 `NUM_EPOCHS/total_epochs=80`，可按需要调低或调高。
5. `output/`、`work_dirs/`、批量日志、模型权重和数据文件不进入 Git；提交源码前只提交配置、脚本、公共指标代码和文档。
6. 已经训练完成但没有周期验证的旧实验不会自动补出历史指标；可以单独评估其最终 checkpoint，或者用新 run tag 重新训练。

## 10. 2026-06-24 扩展模型复现补充

这次新加入了 `RadarPillar`、`SD4R`、`RadarNeXt` 三个项目，同时检查了 `OpenPCDet` 自带的 PointRCNN、Part-A2、PV-RCNN、Voxel R-CNN、PV-RCNN++、MPPNet。结论如下：

| 模型/项目   |        当前处理结论 | 环境            | 说明                                                                                                                      |
| ----------- | ------------------: | --------------- | ------------------------------------------------------------------------------------------------------------------------- |
| PointRCNN   | 已接入并 smoke 通过 | `PointPillar` | OpenPCDet 系列，共用现有环境                                                                                              |
| Part-A2     | 已接入并 smoke 通过 | `PointPillar` | 需要把 BEV 压缩通道改为 128                                                                                               |
| PV-RCNN     | 已接入并 smoke 通过 | `PointPillar` | PFE 特征源收窄为`['bev']`，适配当前毫米波稀疏输入                                                                       |
| Voxel R-CNN | 已接入并 smoke 通过 | `PointPillar` | 需要把 BEV 压缩通道改为 128                                                                                               |
| PV-RCNN++   | 已接入并 smoke 通过 | `PointPillar` | CenterHead 的`MAX_OBJ_PER_SAMPLE` 从 500 调小到 100                                                                     |
| MPPNet      |            暂不纳入 | `PointPillar` | 当前公开配置主要面向 Waymo 多帧序列/记忆库，现有 aiQii/MMAUD info 是单帧雷达格式，缺少序列元数据                          |
| RadarPillar | 已接入并 smoke 通过 | `RadarPillar` | 独立环境，已编译 CUDA 扩展                                                                                                |
| RadarNeXt   | 已接入并 smoke 通过 | `RadarNeXt`   | 独立环境，MMEngine/MMCV 栈已适配                                                                                          |
| SD4R        |            暂不纳入 | `SD4R`        | 原项目依赖 PyTorch 1.9.1/cu111、mmcv-full 1.4.0、旧版 mmdet/numba 等，和 RTX 5070 Ti + CUDA 12.8 + Python 3.10 栈冲突过大 |

新下载的三个项目已经清理掉内层版本管理信息：`RadarPillar`、`SD4R`、`RadarNeXt` 内部不再保留独立 `.git` / `.github` / `.gitattributes` 等仓库痕迹，后续统一由 `E:\Scholar\mmradarDetect` 根仓库管理。

### 新增环境

| 环境            | Python | Torch/CUDA             | 主要用途                               |
| --------------- | -----: | ---------------------- | -------------------------------------- |
| `RadarPillar` |   3.10 | `torch==2.7.1+cu128` | RadarPillar 训练与自带 CUDA ops        |
| `RadarNeXt`   |   3.10 | `torch==2.7.1+cu128` | RadarNeXt / MMEngine / MMCV            |
| `SD4R`        |   3.10 | `torch==2.7.1+cu128` | 已创建基础环境，但项目本身暂不纳入复现 |

RTX 50 系列在 WSL 下编译扩展时，重点是让编译器使用 conda 环境里的 CUDA 12.8，而不是系统默认 `/usr/local/cuda`：

```bash
export CUDA_HOME=/home/yuehui/miniforge3/envs/RadarPillar
export CPATH=$CUDA_HOME/targets/x86_64-linux/include
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$CUDA_HOME/lib/python3.10/site-packages/torch/lib:$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib
export TORCH_CUDA_ARCH_LIST="12.0"
```

### 关键适配点

- `OpenPCDet/pcdet/models/roi_heads/roi_head_template.py`：修复 ROI BCE loss 在 `rcnn_cls_labels=-1` ignore 样本上触发 CUDA assert 的问题。现在先把 ignore 标签临时夹到 `[0, 1]`，再用 mask 排除。
- `OpenPCDet/pcdet/datasets/processor/data_processor.py`：稀疏雷达点数不足时，`sample_points` 允许有放回采样，避免小样本直接报错。
- `environment/cfgs/pcdet_models/*`：新增 PointRCNN、Part-A2、PV-RCNN、Voxel R-CNN、PV-RCNN++ 的 aiQii/MMAUD smoke/full 配置。
- `RadarPillar`：新增 `MMRadarDataset` 接入；关闭原项目对速度分解特征的假设；修复 Python 3.10 `collections.Iterable`；修复 PyTorch 2.6+ `torch.load(weights_only=False)`；延迟导入 `eval_pointseg`，避开 KITTI/numba CUDA rotate IoU 在 RTX 50/CUDA12.8 上的崩溃；训练中加入周期验证和 `periodic_metrics/metrics_history.json`。
- `RadarNeXt`：新增 `MMRadarDataset` 和 `MMRadarMetric`；把硬编码 `CUDA_VISIBLE_DEVICES=6` 改为默认 GPU 0；配置 CenterPoint/RadarNeXt 单类 `Drone` 检测和 aiQii 四分类检测，使用 4 维毫米波点云特征。四分类时数据集元信息会把 `Air3S`、`Mini4pro`、`Mavic3Pro`、`jingling4` 映射到配置中的 4 类标签。

### 扩展模型 smoke 结论

以下 smoke 都完成了 1 epoch train → validation → 指标输出：

| 模型        | aiQiiDataset | MMAUD |
| ----------- | -----------: | ----: |
| PointRCNN   |           OK |    OK |
| Part-A2     |           OK |    OK |
| PV-RCNN     |           OK |    OK |
| Voxel R-CNN |           OK |    OK |
| PV-RCNN++   |           OK |    OK |
| RadarPillar |           OK |    OK |
| RadarNeXt   |           OK |    OK |

1 epoch smoke 的分数只用于确认环境、数据流、loss、checkpoint、验证和指标链路正常，不代表正式收敛结果。

### 一键运行扩展模型

新增脚本：

```text
E:\Scholar\mmradarDetect\environment\run_mmradar_extended_suite_wsl.py
```

正式跑 MMAUD，仍然保持单类 `Drone`：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_extended_suite_wsl.py `
  --dataset mmaud --run-tag mmaud_ext_v1 --mode full `
  --workers 4 --batch-size 2 --radarnext-batch-size 1 --retries 2
```

正式跑 aiQiiDataset 单类版本，四种无人机统一归为 `Drone`：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_extended_suite_wsl.py `
  --dataset aiqii --aiqii-classes single --run-tag aiqii_ext_single_v1 --mode full `
  --workers 4 --batch-size 2 --radarnext-batch-size 1 --retries 2
```

正式跑 aiQiiDataset 四分类版本，分别检测 `Air3S`、`Mini4pro`、`Mavic3Pro`、`jingling4`：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_extended_suite_wsl.py `
  --dataset aiqii --aiqii-classes multiclass --run-tag aiqii_ext_multiclass_v1 --mode full `
  --workers 4 --batch-size 2 --radarnext-batch-size 1 --retries 2
```

只快速复测 smoke：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_extended_suite_wsl.py `
  --dataset aiqii --aiqii-classes multiclass --run-tag smoke_check --mode smoke `
  --workers 2 --batch-size 2 --radarnext-batch-size 1 --retries 0
```

只跑某几个模型：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_extended_suite_wsl.py `
  --dataset mmaud --run-tag mmaud_ext_v1 --mode full `
  --workers 4 --batch-size 2 --radarnext-batch-size 1 `
  --only PointRCNN RadarPillar RadarNeXt
```

可用模型名：

```text
PointRCNN PartA2 PVRCNN VoxelRCNN PVRCNNPlusPlus RadarPillar RadarNeXt
```

扩展脚本输出目录：

```text
E:\Scholar\mmradarDetect\environment\extended_runs\<dataset>\<mode>\<run-tag>\
E:\Scholar\mmradarDetect\environment\extended_runs\aiqii_multiclass\<mode>\<run-tag>\
├── suite_status.json
├── EXTENDED_MODELS_METRICS.md
├── EXTENDED_MODELS_METRICS.json
└── <Model>_attempt_<N>.log
```

`EXTENDED_MODELS_METRICS.json` 会汇总所有历史验证指标。OpenPCDet/RadarPillar 读取 `periodic_metrics/metrics_history.json`，RadarNeXt 读取 MMEngine 的 `vis_data/scalars.json`。

### 单模型命令示例

OpenPCDet 扩展模型以 PointRCNN 为例：

```bash
cd /mnt/e/Scholar/mmradarDetect/OpenPCDet/tools
/home/yuehui/miniforge3/envs/PointPillar/bin/python train.py \
  --cfg_file cfgs/mmradar_models/pointrcnn_aiqii_full.yaml \
  --workers 4 --batch_size 2 --extra_tag aiqii_ext_v1 \
  --ckpt_save_interval 10 --max_ckpt_save_num 9 --eval_interval 5 --max_waiting_mins 0
```

把配置名替换为：

```text
pointrcnn_aiqii_full.yaml / pointrcnn_aiqii_multiclass_full.yaml / pointrcnn_mmaud_full.yaml
parta2_aiqii_full.yaml / parta2_aiqii_multiclass_full.yaml / parta2_mmaud_full.yaml
pvrcnn_aiqii_full.yaml / pvrcnn_aiqii_multiclass_full.yaml / pvrcnn_mmaud_full.yaml
voxelrcnn_aiqii_full.yaml / voxelrcnn_aiqii_multiclass_full.yaml / voxelrcnn_mmaud_full.yaml
pvrcnnplusplus_aiqii_full.yaml / pvrcnnplusplus_aiqii_multiclass_full.yaml / pvrcnnplusplus_mmaud_full.yaml
```

RadarPillar：

```bash
cd /mnt/e/Scholar/mmradarDetect/RadarPillar
LD_LIBRARY_PATH=/home/yuehui/miniforge3/envs/RadarPillar/lib/python3.10/site-packages/torch/lib:/home/yuehui/miniforge3/envs/RadarPillar/targets/x86_64-linux/lib:/home/yuehui/miniforge3/envs/RadarPillar/lib \
PYTHONPATH=/mnt/e/Scholar/mmradarDetect:/mnt/e/Scholar/mmradarDetect/RadarPillar \
/home/yuehui/miniforge3/envs/RadarPillar/bin/python tools/train.py \
  --cfg_file tools/cfgs/mmradar_models/radarpillar_aiqii_full.yaml \
  --workers 4 --batch_size 2 --extra_tag aiqii_ext_v1 \
  --ckpt_save_interval 10 --max_ckpt_save_num 9 --max_waiting_mins 0
```

RadarNeXt：

```bash
cd /mnt/e/Scholar/mmradarDetect/RadarNeXt
PYTHONPATH=/mnt/e/Scholar/mmradarDetect:/mnt/e/Scholar/mmradarDetect/RadarNeXt \
/home/yuehui/miniforge3/envs/RadarNeXt/bin/python tools/train.py \
  configs/mmradar/radarnext_centerpoint_aiqii_full.py \
  --work-dir work_dirs/radarnext_aiqii_ext_v1
```

| 脚本                                  | 用途                                                 | 跑哪些模型                                                                   |
| ------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------- |
| `run_mmradar_smoke_train_wsl.py`    | 快速 smoke，主要确认环境、数据、loss、验证和指标链路 | 原始 7 项目，短跑                                                            |
| `run_mmradar_full_suite_wsl.py`     | 正式连续训练原始 7 项目                              | OpenPCDet、InterFusion、PFA-NET、DSVT、VoxelNeXt、CenterPoint、PillarNet-LTS |
| `run_mmradar_extended_suite_wsl.py` | 正式/烟测扩展模型                                    | PointRCNN、Part-A2、PV-RCNN、Voxel R-CNN、PV-RCNN++、RadarPillar、RadarNeXt  |
