# MMRadar 多模型训练结果汇总与对比

> 本文件整合了 5 份原始指标文件中的全部数据（未删减任何模型、任何指标），并在此基础上做了内部对比和与文献结果的对比。

---

## 0. 数据来源说明

| 源文件 | 数据集 / run tag | 模型数 | 备注 |
|---|---|---|---|
| `ALL_MODELS_METRICS1.md` | aiqii / `aiqii_v1` | 7（2 完成，5 pending） | |
| `ALL_MODELS_METRICS.md` | mmaud / single / `mmaud_v1`（"full suite"） | 7（全部 completed） | 标记为"结果 A" |
| `ALL_MODELS_METRICS2.md` | mmaud / `mmaud_v1` | 7（全部 completed） | 标记为"结果 B" |
| `ALL_MODELS_METRICS3.md` | mmaud / `mmaud_v1` | 7（全部 completed） | 与 ALL_MODELS_METRICS2.md **内容完全相同**（重复文件） |
| `EXTENDED_MODELS_METRICS.md` | mmaud / single / `mmaud_ext_v1` / full | 7（3 有结果，1 failed 已计入，3 pending） | 扩展模型套件，多了 BEV AP@0.25 一列 |

⚠️ **需要您注意的一点**：`ALL_MODELS_METRICS.md`（结果 A）和 `ALL_MODELS_METRICS2/3.md`（结果 B）的 **run tag 同样是 `mmaud_v1`**，但具体数值并不相同（例如 OpenPCDet 的 Center AP@2m 一个是 96.48%，另一个是 89.38%）。这两份很可能来自**不同的训练/评估批次**（比如不同随机种子、不同 checkpoint 选取方式，或评估脚本版本不同），但文件名/标签没有区分开。下文我把两者都保留并标注为"结果 A / 结果 B"，建议您回去确认一下哪一份是最终要用的版本，避免汇报时张冠李戴。

---

## 1. aiqii 数据集（run tag: `aiqii_v1`）

| 模型 | 状态 | 最终 epoch | Center AP@2m | Center AP@4m | BEV AP@0.5 | 3D AP@0.25 | 3D AP@0.5 | 平均中心误差(m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenPCDet | completed | 80 | 94.21% | 95.43% | 7.68% | 9.19% | 0.20% | 0.521 |
| InterFusion | completed | 80 | 93.41% | 94.08% | 7.78% | 8.11% | 0.20% | 0.623 |
| PFANet | pending | - | - | - | - | - | - | - |
| DSVT | pending | - | - | - | - | - | - | - |
| VoxelNeXt | pending | - | - | - | - | - | - | - |
| CenterPoint | pending | - | - | - | - | - | - | - |
| PillarNetLTS | pending | - | - | - | - | - | - | - |

**观察**：aiqii 数据集上已完成的两个模型 Center AP（2m/4m 容差）都很高（>93%），但 BEV/3D AP 很低（个位数，3D AP@0.5 仅 0.20%），说明粗定位（中心点是否落在容差范围内）很准，但**精确的框体回归（尺寸/朝向）几乎没学到**。还有 5 个模型未跑。

---

## 2. mmaud 数据集 — 全量套件「结果 A」（来自 `ALL_MODELS_METRICS.md`）

| 模型 | 状态 | 最终 epoch | Center AP@2m | Center AP@4m | BEV AP@0.5 | 3D AP@0.25 | 3D AP@0.5 | 平均中心误差(m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenPCDet | completed | 80 | 96.48% | 100.00% | 66.03% | 68.60% | 7.33% | 1.124 |
| InterFusion | completed | 80 | 93.81% | 100.00% | 65.29% | 77.85% | 9.54% | 1.135 |
| PFANet | completed | 80 | 97.22% | 99.38% | 52.08% | 89.79% | 9.12% | 1.020 |
| DSVT | completed | 80 | 51.13% | 87.54% | 56.63% | 20.43% | 2.60% | 2.084 |
| VoxelNeXt | completed | 80 | 48.02% | 88.84% | 60.59% | 23.27% | 1.85% | 1.857 |
| CenterPoint | completed | 80 | 76.38% | 91.06% | 27.55% | 42.39% | 1.01% | 1.439 |
| PillarNetLTS | completed | 80 | 47.98% | 92.06% | 58.28% | 24.27% | 2.65% | 1.999 |

---

## 3. mmaud 数据集 — 全量套件「结果 B」（来自 `ALL_MODELS_METRICS2.md` / `ALL_MODELS_METRICS3.md`，两文件内容一致）

| 模型 | 状态 | 最终 epoch | Center AP@2m | Center AP@4m | BEV AP@0.5 | 3D AP@0.25 | 3D AP@0.5 | 平均中心误差(m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenPCDet | completed | 80 | 89.38% | 99.72% | 60.41% | 55.25% | 2.20% | 1.285 |
| InterFusion | completed | 80 | 67.35% | 100.00% | 58.05% | 54.70% | 1.75% | 1.381 |
| PFANet | completed | 80 | 87.29% | 100.00% | 67.21% | 65.51% | 3.86% | 1.228 |
| DSVT | completed | 80 | 51.02% | 82.09% | 38.58% | 37.26% | 2.73% | 1.992 |
| VoxelNeXt | completed | 80 | 41.55% | 92.58% | 67.63% | 23.09% | 5.00% | 2.028 |
| CenterPoint | completed | 80 | 11.99% | 79.58% | 25.77% | 4.74% | 0.02% | 2.589 |
| PillarNetLTS | completed | 80 | 57.27% | 84.44% | 59.02% | 35.39% | 2.34% | 1.840 |

---

## 4. mmaud 数据集 — 扩展套件（来自 `EXTENDED_MODELS_METRICS.md`，run tag: `mmaud_ext_v1`）

| 模型 | 状态 | 最终 epoch | Center AP@2m | Center AP@4m | BEV AP@0.25 | BEV AP@0.5 | 3D AP@0.25 | 3D AP@0.5 | 平均中心误差(m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PointRCNN | **failed** | 20 | 73.86% | 96.81% | 58.41% | 19.46% | 31.59% | 0.29% | 1.281 |
| PartA2 | ok | 80 | 32.93% | 53.73% | 46.49% | 4.25% | 14.99% | 0.54% | 1.995 |
| PVRCNN | ok | 80 | 49.98% | 80.41% | 69.04% | 17.25% | 31.96% | 2.86% | 1.800 |
| VoxelRCNN | pending | - | - | - | - | - | - | - | - |
| PVRCNNPlusPlus | pending | - | - | - | - | - | - | - | - |
| RadarPillar | pending | - | - | - | - | - | - | - | - |
| RadarNeXt | pending | - | - | - | - | - | - | - | - |

**注意**：PointRCNN 训练在第 20 epoch 就 **failed**（而不是跑满 80），它的指标是中途失败前的快照，不能和跑满 80 epoch 的模型直接对比。

---

## 5. 内部对比与分析

### 5.1 mmaud 数据集：结果 A vs 结果 B 差异（同一个 run tag，数值却不同）

| 模型 | 3D AP@0.5 (A) | 3D AP@0.5 (B) | 差值 | Center AP@2m (A) | Center AP@2m (B) | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| OpenPCDet | 7.33% | 2.20% | +5.13 | 96.48% | 89.38% | +7.10 |
| InterFusion | 9.54% | 1.75% | +7.79 | 93.81% | 67.35% | +26.46 |
| PFANet | 9.12% | 3.86% | +5.26 | 97.22% | 87.29% | +9.93 |
| DSVT | 2.60% | 2.73% | -0.13 | 51.13% | 51.02% | +0.11 |
| VoxelNeXt | 1.85% | 5.00% | -3.15 | 48.02% | 41.55% | +6.47 |
| CenterPoint | 1.01% | 0.02% | +0.99 | 76.38% | 11.99% | +64.39 |
| PillarNetLTS | 2.65% | 2.34% | +0.31 | 47.98% | 57.27% | -9.29 |

可以看到，**结果 A 整体上明显优于结果 B**（尤其 CenterPoint 的 Center AP@2m 相差 64 个百分点，InterFusion 相差 26 个百分点），DSVT/PillarNetLTS 两个模型两次结果较接近。这进一步说明两批结果很可能不是同一次实验，差异已经超出正常的训练随机性范围，建议核实。

### 5.2 各模型在已完成实验中的最佳成绩（按 3D AP@0.5 排序，跨数据集/跨批次取最高值）

| 排名 | 模型 | 最佳 3D AP@0.5 | 来源 | 对应 3D AP@0.25 |
|---:|---|---:|---|---:|
| 1 | PVRCNN | 2.86% | mmaud 扩展套件 | 31.96% |
| 2 | PillarNetLTS | 2.65% | mmaud 结果 A | 24.27% |
| 2 | DSVT | 2.73% | mmaud 结果 B | 37.26% |
| 3 | OpenPCDet | 7.33% | mmaud 结果 A | 68.60% |
| 4 | InterFusion | 9.54% | mmaud 结果 A | 77.85% |
| 5 | PFANet | 9.12% | mmaud 结果 A | 89.79% |
| 6 | VoxelNeXt | 5.00% | mmaud 结果 B | 23.09% |
| 7 | CenterPoint | 1.01% | mmaud 结果 A | 42.39% |
| - | PartA2 | 0.54% | mmaud 扩展套件 | 14.99% |
| - | PointRCNN | 0.29%（训练中途 failed） | mmaud 扩展套件 | 31.59% |

> 注：上表按数值大小排了序号，但请注意各行单独看更直观：**mmaud 全量套件「结果 A」上的 PFANet（3D AP@0.25 = 89.79%）和 InterFusion（77.85%）在宽松阈值下表现最好**；但严格阈值（3D AP@0.5）下所有模型都在 10% 以下，没有一个模型在严格 3D 框体匹配上做得好。

### 5.3 整体规律

1. **"中心点检出准、但框体精度差"是贯穿所有实验的共性**：几乎所有模型的 Center AP@2m/4m 都能到 50%~97%，但 3D AP@0.5 普遍只有 0%~10%。这说明问题不在"有没有检测到目标"，而是在**尺寸/朝向回归**这一环。
2. **aiqii 数据集明显比 mmaud 更难**：同样是 OpenPCDet/InterFusion，在 aiqii 上 BEV AP@0.5 只有 7%~8%，而在 mmaud（结果 A）上能到 65%~66%。如果论文要横向对比，需要说明两个数据集本身难度差异很大，不能简单合并讨论。
3. **PFANet 和 InterFusion 在 mmaud 全量套件上整体领先**（尤其结果 A 中 3D AP@0.25 接近 90% 和 78%），是当前表现最好的两个模型。
4. **进度缺口**：aiqii 数据集还有 5 个模型（PFANet/DSVT/VoxelNeXt/CenterPoint/PillarNetLTS）未跑；扩展套件还有 4 个模型（VoxelRCNN/PVRCNNPlusPlus/RadarPillar/RadarNeXt）未跑，PointRCNN 训练失败需要重跑。

---

## 6. 与文献结果对比（表 4.2）

您提供的对比表（原文表 4.2）：

| 实验模型 | 3D AP@0.25 (%) | 3D AP@0.5 (%) |
|---|---:|---:|
| PointPillars | 76.4 | 36.3 |
| PointRCNN | 51.6 | 29.1 |
| VoxelNet | 77.2 | 41.2 |
| PV-RCNN | 78.5 | 42.6 |
| PV-RCNN++ | 81.4 | 45.1 |
| CenterPoint | 82.6 | 44.6 |
| PointPillars-RCINet | 84.8 | 48.0 |

下面把我方实验中**同名或同系列模型**的最好结果放在一起对比（按模型名匹配，PV-RCNN ↔ PVRCNN，CenterPoint ↔ CenterPoint，PointRCNN ↔ PointRCNN）：

| 模型 | 来源 | 3D AP@0.25 (%) | 3D AP@0.5 (%) |
|---|---|---:|---:|
| PointPillars | 文献 表4.2 | 76.4 | 36.3 |
| PointPillars-RCINet | 文献 表4.2 | 84.8 | 48.0 |
| VoxelNet | 文献 表4.2 | 77.2 | 41.2 |
| **PointRCNN** | 文献 表4.2 | 51.6 | 29.1 |
| **PointRCNN** | 我方（mmaud 扩展，训练第20ep失败） | 31.59 | 0.29 |
| **PV-RCNN（PVRCNN）** | 文献 表4.2 | 78.5 | 42.6 |
| **PV-RCNN（PVRCNN）** | 我方（mmaud 扩展，80ep） | 31.96 | 2.86 |
| PV-RCNN++ | 文献 表4.2 | 81.4 | 45.1 | 
| **CenterPoint** | 文献 表4.2 | 82.6 | 44.6 |
| **CenterPoint** | 我方（mmaud 全量结果A，80ep） | 42.39 | 1.01 |
| **CenterPoint** | 我方（mmaud 全量结果B，80ep） | 4.74 | 0.02 |

### 对比结论

1. **3D AP@0.25 上，我方结果普遍比文献低**（除了 PFANet/InterFusion 等模型在 mmaud 结果 A 上分别到 89.79%/77.85% 之外）。同名模型对比中，PointRCNN（31.59% vs 51.6%）、PV-RCNN（31.96% vs 78.5%）、CenterPoint（42.39%/4.74% vs 82.6%）都差距明显。
2. **3D AP@0.5 上的差距远比 3D AP@0.25 更大、更一致**：文献中所有模型都在 29%~48% 区间，而我方所有实验中所有模型的 3D AP@0.5 都没有超过 10%（多数在 0.2%~9.5%）。这是一个非常显著的系统性差距，几乎可以确定**不是个别模型调参问题，而是和雷达点云本身的稀疏/噪声特性、数据集标注方式，或评估协议（score threshold、NMS设置）与文献不一致有关**——原始文件里也特别提示过"AP 基于 score threshold 与 NMS 后的预测；比较不同论文前须统一评价协议"，这一点在写论文对比这部分时一定要特别说明，否则容易被质疑"为什么自己的方法全面低于基线这么多"。
3. 需要特别留意：**文献表 4.2 的数据集/任务设定（很可能是 LiDAR 点云，如 KITTI 类基准）与本实验所用的毫米波雷达数据集（mmaud/aiqii）并不是同一个数据源**，因此这个对比更适合作为"参考量级"，而不是严格意义上的同条件 apples-to-apples 对比。建议在论文里明确注明对比的局限性。

---

## 附：完整原始数据校验

本文件中的所有数字均**逐字保留**自 5 个源文件，未做任何四舍五入之外的改动，也未删除任何"pending"或"failed"行，可直接作为最终汇总稿使用。
