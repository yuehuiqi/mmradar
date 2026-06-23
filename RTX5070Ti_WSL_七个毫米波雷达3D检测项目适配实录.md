# 在 RTX 5070 Ti + Windows/WSL 上跑通七套 3D 检测代码：从 CUDA 编译到毫米波雷达数据集适配

> 本文记录一次真实的工程迁移：在一台 Windows 笔记本上，把 OpenPCDet、CenterPoint、PillarNet-LTS、DSVT、InterFusion、PFA-NET 和 VoxelNeXt 七套代码分别配置到独立环境中，再让它们同时支持两套毫米波雷达数据——公开的 MMAUD 和自制的 aiQiiDataset。最后不是停在“可以 import”这一步，而是把 7 个项目 × 2 套数据的 14 个训练、保存和评估流程全部跑通。

这类工作最麻烦的地方通常不是某一条安装命令，而是年代不同的代码、最新显卡、不同 CUDA API、不同数据坐标系和不同训练框架同时撞在一起。单独看每个问题都不算大，叠到七个仓库里以后，就很容易变成“这个项目能编译，另一个项目能读数据，第三个项目训练到一半又崩了”。所以这次没有把它当成七次互不相干的安装，而是先统一底座，再抽出公共数据层，最后逐项目消掉兼容问题。

为了便于复现，下面会给出关键命令、配置和源码补丁；不太关键的重复代码则只讲修改原则，不整段复制。

## 一、最后做到什么程度

先说结果。机器与软件栈如下：

| 项目 | 实际配置 |
|---|---|
| 操作系统 | Windows + WSL2，Ubuntu 24.04.4 LTS |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| 显存 | 12227 MiB |
| 驱动 | 595.79 |
| Compute Capability | 12.0，也就是 `sm_120` |
| Python | 3.10 |
| PyTorch | 2.7.1+cu128 |
| CUDA Runtime | 12.8 |
| CUDA Toolkit / nvcc | 12.8，V12.8.93 |
| spconv | 2.3.6，安装包为 `spconv-cu120` |
| NumPy / Numba | 1.23.5 / 0.57.1 |

七个项目各自使用一个 WSL conda 环境：

| 源码项目 | 环境名 | 备注 |
|---|---|---|
| OpenPCDet | `PointPillar` | Windows 侧已经有一个 `openpcdet`，因此换名，避免混淆 |
| CenterPoint | `CenterPoint` | Det3D 体系 |
| PillarNet-LTS | `PillarNetLTS` | 去掉连字符，按驼峰命名 |
| DSVT | `DSVT` | 动态 pillar 与 transformer 主干 |
| InterFusion | `InterFusion` | 较老的 PCDet 分支，兼容工作较多 |
| PFA-NET | `PFANet` | 较老的 PCDet 分支 |
| VoxelNeXt | `VoxelNeXt` | 3D 稀疏卷积主干 |

这里有一个很重要的边界：Windows 里原来已有的 `CCF`、`TXGH`、`chatapi`、`meditune`、`openpcdet` 和 `base` 环境没有被改动。新环境全部建在 WSL 的 Miniforge 下，路径统一为：

```text
/home/yuehui/miniforge3/envs/<EnvName>
```

最终完成了 14 次 smoke 实验，每次至少跑完 1 个 epoch。当前版本中 PCDet 与 Det3D 两类项目都会在 smoke 最终 epoch 后执行验证：

| 实验 | 状态 | 用时 |
|---|---:|---:|
| OpenPCDet + aiQiiDataset | OK | 23.92 s |
| OpenPCDet + MMAUD | OK | 24.08 s |
| InterFusion + aiQiiDataset | OK | 20.59 s |
| InterFusion + MMAUD | OK | 21.40 s |
| PFA-NET + aiQiiDataset | OK | 21.93 s |
| PFA-NET + MMAUD | OK | 18.56 s |
| DSVT + aiQiiDataset | OK | 23.37 s |
| DSVT + MMAUD | OK | 25.33 s |
| VoxelNeXt + aiQiiDataset | OK | 32.77 s |
| VoxelNeXt + MMAUD | OK | 22.43 s |
| CenterPoint + aiQiiDataset | OK | 21.42 s |
| CenterPoint + MMAUD | OK | 16.53 s |
| PillarNet-LTS + aiQiiDataset | OK | 39.08 s |
| PillarNet-LTS + MMAUD | OK | 33.60 s |

这些数字只是通路测试耗时，不是性能 benchmark。smoke 的意义是确认数据读取、预处理、前向、反向、loss、CUDA 算子、checkpoint 和评估都能走完。只训练一个 epoch 时，零召回、较大的定位损失或者指标波动都不值得过度解读；真正值得警惕的是 `NaN`、维度突变、CUDA illegal memory access、保存后无法加载，以及训练与验证坐标系不一致。

## 二、为什么选 WSL，而不是继续堆 Windows conda 环境

这台机器原本就有多个 Windows conda 环境。如果只是跑一段纯 Python 推理，继续在 Windows 下装依赖未必不行；但这七个项目都带着 CUDA/C++ 扩展，里面有 `iou3d_nms`、PointNet2、DCN、pillar ops、roiaware pool 等组件。它们的构建脚本大多默认 Linux 工具链，很多源码还直接写了 `/usr/local/cuda/include`、GCC 参数或 Linux 风格路径。

在 Windows 原生环境里硬改当然也能做，但会额外引入 MSVC、Windows CUDA 路径、`.pyd` 链接、编译器版本和 shell 脚本改写等问题。WSL 的优势不是“自动解决 CUDA”，而是让这些以 Linux 为默认平台的研究代码回到自己熟悉的工具链里：Bash、GCC、Ninja、`setup.py build_ext` 和 Linux 版 PyTorch 扩展。

WSL 下有两个概念要分清：

1. GPU 驱动由 Windows 主机提供，WSL 里通过 NVIDIA 的虚拟化接口访问显卡。不要为了“补驱动”又在 Ubuntu 里装一套 Linux 显卡驱动。
2. 编译 CUDA 扩展仍然需要 `nvcc` 和头文件。PyTorch wheel 自带的是运行时依赖，不等于完整的 CUDA 编译工具链。

这也是为什么环境中同时有 PyTorch `cu128` 和 conda 安装的 CUDA Toolkit 12.8。前者让 PyTorch 运行，后者给各仓库编译 `.so` 扩展。

先用以下命令确认 WSL 确实能看到新显卡：

```powershell
wsl.exe -d Ubuntu-24.04 -- nvidia-smi
```

在 WSL 中再检查 PyTorch：

```bash
/home/yuehui/miniforge3/envs/CenterPoint/bin/python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
```

实际输出对应 `2.7.1+cu128`、CUDA 12.8、RTX 5070 Ti Laptop GPU 和 `(12, 0)`。最后这个 `(12, 0)` 很关键。50 系显卡是较新的架构，如果继续沿用旧项目常见的 CUDA 11.x、早期 PyTorch和只包含 `sm_75/sm_80/sm_86` 的编译产物，往往不是安装阶段报错，而是运行 kernel 时才告诉你没有对应架构的代码。

## 三、环境策略：一个种子环境，七个独立环境

最开始需要决定：七个项目共用一个环境，还是每个项目单独建环境？这里选择“依赖底座一致，但环境物理隔离”。具体做法是先创建 `CenterPoint` 种子环境，装好 Python、PyTorch、CUDA Toolkit 和公共 Python 包；验证 GPU 后，再克隆出另外六个环境。

这样做比完全独立安装七遍省时间，也比所有项目挤在同一个环境里稳妥。研究仓库很喜欢在 `pip install -e .` 时注册同名包，例如多个项目都叫 `pcdet`，CenterPoint 与 PillarNet-LTS 又都叫 `det3d`。如果共用环境，后安装的 editable package 很容易把前一个覆盖掉。更麻烦的是，它们各自编译出的扩展可能同名，但源代码版本不同。环境隔离以后，至少某个仓库的重编译不会直接改变另外六个项目的运行状态。

环境创建脚本的核心如下：

```bash
ENV_NAMES=(
  CenterPoint
  DSVT
  InterFusion
  PointPillar
  PFANet
  PillarNetLTS
  VoxelNeXt
)

conda create -y -n CenterPoint \
  python=3.10 pip=25.1 setuptools=69.5.1 wheel ninja cmake packaging

conda install -y -n CenterPoint -c nvidia/label/cuda-12.8.1 \
  cuda-toolkit=12.8

/home/yuehui/miniforge3/envs/CenterPoint/bin/python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

for env_name in "${ENV_NAMES[@]:1}"; do
  conda create -y -n "${env_name}" --clone CenterPoint
done
```

公共依赖没有盲目追最新。最终固定了这些容易互相牵制的版本：

```text
numpy==1.23.5
llvmlite==0.40.1
numba==0.57.1
scipy==1.10.1
matplotlib==3.7.5
opencv-python==4.8.1.78
pandas==2.0.3
scikit-learn==1.3.2
timm==0.9.16
```

固定 NumPy 1.23.5 主要是照顾老代码和 Numba。新 NumPy 对被移除的别名、pickle 模块路径和旧二进制扩展更严格；另一方面，如果为了老项目一味退回 Python 3.7、PyTorch 1.x，又会失去对 `sm_120` 和 CUDA 12.8 的现实支持。因此这里实际走的是“现代 GPU 栈 + 有控制地修老代码”，而不是复刻论文作者当年的整套环境。

spconv 的实际版本为 2.3.6，包名是 `spconv-cu120`，同时带有 `cumm-cu120`。虽然 PyTorch 是 cu128，但这个 CUDA 12 系 wheel 在当前环境中可以正常工作。安装时不能把老仓库 `setup.py` 中那个不带 CUDA 后缀的 `spconv` 依赖留着，否则 pip 可能解析到不合适的包。因此 InterFusion 和 PFA-NET 的 `install_requires` 中移除了裸 `spconv`，改为在环境准备阶段显式管理。

每个环境创建后都做一次硬检查，而不是只看 `pip install` 返回 0：

```python
import torch

assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (12, 0)
print(torch.__version__, torch.cuda.get_device_name(0))
```

这一步看起来有点较真，但很值。环境克隆、Windows/WSL 路径和多个 Python 可执行文件同时存在时，“刚才装包的 Python”和“训练脚本实际调用的 Python”不是同一个，是非常常见的事故。

## 四、为 `sm_120` 编译仓库自带 CUDA 扩展

50 系显卡上的关键变量是：

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
```

此外还要让 PyTorch 的扩展构建器找到 conda 环境里的 Toolkit。每个项目编译前统一设置：

```bash
PREFIX="/home/yuehui/miniforge3/envs/${ENV_NAME}"

export PATH="${PREFIX}/bin:${PATH}"
export CUDA_HOME="${PREFIX}"
export CUDA_PATH="${PREFIX}"
export CPATH="${PREFIX}/targets/x86_64-linux/include:${PREFIX}/include:${CPATH:-}"
export LIBRARY_PATH="${PREFIX}/targets/x86_64-linux/lib:${PREFIX}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${PREFIX}/targets/x86_64-linux/lib:${PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
export MAX_JOBS=4
```

`CUDA_HOME` 不能只凭习惯写 `/usr/local/cuda`，因为这次完整 Toolkit 位于 conda 环境。`CPATH`、`LIBRARY_PATH` 和 `LD_LIBRARY_PATH` 分别补上编译期头文件、链接期库和运行时动态库路径。`MAX_JOBS=4` 是为了避免七个老项目编译时把内存和 CPU 一口气吃满；它不影响最后生成的 kernel 性能。

PCDet 系仓库使用 editable 安装：

```bash
cd /mnt/e/Scholar/mmradarDetect/OpenPCDet
/home/yuehui/miniforge3/envs/PointPillar/bin/python \
  -m pip install -e . --no-build-isolation
```

其他项目只替换目录和环境名。这里加 `--no-build-isolation` 是有意的：CUDA 扩展编译必须看到当前环境里已经验证过的 PyTorch、CUDA 路径与编译参数。如果让 pip 临时建一个隔离构建环境，它可能下载另一套 torch，或者根本拿不到当前 Toolkit。

CenterPoint 和 PillarNet-LTS 属于 Det3D 体系，需要单独编译 ops。实际清单如下：

```text
CenterPoint
├── det3d/ops/iou3d_nms
└── det3d/ops/dcn

PillarNet-LTS
├── det3d/ops/iou3d_nms
├── det3d/ops/pillar_ops
├── det3d/ops/pillar_ops-ba
└── det3d/ops/roiaware_pool3d
```

每个目录都执行：

```bash
python setup.py build_ext --inplace
```

老 `iou3d_nms/setup.py` 把 CUDA include 写死成了 `/usr/local/cuda/include`。在 conda Toolkit 下这个路径自然不对，所以改成读取环境变量：

```python
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
cxx_args = ["-g"]
if cuda_home:
    cxx_args.append(f'-I{os.path.join(cuda_home, "include")}')

CUDAExtension(
    # sources 省略
    extra_compile_args={"cxx": cxx_args, "nvcc": ["-O2"]},
)
```

CenterPoint 的 DCN 还使用了旧版 ATen 宏 `AT_CHECK`。新 PyTorch 中对应的是 `TORCH_CHECK`，最小兼容补丁是：

```cpp
#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK
#endif
```

InterFusion 与 PFA-NET 的 PointNet2 C++ 源码引用了已经退出当前 PyTorch 公共头文件体系的 `THC/THC.h`，还声明了 `extern THCState *state;`。这些扩展的当前实现并没有真正依赖这个全局状态，因此删掉旧头文件和声明即可继续编译。相比为了两个过时符号把 PyTorch 降回 1.x，这种修改更符合这台机器的实际约束。

编译结束后不能只检查 `.so` 是否存在，还要在七个环境中分别导入 `spconv.pytorch` 和项目 ops，再构造小张量做一次前向。一个二进制文件生成成功，不代表它包含 `sm_120`，也不代表动态链接时能找到正确的 CUDA 库；真正 import 和执行一次，才算过第一关。

## 五、两套毫米波雷达数据不能直接“改个路径”就训练

两套数据最后都提供四维点特征：

```text
[x, y, z, intensity]
```

但它们的数据规模、点云稀疏度、文件格式、坐标定义和目标框尺寸差异很大。

| 数据集 | train | val | train 平均点数 | val 平均点数 | 点云格式 |
|---|---:|---:|---:|---:|---|
| aiQiiDataset | 13200 | 2560 | 123.55 | 116.92 | 元数据可对应 `.bin`，同时保留 `.npy` |
| MMAUD | 3834 | 185 | 19.96 | 13.42 | `.npy` |

aiQiiDataset 训练帧最少 3 个点，最多 649 个点；MMAUD 训练帧最少只有 1 个点，最多 169 个点。MMAUD 验证集平均一帧只有约 13 个点。这种差距会直接触发很多“在激光雷达数据上永远遇不到”的边界条件，例如一批数据里只有一个非空 voxel，或者某个 batch 中不同样本可供 top-k 的候选数量不一样。

目标框尺寸也明显不同。aiQiiDataset 的训练集平均尺寸约为：

```text
[0.476, 0.448, 0.189]
```

MMAUD 中统一为：

```text
[2.0, 3.0, 3.0]
```

因此 PointPillar 的 anchor 不能共用。aiQiiDataset 使用接近无人机真实小尺寸的 anchor：

```yaml
anchor_sizes: [[0.50, 0.45, 0.20]]
anchor_bottom_heights: [-0.10]
matched_threshold: 0.45
unmatched_threshold: 0.25
```

MMAUD 则改为：

```yaml
anchor_sizes: [[2.0, 3.0, 3.0]]
anchor_bottom_heights: [-1.5]
```

如果只把 `DATA_PATH` 换掉而继续用原 anchor，训练程序可能完全不报错，但正负样本分配会非常糟。这种“能跑但学不对”的问题，比 import error 更难发现。

## 六、统一坐标系：MMAUD 的 camera 风格坐标转换

aiQiiDataset 已经按 lidar 风格组织，而 MMAUD 的原始点与框使用 camera 风格坐标：`x` 向右、`y` 向下、`z` 向前。七个检测项目的 BEV 管线更适合统一成 lidar 风格：`x` 向前、`y` 向左、`z` 向上。

点坐标变换为：

```python
xyz = points[:, :3].copy()
points[:, 0] = xyz[:, 2]
points[:, 1] = -xyz[:, 0]
points[:, 2] = -xyz[:, 1]
```

也就是：

```text
x_lidar =  z_camera
y_lidar = -x_camera
z_lidar = -y_camera
```

框中心做同样的变换，尺寸轴也需要跟着换：

```python
centers = boxes[:, :3].copy()
sizes = boxes[:, 3:6].copy()

boxes[:, 0] = centers[:, 2]
boxes[:, 1] = -centers[:, 0]
boxes[:, 2] = -centers[:, 1]
boxes[:, 3] = sizes[:, 2]
boxes[:, 4] = sizes[:, 0]
boxes[:, 5] = sizes[:, 1]
boxes[:, 6] = 0.0
```

这里不能只转换点而忘记框。点云和标签分别看都“像是正常数值”，但只要它们处在不同坐标系里，模型看到的就是错误监督。

在生成元数据时，MMAUD 的 info 中记录：

```python
"point_format": "npy",
"coordinate_transform": "camera_to_lidar",
```

在 PCDet 配置中也显式写出：

```yaml
COORDINATE_TRANSFORM: camera_to_lidar
```

这比看到某个目录名就偷偷转换更可靠。数据行为由元数据或配置明确决定，以后加第三套数据时不会因为文件恰好也是 `.npy` 就被误转换。

## 七、统一数据范围与网格尺寸

七个项目的输入点云范围统一为：

```text
[-16, -20, -8, 64, 20, 8]
```

即 x 方向跨度 80 米，y 方向跨度 40 米，z 方向跨度 16 米。PointPillar 和 DSVT 在平面上采用 `0.5 × 0.5` 的网格，因此 BEV 尺寸是：

```text
80 / 0.5 = 160
40 / 0.5 = 80
```

最终得到 `160 × 80`。这个尺寸不是随手凑出来的，它要同时满足两套数据的覆盖范围，以及 DSVT 窗口、BEV backbone 多次下采样和上采样的整除关系。最初 DSVT 就出现过上采样分支尺寸分别为 130 和 132，最后在 `torch.cat` 时失败。把输入范围、voxel size、`sparse_shape` 和 `output_shape` 一起统一后，这个问题才真正消失。

不同模型的 z 向处理不必完全一样：

- PointPillar/CenterPoint 对 aiQiiDataset 使用 `[0.5, 0.5, 16]`，整段高度压成一个 pillar；MMAUD 使用 `[0.5, 0.5, 18]`，同样保证单层柱体。
- DSVT 使用动态 pillar 占位处理，`sparse_shape: [160, 80, 1]`。
- VoxelNeXt 需要真正的 3D 稀疏体素，使用 `[0.5, 0.5, 0.5]`，每个 voxel 最多 8 个点。
- PillarNet-LTS 直接使用 `pillar_size = 0.5`，由动态 pillar 编码器处理。

输入范围统一不等于所有模型的内部配置必须复制成同一份。模型结构不同，关键是每一级张量尺寸彼此自洽。

## 八、同时生成 PCDet 和 Det3D 两种 info

七个项目实际上分成两大代码体系：

- OpenPCDet、InterFusion、PFA-NET、DSVT、VoxelNeXt 使用 PCDet 风格数据集接口。
- CenterPoint、PillarNet-LTS 使用 Det3D 风格 pipeline。

如果为每个项目各写一套数据转换脚本，后面改标签格式会非常痛苦。因此准备脚本一次生成两种元数据：

```text
mmradar_infos_train.pkl
mmradar_infos_val.pkl
mmradar_infos_train_smoke.pkl
mmradar_infos_val_smoke.pkl

mmradar_det3d_infos_train.pkl
mmradar_det3d_infos_val.pkl
mmradar_det3d_infos_train_smoke.pkl
mmradar_det3d_infos_val_smoke.pkl
```

PCDet info 的核心结构为：

```python
{
    "point_cloud": {
        "num_features": 4,
        "lidar_idx": sample_id,
        "num_points": point_count,
    },
    "annos": {
        "name": names,
        "gt_boxes_lidar": boxes,
        "num_points_in_gt": np.full(len(boxes), -1, dtype=np.int32),
    },
}
```

Det3D info 则记录完整点云路径和格式：

```python
{
    "token": sample_id,
    "lidar_path": str(lidar_path),
    "point_format": point_format,
    "coordinate_transform": coordinate_transform,
    "sweeps": [],
    "gt_boxes": boxes,
    "gt_names": names,
}
```

两套数据都统一为单类 `Drone`。准备阶段还会检查点云必须是 `[N, >=4]`，标签必须有 8 个字段，每一帧文件都存在，而且不能没有目标框。比起让错误样本在训练第几个 epoch 才炸掉，预处理时就报出具体 sample id 更省时间。

另一个小坑来自 NumPy pickle。部分原始 info 由 NumPy 2.x 写出，而训练环境为了兼容 Numba 固定在 NumPy 1.23.5。新 NumPy 的内部模块路径包含 `numpy._core`，旧版本反序列化时找不到。这里加了一个兼容 unpickler：

```python
class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)
```

这比为了读一次 pickle 把七个训练环境整体升级到 NumPy 2.x 稳得多。

smoke 子集不是简单取前 64 帧，而是通过 `np.linspace` 从完整序列中均匀抽取 64 个训练样本和 32 个验证样本：

```python
indices = np.linspace(0, len(items) - 1, num=count, dtype=np.int64)
subset = [items[int(index)] for index in indices]
```

这样可以覆盖序列前中后段，避免刚好前几十帧的距离或场景过于单一。

## 九、公共数据集层：七个仓库只保留薄适配器

为了避免在五个 PCDet 分支里复制上百行数据读取逻辑，仓库根目录增加了 `mmradar_common`：

```text
mmradar_common/
├── pcdet_dataset.py
├── det3d_dataset.py
└── metrics.py
```

每个项目内部的 `mmradar_dataset.py` 只是很薄的一层。例如 PCDet 分支基本只需要：

```python
from pcdet.datasets.dataset import DatasetTemplate
from mmradar_common.pcdet_dataset import MMRadarDatasetMixin


class MMRadarDataset(MMRadarDatasetMixin, DatasetTemplate):
    pass
```

公共 mixin 负责加载 info、读取点云、坐标转换、组装训练字典、整理预测结果和调用评估。读取点云时对形状做硬检查：

```python
points = np.load(path)
if points.ndim != 2 or points.shape[1] != 4:
    raise ValueError(f"Expected Nx4 points in {path}, got {points.shape}")
points = points.astype(np.float32, copy=False)
```

给新 OpenPCDet 分支准备数据时，还补齐了增强状态字段：

```python
data_dict = {
    "frame_id": sample_id,
    "points": points,
    "flip_x": False,
    "flip_y": False,
    "noise_rot": 0.0,
    "noise_scale": 1.0,
}
```

这些字段在没有做增强时看似多余，但较新的后处理或数据增强代码会默认它们存在。显式初始化能避免同一公共数据集在不同 PCDet fork 中表现不一致。

Det3D 的公共 mixin 则实现 `get_sensor_data`、`ground_truth_annotations` 和 `evaluation`。这里明确限制 `nsweeps == 1`，因为当前两套毫米波数据都是单帧；如果以后做多帧累积，应当把时序位姿、时间特征和 sweep 合并正式设计进去，而不是先用重复帧冒充。

## 十、为什么没有直接沿用 KITTI AP

两套数据都是稀疏、单类的小目标毫米波检测，smoke 阶段的第一需求是判断预测中心有没有走到正确位置，而不是立即宣称某个标准 benchmark 的 AP。因此增加了一套简单透明的中心距离指标，在每帧内按预测分数降序做贪心一对一匹配，统计 0.5、1、2、4 米阈值下的 precision 和 recall，同时记录真实框到最近预测框的平均中心距离。

核心距离矩阵只有一行：

```python
distances = np.linalg.norm(
    gt_boxes[:, None, :3] - pred_boxes[None, :, :3],
    axis=2,
)
```

这套指标的优点是两个框架都能共用，而且坐标系或范围配错时，中心距离会很快暴露问题。它当然不能替代正式论文里的 3D AP、BEV AP 或 nuScenes 风格指标。后续做严肃对比时，应该先确定任务定义、IoU 阈值、距离分桶和难度划分，再固定正式评价协议。

## 十一、PCDet 配置继承的一个隐蔽问题

为了不在七个项目里复制完整 YAML，实际配置采用多层 `_BASE_CONFIG_`：项目目录里的配置只指向 `environment/cfgs` 下的公共模型配置，公共模型配置再引用公共数据配置。例如：

```yaml
# OpenPCDet/tools/cfgs/mmradar_models/pointpillar_aiqii_full.yaml
_BASE_CONFIG_: /mnt/e/Scholar/mmradarDetect/environment/cfgs/pcdet_models/pointpillar_aiqii_full.yaml
```

而 full 配置又继承 smoke 模型配置，只替换完整 info 和 epoch：

```yaml
_BASE_CONFIG_: /mnt/e/Scholar/mmradarDetect/environment/cfgs/pcdet_models/pointpillar_aiqii_smoke.yaml

DATA_CONFIG:
  _BASE_CONFIG_: /mnt/e/Scholar/mmradarDetect/environment/cfgs/pcdet/mmradar_aiqii_full.yaml

OPTIMIZATION:
  NUM_EPOCHS: 80
```

原始 OpenPCDet 配置合并代码在读到 base YAML 后直接 `config.update`，只处理一层，遇到 base 里面还有 base 时，内层配置可能没有递归展开。修复方式是先把 base 配置递归合并到临时字典：

```python
base_config = EasyDict()
merge_new_config(base_config, EasyDict(yaml_config))
config.update(base_config)
```

这个问题很容易伪装成“我的 YAML 参数为什么没生效”。路径没有错，文件也能打开，但继承链中间丢了一层。

## 十二、逐项目兼容：真正花时间的地方

下面按问题类型和项目说明关键改动。这里保留错误发生的上下文，因为以后换显卡、升级 PyTorch 或拉取新分支时，这些现象比一份静态 requirements 更有参考价值。

### 12.1 OpenPCDet：单 voxel 时 `squeeze()` 把特征维度挤没了

OpenPCDet 的 PointPillar 在 aiQiiDataset 上先跑通，但换到极稀疏的 MMAUD 后，第一批就报：

```text
IndexError: too many indices for tensor of dimension 1
```

错误出现在 `pointpillar_scatter.py`：代码期待 `pillar_features` 是 `[M, C]`，实际却变成了一维 `[C]`。往前追到 VFE，原代码使用：

```python
features = features.squeeze()
```

正常激光雷达一帧有成千上万个 voxel，`M` 基本不会等于 1，所以这个 bug 长期藏着。MMAUD 一帧可能只有一个有效 voxel，此时无参数的 `squeeze()` 会把所有长度为 1 的维度都删掉。正确写法是只删 PFN 保留的那个固定维度：

```python
features = features.squeeze(dim=1)
```

同类修复也应用到了 InterFusion、PFA-NET、DSVT、VoxelNeXt 和 CenterPoint 对应的 pillar encoder。这个补丁很小，却是毫米波稀疏数据适配中最典型的一类问题：原模型不是逻辑上不支持稀疏点云，而是实现中偷偷假设“有效单元数量一定大于 1”。

### 12.2 InterFusion 与 PFA-NET：先把老 PCDet 带到 spconv 2.x

InterFusion 分支里有几处与数据适配无关、但会直接阻止启动的问题。训练脚本原先硬编码：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
```

在单卡 WSL 机器上这等于主动隐藏唯一的 GPU。修改后保留用户设置：

```python
gpu_list = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
```

源码中还有断开的三元表达式导致 `IndentationError`，以及 `pointpillar_scatter.py` 中残留的一行不完整代码 `matrix =`。这些都先做最小清理。这里的原则是只修复确定的语法和环境假设，不借机大改模型结构。

InterFusion 与 PFA-NET 都使用旧 PCDet 数据处理器。spconv 1.x 常见接口是 `VoxelGenerator.generate(points)`，spconv 2.x 则提供 `Point2VoxelCPU3d.point_to_voxel(tensorview)`。为了同一份代码兼容不同 spconv，增加了 wrapper：

```python
class VoxelGeneratorWrapper:
    def __init__(self, vsize_xyz, coors_range_xyz,
                 num_point_features, max_num_points_per_voxel,
                 max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except ImportError:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except ImportError:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2
```

spconv 2.x 路径通过 `cumm.tensorview` 转换：

```python
tv_voxels, tv_coordinates, tv_num_points = \
    self._voxel_generator.point_to_voxel(tv.from_numpy(points))

return tv_voxels.numpy(), tv_coordinates.numpy(), tv_num_points.numpy()
```

稀疏卷积模块的导入也从旧写法：

```python
import spconv
```

改为 spconv 2.x 的 PyTorch 命名空间：

```python
from spconv import pytorch as spconv
```

这组修改覆盖 backbone、UNet 和 PartA2 head。它不是“改个 import 就完事”，voxel generator 的返回类型也必须一起适配，否则数据预处理阶段仍然会失败。

### 12.3 Python 3.10：`collections.Iterable` 已经不能再拖了

几个老项目的 fastai optimizer 仍然写：

```python
from collections import Iterable
```

在 Python 3.10 中会报：

```text
ImportError: cannot import name 'Iterable' from 'collections'
```

统一改成：

```python
from collections.abc import Iterable
```

CenterPoint 和 PillarNet-LTS 的 optimizer 代码也有同类修改。这种兼容补丁没有必要通过降 Python 规避，因为标准库迁移位置非常明确。

### 12.4 PyTorch 2.6 以后：checkpoint 默认 `weights_only=True`

新 PyTorch 改变了 `torch.load` 的默认安全策略。老仓库保存的 checkpoint 不只有纯 tensor，还包含 epoch、优化器状态、版本信息等结构。直接加载可能被新的 `weights_only` 行为拦住。

在确认 checkpoint 来自本地训练、来源可信的前提下，加载位置显式写成：

```python
checkpoint = torch.load(
    filename,
    map_location=loc_type,
    weights_only=False,
)
```

模型 checkpoint、预训练 checkpoint 和 optimizer checkpoint 都需要一致处理；CenterPoint/PillarNet-LTS 的 Det3D checkpoint 工具也做了相同修改。这里不能只改一个入口，否则首次训练能保存，恢复训练或自动评估时仍会报错。

### 12.5 DSVT：没有 `torch_scatter` 时使用原生 PyTorch fallback

DSVT 的动态 pillar VFE 依赖 `torch_scatter.scatter_mean` 和 `scatter_max`。在最新 PyTorch/CUDA/`sm_120` 组合下，为一个小依赖再找完全匹配的第三方二进制包并不划算，因此实现了原生 PyTorch fallback。

均值聚合使用 `index_add_`：

```python
def scatter_mean(src, index, dim=0):
    if torch_scatter is not None:
        return torch_scatter.scatter_mean(src, index, dim=dim)

    dim_size = int(index.max().item()) + 1 if index.numel() else 0
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)

    count = src.new_zeros((dim_size,))
    count.index_add_(
        0, index,
        torch.ones(index.shape[0], dtype=src.dtype, device=src.device),
    )
    return out / count.clamp_min(1).view(-1, 1)
```

最大值聚合使用 PyTorch 的 `scatter_reduce_`：

```python
out.scatter_reduce_(
    0,
    expanded_index,
    src,
    reduce="amax",
    include_self=True,
)
```

这样既保留了已安装 `torch_scatter` 时的原实现，也让当前环境不依赖它。fallback 的重点不是追求比专用库更快，而是先保证功能正确和版本可维护。

DSVT 的 TensorRT wrapper 也改为可选导入。普通 PyTorch 训练并不需要 TensorRT，不应该因为没有部署依赖就阻止整个 backbone import：

```python
try:
    from ..model_utils.tensorrt_utils.trtwrapper import TRTWrapper
except ImportError:
    TRTWrapper = None
```

### 12.6 VoxelNeXt：候选 voxel 少于 K 时，top-k 解码不能假装数量足够

VoxelNeXt 在训练阶段已经完成，进入验证后却报：

```text
RuntimeError: shape '[1, 500]' is invalid for input of size 154
```

原因是解码器配置 `MAX_OBJ_PER_SAMPLE=500`，原实现后面固定按 K=500 reshape；但稀疏毫米波点云可能总共只有 154 个候选 voxel。另一次 MMAUD batch 中，两个样本分别只有 29 和 33 个候选，直接 `torch.stack` 又因为长度不同失败。

修复分三步：

1. 每个样本实际取 `min(K, candidate_count)`；
2. 一个 batch 内把不同长度 padding 到本批最大长度；
3. 后续 reshape 使用 `scores.shape[1]`，不再使用配置中的旧 K。

关键代码为：

```python
max_len = max(x.shape[0] for x in topk_inds_list)

def pad_first_dim(tensor, value):
    if tensor.shape[0] == max_len:
        return tensor
    pad_shape = (max_len - tensor.shape[0],) + tensor.shape[1:]
    return torch.cat(
        [tensor, tensor.new_full(pad_shape, value)],
        dim=0,
    )

topk_score = torch.stack([
    pad_first_dim(x, -torch.inf) for x in topk_score_list
])
topk_inds = torch.stack([
    pad_first_dim(x, 0) for x in topk_inds_list
])
```

分数用负无穷 padding，后续阈值过滤时不会变成假阳性；索引和类别用 0 占位。然后让后续解码使用动态 K：

```python
scores, inds, class_ids = _topk_1d(...)
K = scores.shape[1]
```

这才是对极稀疏输入的真实支持，而不是把 `MAX_OBJ_PER_SAMPLE` 临时调小到某个碰巧能跑的数。

### 12.7 CenterPoint 与 PillarNet-LTS：Det3D 默认假设框里带速度

Det3D 原始管线主要围绕 nuScenes 和 Waymo，标签通常不只是 7 维框，还包含速度。当前毫米波数据的目标框是：

```text
[x, y, z, dx, dy, dz, yaw]
```

加上类别 id 后是 8 维，不是带 `vx, vy` 的 10 维。因此 label assigner 要对 `MMRadarDataset` 单独处理：

```python
if res["type"] == "MMRadarDataset":
    anno_box = np.zeros((max_objs, 8), dtype=np.float32)
else:
    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
```

CenterHead 的回归目标也不再拼速度：

```python
anno_box[new_idx] = np.concatenate(
    (
        ct - (x, y),
        z,
        np.log(gt_box[3:6]),
        np.sin(rot),
        np.cos(rot),
    ),
    axis=None,
)
```

数据加载 pipeline 同时加入 `.npy`/`.bin` 双格式读取和 MMAUD 坐标转换。可选依赖 `pycocotools` 改为容错导入，因为纯点云训练不需要 COCO mask，不应该因为没有图像分割依赖就卡住。

PillarNet-LTS 还暴露了一个日志聚合问题。它的 `loc_loss_elem` 是列表，日志缓冲区把多次记录直接 `np.array` 后求平均。不同记录嵌套形状不完全一致时会报：

```text
TypeError: unsupported operand type(s) for /: 'list' and 'int'
```

修复后的逻辑先尝试转为 `float64` 数组；如果确实是规则的多维数值，就按 batch 权重广播求平均；如果是 ragged 结构，则保留最后一个可展示值，不让日志系统打断训练。训练核心已经正常前向反向，却被打印日志杀死，是很亏的一种失败，所以训练框架外围也必须放进 smoke 范围。

### 12.8 MMAUD 初次打通时关闭几何增强

aiQiiDataset 的 smoke/full 配置保留了 y 轴翻转、小角度旋转和 0.95～1.05 缩放：

```yaml
DATA_AUGMENTOR:
  AUG_CONFIG_LIST:
    - NAME: random_world_flip
      ALONG_AXIS_LIST: ["y"]
    - NAME: random_world_rotation
      WORLD_ROT_ANGLE: [-0.3925, 0.3925]
    - NAME: random_world_scaling
      WORLD_SCALE_RANGE: [0.95, 1.05]
```

MMAUD 初始测试中则关闭几何增强：

```yaml
DATA_AUGMENTOR:
  AUG_CONFIG_LIST: []
```

Det3D 配置对应 `no_augmentation=True`。原因不是说 MMAUD 永远不能增强，而是它极度稀疏，且部分目标接近统一范围边界。初次打通时先消除“旋转后仅有的一两个点离开范围”这种额外变量。等完整训练稳定后，可以逐项恢复增强并做消融，不能一上来把 KITTI/nuScenes 的增强强度照搬过来。

## 十三、smoke 配置与正式配置为什么分开

每个项目、每套数据都有 smoke 和 full 两份入口，但 full 不复制整份模型。smoke 使用均匀抽取的 64/32 样本、`workers=0` 和 1 epoch；full 只替换完整 info，默认训练 80 epoch。

这种分层有三个好处：

1. 改模型结构时先跑几十秒的 smoke，不用等完整 dataloader 和长训练。
2. smoke 与 full 共享模型主体，避免“测试配置能跑，正式配置是另一份陈旧 YAML”。
3. 出现错误时日志短、样本可重复，更适合定位边界帧。

PCDet 项目 smoke 实际由统一脚本强制 `--epochs 1 --workers 0`。Det3D 配置中直接设置：

```python
total_epochs = 1
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    # ...
)
```

full 配置继承 smoke 后只改：

```python
train_anno = data_root + "/mmradar_det3d_infos_train.pkl"
val_anno = data_root + "/mmradar_det3d_infos_val.pkl"
total_epochs = 80
```

## 十四、自动跑完 14 组实验，而不是手工记忆命令

七个项目有两种训练入口、七个环境和十四个配置，手工逐条敲很容易用错 Python。于是增加了一个实验清单，每个实验明确记录：

```python
Experiment(
    name="DSVT_mmaud",
    kind="pcdet",
    project="DSVT",
    env="DSVT",
    cfg="cfgs/mmradar_models/dsvt_mmaud_smoke.yaml",
    dataset="mmaud",
    extra_tag="smoke_mmaud_v1",
    batch_size="2",
)
```

脚本根据 `kind` 生成 PCDet 或 Det3D 命令，为每个进程写入准确的 `PYTHONPATH`，将 stdout/stderr 单独保存，并在任一实验失败时停止。最后输出 JSON 状态，包含 return code、耗时和日志路径。

完整复测命令：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 \
  /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_smoke_train_wsl.py
```

只复测指定实验：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 \
  /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_smoke_train_wsl.py \
  --only OpenPCDet_aiqii VoxelNeXt_mmaud
```

日志位于：

```text
E:\Scholar\mmradarDetect\environment\smoke_logs
```

每次失败都留下了现场。比如 OpenPCDet 的 1D pillar feature、DSVT 缺少 `torch_scatter`、DSVT 上采样尺寸不一致、VoxelNeXt 的 154/500 reshape、PillarNet-LTS 的 ragged log buffer，都是靠这套短链路逐个暴露的。修完以后只重跑失败项，最终再看十四项全通过，比“改一堆代码后直接开 80 epoch”踏实得多。

## 十五、正式训练命令

下面列出当前代码中的正式入口。完整配置默认 80 epoch，建议第一次先把 epoch 改成 5，观察 loss、显存、数据吞吐和验证预测，再决定是否拉满。

### OpenPCDet、InterFusion 与 PFA-NET

OpenPCDet + aiQiiDataset：

```bash
cd /mnt/e/Scholar/mmradarDetect/OpenPCDet/tools
/home/yuehui/miniforge3/envs/PointPillar/bin/python train.py \
  --cfg_file cfgs/mmradar_models/pointpillar_aiqii_full.yaml \
  --workers 4 --batch_size 2 --extra_tag full_aiqii
```

MMAUD：

```bash
/home/yuehui/miniforge3/envs/PointPillar/bin/python train.py \
  --cfg_file cfgs/mmradar_models/pointpillar_mmaud_full.yaml \
  --workers 4 --batch_size 2 --extra_tag full_mmaud
```

InterFusion 和 PFA-NET 的命令形式相同，只需进入各自 `tools` 目录并使用 `InterFusion` 或 `PFANet` 环境。

### DSVT

```bash
cd /mnt/e/Scholar/mmradarDetect/DSVT/tools
/home/yuehui/miniforge3/envs/DSVT/bin/python train.py \
  --cfg_file cfgs/mmradar_models/dsvt_aiqii_full.yaml \
  --workers 4 --batch_size 1 --extra_tag full_aiqii
```

MMAUD 可从 batch size 2 起步：

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

MMAUD：

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

MMAUD 只替换配置与输出目录：

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

## 十六、如何判断“训练跑通”不是假象

研究代码能打印进度条，不等于实验已经可信。通常按下面的顺序检查。

第一，数据层。随机抽几帧，把转换后的点坐标范围、框中心范围、框尺寸和点数打印出来。MMAUD 必须确认点与框同时执行 camera-to-lidar；aiQiiDataset 不应被重复转换。再检查 mask 之后每帧还剩多少点、目标框是否仍在范围内。

第二，张量层。记录 voxel 数、pillar feature 形状、BEV feature map 尺寸和 head 输出尺寸。尤其注意 batch size 1、单 voxel、空预测和不同候选数量的情况。激光雷达模型常在大样本上默认这些维度“肯定不为 1”，毫米波数据正好会把假设掀出来。

第三，数值层。前几个 iteration 的总 loss 和各子项应该是有限数，不出现 `NaN/Inf`。不同模型的绝对 loss 不能横向直接比较，PillarNet 的联合损失比 PointPillar 大并不自动意味着异常。更值得看的是同一模型连续若干 epoch 的趋势、正样本数，以及定位子项是否完全不变化。

第四，保存与恢复。smoke 中必须实际写 checkpoint，再从磁盘加载做一次验证。只训练不加载，会漏掉 PyTorch 2.6+ 的 `weights_only`、版本字段和 optimizer state 问题。

第五，预测几何。抽一批预测框，确认 x/y/z 范围、尺寸和 yaw 没有明显错位。一个 epoch 的 recall 为 0 可以接受；预测中心全部跑到范围边界外，就不能用“还没收敛”解释。

## 十七、后续正式实验怎么做更稳

当前版本已经把工程通路打通，但还没有替代模型选择和超参数研究。下一阶段可以按以下节奏推进。

先让七个模型分别在两套数据上跑 5 epoch，保存每个 epoch 的训练 loss、验证中心距离、0.5/1/2/4 米 recall 与 precision、显存峰值和单 epoch 用时。这个阶段主要找不稳定配置，不急着排名。

然后单独处理两套数据的差异。aiQiiDataset 目标尺寸较小、每帧点数相对多，可以尝试适度减小 xy voxel、恢复旋转与缩放增强，并调整小目标高斯半径。MMAUD 极稀疏，优先研究多帧累积、点特征归一化和更克制的数据增强，而不是简单把网络堆大。

再往后才是公平比较。七个项目的参数量、head、voxel size 和优化器并不完全相同。如果要写实验表格，需要固定训练/验证划分、输入范围、输入帧数、类别定义、epoch 或总 iteration、评价指标和随机种子。否则“同样 80 epoch”不一定代表同等训练预算。

最后，给每次可复现实验保留三样东西：配置文件、Git commit 和运行命令。模型权重、日志、`.npy/.pkl` 数据和编译产物不要提交进 Git。当前仓库已经忽略 `build/`、`*.so`、`output/`、`work_dirs/`、日志和 TensorBoard 文件，后面还应继续避免把原始数据与大 checkpoint 混进源码版本。

## 十八、这次迁移最值得留下的几条经验

第一，新显卡适配不能只看 `nvidia-smi`。驱动能看到 GPU，只说明第一层通了；PyTorch wheel、CUDA Toolkit、扩展编译架构和第三方稀疏卷积仍然要分别验证。对于 RTX 5070 Ti，这次真正贯穿编译过程的是 CUDA 12.8 与 `TORCH_CUDA_ARCH_LIST=12.0`。

第二，多个研究仓库不要共用一个 editable Python 环境。七个项目看上去都叫 3D detection，内部却有不同年代的 `pcdet`、`det3d` 和 CUDA ops。用统一种子环境克隆，再各自编译，是速度和隔离之间比较舒服的折中。

第三，稀疏毫米波数据会放大代码里的隐含假设。无参数 `squeeze()`、固定 K 的 top-k、每个样本候选数相同、日志向量等长、每帧一定有很多 voxel，这些在 KITTI/Waymo 上不容易触发，在 MMAUD 上几分钟内就会全部出现。不要把这些错误简单归因于“数据太少”，它们本质上是实现没有覆盖边界输入。

第四，数据适配的核心是坐标和监督，不是目录结构。点云能读进来只是开始。坐标轴、框尺寸顺序、yaw、速度维度、anchor、输入范围和评价坐标系必须成套一致。

第五，smoke 测试要覆盖训练后的自动评估。很多兼容问题只在 checkpoint 重载和 decode 时出现，VoxelNeXt 就是典型：训练完整结束，第一帧验证才因为固定 K reshape 崩溃。只看训练进度条会误判为已经跑通。

第六，公共逻辑应该抽出来，项目差异留在配置和薄适配层。两套数据的读取、坐标变换、info 生成和中心距离评价只有一份实现，后面修正一处即可同步七个项目。否则同一个 yaw 或坐标 bug 很可能在七份复制代码里留下六个版本。

## 十九、补上周期验证与七模型连续调度

最初的 full 配置只负责训练 80 epoch，CenterPoint/PillarNet-LTS 的工作流甚至只有 `train`，因此日志里只有 loss，没有检测指标。后续把两类框架统一成了相同实验节奏：每训练 5 epoch，立即在完整 val 上验证一次，最终 epoch 无条件验证。

PCDet 五个项目在训练循环中调用当前模型做评估；CenterPoint/PillarNet-LTS 使用：

```python
workflow = [("train", 5), ("val", 1)]
checkpoint_config = dict(interval=10, max_keep_ckpts=10)
```

每次验证都会把完整结果写进 `periodic_metrics/epoch_XXX.json`，并追加到 `metrics_history.json`。指标包括中心距离、BEV IoU、3D IoU 对应的 TP/FP/FN、Precision、Recall、F1 和 AP，以及中心、尺寸、yaw 和置信度误差。checkpoint 不再每轮保存：指标每 5 epoch 计算，但正式权重每 10 epoch 保存，80 epoch 只得到 8 个代表性 checkpoint；未保存节点的 JSON 指标仍然完整保留。

七模型按数据集连续训练使用统一入口：

```powershell
wsl.exe -d Ubuntu-24.04 -- python3 /mnt/e/Scholar/mmradarDetect/environment/run_mmradar_full_suite_wsl.py `
  --dataset mmaud --run-tag mmaud_v1 --workers 4 --retries 2
```

把 `mmaud` 换成 `aiqii` 即可运行另一套数据。脚本会按顺序执行七个项目，失败后从最近 checkpoint 重试；重试仍失败则记录日志并继续下一个。电脑或 WSL 中断后，重新执行相同命令和 `run-tag`，已完成模型会被跳过，未完成模型继续训练。最终七模型关键指标汇总到 `environment/full_runs/<dataset>/<run-tag>/ALL_MODELS_METRICS.md`，全部原始指标保存在同目录 JSON 及各模型的 `metrics_history.json` 中。

## 结语

把一套 3D 检测代码装起来，和把七套不同年代的代码放到同一台 50 系显卡上、用同一套数据协议训练，是两件完全不同的事。前者常常止于 import 和 demo，后者必须处理环境边界、CUDA ABI、框架 API、数据几何和极端稀疏输入。

这次最终留下的不是一台“碰巧能跑”的电脑，而是一套相对清楚的结构：七个独立环境、统一 CUDA 12.8 底座、两类框架共享的毫米波数据层、smoke/full 分离的配置，以及能重放十四组实验的脚本。后面无论是继续调 PointPillar，还是比较 DSVT、VoxelNeXt 和 PillarNet，至少可以把精力放在模型和数据上，不必每次从 `ModuleNotFoundError` 和 CUDA 编译重新开始。

当前源码版本以 Git 提交 `aa605ba`（“环境与毫米波数据集接入版本”）为界；最初下载的源码是 `fec9f4f`。如果后续升级 PyTorch、替换 spconv 或调整数据标签，可以直接对照这两个版本查看所有兼容改动，也能在新实验出现异常时快速判断：到底是模型变化，还是环境和数据入口发生了变化。
