# Show-o 训练与推理运行指南

本文档供学长参考，用于在集群上运行 Show-o 项目的训练和推理，拉取项目时需要拉取dev分支，里面包含了需要调试，希望能有所帮助~

---

## 项目执行目录

所有脚本均在以下目录执行：

```
/showO/Show-o/show-o2
```

---

## 一、环境准备

### 1.1 Conda 环境

Show-o 需要专用的 conda 环境，可以参考以下步骤创建：

1. 从 HuggingFace 下载 conda 环境包：
   > https://huggingface.co/datasets/xuyang2003/data_indoor （`conda` 目录下）

2. 将下载的压缩包上传到集群，然后在集群上解压：

   ```bash
   cd ~/anaconda/env
   mkdir showO
   cd showO
   sudo tar -zxvf show.tar.gz
   ```

3. 完成后即创建了名为 `showO` 的 conda 环境。

   > **参考教程**：[Conda 环境迁移方法一](https://blog.csdn.net/baidu_35692628/article/details/136519579?fromshare=blogdetail&sharetype=blogdetail&sharerId=136519579&sharerefer=PC&sharesource=xy2003king&sharefrom=from_link)

4. 验证环境路径 —— 在集群上运行以下命令，记下 `showO` 环境对应的实际目录：

   ```bash
   conda env list
   ```

   针对脚本中的 `PYTHON_EXECUTABLE` 和 `ACCELERATE_LAUNCH_MODULE` 里原有的 `/root/miniconda3/envs/showO`，麻烦替换为 `showO` 环境对应的实际目录（具体会在训练环节中说明）。

### 1.2 数据集

从 HuggingFace 下载数据集：

> https://huggingface.co/datasets/xuyang2003/data_indoor （下载 `data_indoor.zip`）

解压后，数据集目录结构应包含 `3dprint`、`3dpirnt2` 等多个物体子文件夹。请留意**包含这些子文件夹的那一层目录路径**，后续用于替换脚本中的 `TACTILE_DATA_ROOT` 变量。

### 1.3 模型

所有预训练模型存放在：

> https://huggingface.co/xuyang2003/tac_showO

麻烦创建一个 `models` 文件夹，将该仓库下的 **4 个模型** 全部下载放入其中。记下 `models` 文件夹的路径，后续用于替换脚本中的 `MODEL_ROOT` 变量。

---

## 二、训练

### 2.1 启动脚本

任选其一即可，两者等价：

| 类型 | 脚本路径 |
|------|----------|
| Bash | `show-o2/run_tactile_stage_one.sh` |
| Python | `show-o2/run_tactile_inference.py` |

### 2.2 需要替换的变量

打开 `run_tactile_stage_one.sh`，需要替换以下几项：

#### Conda 路径

将脚本中的 `/root/miniconda3/envs/showO` 替换为集群上 `showO` 环境的实际路径（通过 `conda env list` 查看即可）。涉及以下两个变量：

```bash
PYTHON_EXECUTABLE="/root/miniconda3/envs/showO/bin/python"
ACCELERATE_LAUNCH_MODULE="/root/miniconda3/envs/showO/lib/python3.10/site-packages/accelerate/commands/launch.py"
```

#### 数据集路径

```bash
TACTILE_DATA_ROOT=/defaultShare/data_indoor
```

麻烦替换为 **1.2 节** 中解压后的数据集目录路径（即包含 `3dprint`、`3dpirnt2` 等子文件夹的那一层）。

#### 模型路径

```bash
MODEL_ROOT=/defaultShare/models
```

麻烦替换为 **1.3 节** 中创建的 `models` 文件夹路径。

#### GPU 数量

```bash
export CUDA_VISIBLE_DEVICES=0,1
```

可以根据需要调整 GPU 编号及数量（如 `0,1,2,3` 表示使用 4 块 GPU）。

### 2.3 训练输出

训练完成后，麻烦学长将以下两个目录转发给我：

- `show-o2/wandb`
- `show-o2/outputs`

---

## 三、推理

### 3.1 启动脚本

任选其一即可，两者等价：

| 类型 | 脚本路径 |
|------|----------|
| Bash | `show-o2/run_tactile_inference.sh` |
| Python | `show-o2/run_tactile_inference.py` |

### 3.2 需要替换的变量

以下四个变量的替换方法与**训练完全相同**（参考 **2.2 节**）：

```bash
PYTHON_EXECUTABLE="/root/miniconda3/envs/showO/bin/python"
ACCELERATE_LAUNCH_MODULE="/root/miniconda3/envs/showO/lib/python3.10/site-packages/accelerate/commands/launch.py"
MODEL_ROOT=/defaultShare/models
TACTILE_DATA_ROOT=/defaultShare/data_indoor
```

#### 额外替换：Checkpoint 路径

推理还需要指定训练产出的 checkpoint：

```bash
STAGE1_CHECKPOINT=/Show-o/show-o2/outputs/showo2-1.5b-tactile-stage-1_video/checkpoint-45000/unwrapped_model
```

麻烦将其替换为训练时实际保存的 checkpoint 路径。

### 3.3 推理输出

推理完成后，麻烦学长将以下目录转发给我：

- `show-o2/Inference`

---

## 四、快速检查清单

- [ ] Conda 环境 `showO` 已创建并验证路径
- [ ] 数据集已下载解压，路径已替换
- [ ] 4 个模型已下载到 `models` 文件夹，路径已替换
- [ ] GPU 数量已按需设置
- [ ] 训练：`PYTHON_EXECUTABLE`、`ACCELERATE_LAUNCH_MODULE` 路径已替换
- [ ] 推理：额外替换 `STAGE1_CHECKPOINT` 为实际 checkpoint 路径
