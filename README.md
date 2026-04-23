# 基于 MindSpore 的图像超分辨率实验报告

## 1. 实验目的

本实验旨在使用 MindSpore 实现单幅图像超分辨率（Single Image Super-Resolution, SISR）方法，在 DIV2K 数据集上完成训练，并在 Set5 数据集上完成 ×4 放大倍率测试。按照题目要求，测试阶段先对高分辨率图像进行 Bicubic 下采样得到低分辨率输入，再使用超分辨率模型重建高分辨率图像，最后将重建结果与真实原图进行比较，计算 PSNR 和 SSIM 指标。

本实验的具体目标如下：

1. 掌握图像超分辨率任务的基本流程；
2. 熟悉 MindSpore 下超分辨率模型的实现与训练方法；
3. 理解 Bicubic 下采样、超分辨率重建、PSNR/SSIM 评价之间的关系；
4. 对比轻量卷积模型与 Transformer 风格模型在课程作业场景下的表现。

---

## 2. 实验原理

### 2.1 图像超分辨率任务简介

图像超分辨率是指从低分辨率图像恢复高分辨率图像的过程。设原始高分辨率图像为 $I_{HR}$，由 Bicubic 下采样得到低分辨率图像 $I_{LR}$，超分辨率模型为 $F(\cdot)$，则重建过程可表示为：

$$
\hat{I}_{HR} = F(I_{LR})
$$

其中 $\hat{I}_{HR}$ 为模型输出的超分辨率图像。

本实验采用的退化方式为题目要求的 **Bicubic ×4 下采样**。该方式同时也被用作基线方法：即先将 HR 图像下采样为 LR，再直接使用 Bicubic 插值放大回原尺寸，并与超分模型结果进行比较。

### 2.2 ESPCN 网络原理

ESPCN（Efficient Sub-Pixel Convolutional Network）是经典的轻量级超分模型。其主要思想是：

1. 先在低分辨率空间进行卷积特征提取；
2. 最后一层输出 $3 \times r^2$ 个通道；
3. 使用 `DepthToSpace`（子像素重排）实现上采样。

本实验中 ESPCN 的结构配置为：

- Conv1：`3 → 64`，卷积核 `5×5`
- Conv2：`64 → 32`，卷积核 `3×3`
- Conv3：`32 → 48`，卷积核 `3×3`
- Pixel Shuffle：放大倍率 `×4`


### 2.3 SwinIR 原理与实验尝试

为了进一步提升结果，本实验还尝试引入 **SwinIR 风格模型**。SwinIR 基于 Swin Transformer，将窗口注意力机制用于图像恢复任务，能够在更大范围内建模图像上下文关系。与浅层卷积结构相比，Transformer 风格模型理论上具有更强的纹理建模与细节恢复能力。

在本项目中实现的是 **简化版 SwinIR**，并非官方完整训练框架，因此其结果主要用于补充性对比实验。

### 2.4 损失函数与优化方式

实验中主要采用两种训练设置：

- **ESPCN**：采用 `MSELoss` 训练；
- **SwinIR 尝试版**：采用 `L1Loss` 训练。

优化器均使用 Adam。ESPCN 的学习率设置为 `1e-3`，SwinIR 尝试版学习率设置为 `2e-4`。

---

## 3. 数据集与评价指标

### 3.1 DIV2K 数据集

DIV2K 是图像超分辨率领域常用的高质量数据集。本实验使用：

- `DIV2K_train_HR`：用于模型训练；
- `DIV2K_valid_HR`：用于训练过程中的验证与开发集评估。

训练时程序直接读取 HR 图像，随机裁剪 HR patch，并在程序内部通过 Bicubic 方法动态生成对应 LR patch。

### 3.2 Set5 数据集

Set5 是经典小规模超分辨率测试集，共包含 5 张图像：

- baby
- bird
- butterfly
- head
- woman

本实验在 Set5 上进行最终 ×4 测试。流程如下：

1. 读取 Set5 高分辨率图像；
2. 使用 Bicubic 方法进行 ×4 下采样，生成低分辨率输入；
3. 使用训练好的模型进行重建；
4. 将重建图像与真实 HR 图像比较，计算 PSNR 和 SSIM。

### 3.3 评价指标

#### （1）PSNR

峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）定义为：

$$
PSNR = 10 \log_{10}\left(\frac{MAX^2}{MSE}\right)
$$

其中 $MAX$ 为像素最大值，8 bit 图像通常取 255。PSNR 越高，说明重建图像与原图越接近。

#### （2）SSIM

结构相似性（Structural Similarity Index, SSIM）用于从亮度、对比度和结构三个方面衡量两幅图像的相似程度。SSIM 越接近 1，说明重建质量越好。

本实验后期评估采用更接近标准 SR 设定的方式：**Y 通道 + 按倍率裁边**。

---

## 4. 实验环境与参数设置

### 4.1 实验环境

- Python 3.9 / 3.10+
- MindSpore
- NumPy
- Pillow
- scikit-image

### 4.2 ESPCN 训练参数

| 参数       |               取值 |
| ---------- | -----------------: |
| 放大倍率   |                 ×4 |
| 训练数据集 |     DIV2K_train_HR |
| 验证数据集 |     DIV2K_valid_HR |
| 测试数据集 |               Set5 |
| 模型       |              ESPCN |
| Patch Size |          128 / 192 |
| Batch Size |                 16 |
| Epochs     | 30（最终可用版本） |
| Repeat     |                  5 |
| 学习率     |               1e-3 |
| 优化器     |               Adam |
| 损失函数   |            MSELoss |

### 4.3 SwinIR 尝试版参数

| 参数       |           取值 |
| ---------- | -------------: |
| 放大倍率   |             ×4 |
| 训练数据集 | DIV2K_train_HR |
| 验证数据集 | DIV2K_valid_HR |
| 测试数据集 |           Set5 |
| 模型       |  简化版 SwinIR |
| Patch Size |            192 |
| Batch Size |             16 |
| Epochs     |             60|
| Repeat     |              5 |
| 学习率     |           2e-4 |
| 优化器     |           Adam |
| 损失函数   |         L1Loss |



---

## 5. 项目目录结构说明

当前项目目录如下：

```text
|-- README.md                        # 项目说明与实验报告

|-- SwinIR_run_eval.sh              # 运行 SwinIR 模型在验证集或测试集上的评测脚本
|-- SwinIR_run_eval_Set5.sh         # 运行 SwinIR 模型在 Set5 数据集上的评测脚本
|-- run_SwinIR.sh                   # 启动 SwinIR 训练或实验流程的脚本
|-- run_bicubic.sh                  # 运行 Bicubic 基线评测的脚本
|-- run_eval.sh                     # 运行 ESPCN 模型评测的脚本
|-- run_train.sh                    # 运行 ESPCN 模型训练的脚本

|-- SwinIR_train.py                 # SwinIR 模型训练主脚本
|-- eval.py                         # ESPCN 模型评测脚本，用于在指定数据集上计算 PSNR/SSIM
|-- eval_bicubic.py                 # Bicubic 插值基线评测脚本
|-- infer.py                        # 单张图像推理脚本，用于生成超分辨率结果图
`-- train.py                        # ESPCN 模型训练主脚本

|-- data                            # 数据集目录
|   |-- DIV2K_train_HR              # DIV2K 训练集高分辨率图像
|   |-- DIV2K_valid_HR              # DIV2K 验证集高分辨率图像
|   `-- Set5                        # Set5 测试集高分辨率图像

|-- logs                            # 训练与评测日志目录
|   |-- SwinIR_train.log            # SwinIR 模型训练日志
|   |-- eval_Set5.log               # ESPCN 在 Set5 上评测的日志
|   |-- eval_Set5_SwinIR.log        # SwinIR 在 Set5 上评测的日志
|   |-- eval_SwinIR.log             # SwinIR 在 DIV2K 验证集上评测的日志
|   |-- eval_bicubic.log            # Bicubic 基线评测日志
|   |-- eval_bicubic1.log           # 另一份 Bicubic 评测日志
|   |-- eval_bicubic_Set5.log       # Bicubic 在 Set5 上评测的日志
|   `-- train.log                   # ESPCN 模型训练日志

|-- outputs                         # 模型训练结果、评测结果和生成图像的输出目录

`-- src                            # 项目核心源码目录
    |-- SwinIR.py                   # SwinIR 风格模型定义文件
    |-- dataset.py                  # 当前使用的数据集加载与预处理代码
    |-- dataset_old.py              # 旧版数据集加载代码备份
    |-- model.py                    # ESPCN 模型定义文件
    |-- utils.py                    # 工具函数文件，包含图像转换、评价指标、路径创建等功能
    `-- __pycache__                 # src 目录下的 Python 缓存目录
```



---

## 6. 实验流程

### 6.1 ESPCN 训练流程

1. 读取 DIV2K 高分辨率图像；
2. 随机裁剪 HR patch；
3. 使用 Bicubic 下采样得到 LR patch；
4. 输入 ESPCN 网络进行前向传播；
5. 计算 MSE 损失；
6. 使用 Adam 更新参数；
7. 根据验证集结果保存最优 checkpoint。

### 6.2 SwinIR 尝试版流程

1. 使用与 ESPCN 相同的数据构造方式；
2. 使用简化版 SwinIR 风格模型替代 ESPCN；
3. 采用更小学习率和 L1 损失训练；
4. 在 DIV2K_valid 与 Set5 上进行结果比较。

### 6.3 测试流程

1. 读取验证集或 Set5 高分辨率图像；
2. 使用 Bicubic 生成 ×4 低分辨率图像；
3. 输入模型进行重建；
4. 保存 SR 图像；
5. 计算 PSNR 和 SSIM；
6. 统计平均指标。

---

## 7. 实验结果

### 7.1 Set5 上 Bicubic 基线结果

| 图像名称   |     PSNR/dB |         SSIM |
| ---------- | ----------: | -----------: |
| baby       |     31.7826 |     0.867668 |
| bird       |     30.1835 |     0.881345 |
| butterfly  |     22.1007 |     0.752490 |
| head       |     31.6144 |     0.771507 |
| woman      |     26.4650 |     0.842189 |
| **平均值** | **28.4292** | **0.823040** |

### 7.2 Set5 上 ESPCN 结果

| 图像名称   |     PSNR/dB |         SSIM |
| ---------- | ----------: | -----------: |
| baby       |     32.6595 |     0.890611 |
| bird       |     30.9388 |     0.894209 |
| butterfly  |     23.2355 |     0.777429 |
| head       |     31.9976 |     0.792461 |
| woman      |     27.5497 |     0.865155 |
| **平均值** | **29.2762** | **0.843973** |

相较于 Bicubic，ESPCN 在 Set5 上的提升为：

- **PSNR 提升：+0.8470 dB**
- **SSIM 提升：+0.0209**

说明 ESPCN 能够稳定优于传统插值方法。

### 7.3 DIV2K_valid 上 ESPCN 结果

- **Average PSNR：28.6729 dB**
- **Average SSIM：0.808710**

对应 Bicubic 基线为：

- **Average PSNR：28.1028 dB**
- **Average SSIM：0.785755**

相较于 Bicubic，ESPCN 在 DIV2K_valid 上提升：

- **PSNR：+0.5701 dB**
- **SSIM：+0.0230**

### 7.4 Set5 上 SwinIR 尝试版结果

| 图像名称   |     PSNR/dB |         SSIM |
| ---------- | ----------: | -----------: |
| baby       |     32.6430 |     0.888082 |
| bird       |     30.8706 |     0.892878 |
| butterfly  |     22.9225 |     0.769397 |
| head       |     32.0042 |     0.790885 |
| woman      |     27.3947 |     0.860143 |
| **平均值** | **29.1670** | **0.840277** |

### 7.5 DIV2K_valid 上 SwinIR 尝试版结果

- **Average PSNR：28.6009 dB**
- **Average SSIM：0.806591**

### 7.6 模型结果对比

#### （1）Set5 对比

| 方法          | Average PSNR/dB | Average SSIM |
| ------------- | --------------: | -----------: |
| Bicubic       |         28.4292 |     0.823040 |
| ESPCN         |     **29.2762** | **0.843973** |
| SwinIR 尝试版 |         29.1670 |     0.840277 |

#### （2）DIV2K_valid 对比

| 方法          | Average PSNR/dB | Average SSIM |
| ------------- | --------------: | -----------: |
| Bicubic       |         28.1028 |     0.785755 |
| ESPCN         |     **28.6729** | **0.808710** |
| SwinIR 尝试版 |         28.6009 |     0.806591 |

从结果看，当前实验条件下，**ESPCN 略优于简化版 SwinIR 尝试版**。

---

## 8. 结果分析

### 8.1 ESPCN 的有效性

ESPCN 在 Set5 和 DIV2K_valid 上均稳定优于 Bicubic，说明该模型已经成功学习到了从低分辨率到高分辨率的映射关系。虽然从视觉上看提升并不一定十分夸张，但从 PSNR 和 SSIM 指标来看，模型确实恢复了更多细节与结构信息。


### 8.2 SwinIR 尝试版未超过 ESPCN 的原因

理论上，Transformer 风格模型具有更强的上下文建模能力，但本实验中的 SwinIR 尝试版结果并未超过 ESPCN。主要原因如下：

1. 当前实现为**简化版**，并非官方完整 SwinIR 框架；
2. 训练轮数较少，训练预算不足，复杂模型尚未充分收敛；
3. SwinIR 风格模型参数更多，对学习率、训练时长、显存和数据配置更加敏感；
4. 在课程作业的有限资源条件下，轻量级 ESPCN 反而更容易训练到较优结果。

因此，本实验将 SwinIR 结果作为**补充探索**，而将 ESPCN 作为最终主模型。

### 8.4 方法优缺点总结

#### ESPCN

**优点：**

- 结构简单，易于实现；
- 推理速度快；
- 在当前作业资源下训练更稳定；
- 最终结果优于 Bicubic，也优于本次 SwinIR 尝试版。

**缺点：**

- 模型较浅，复杂纹理恢复能力有限；
- 视觉观感提升不如更强的现代超分模型明显。

#### SwinIR 尝试版

**优点：**

- 引入 Transformer 风格结构，具有更强理论表达能力；
- 为进一步扩展实验提供了方向。

**缺点：**

- 当前简化实现较难充分发挥性能；
- 训练成本更高；
- 在当前实验预算下未超过 ESPCN。

---

## 9. 结论

本实验基于 MindSpore 实现了图像超分辨率系统，采用 DIV2K 数据集训练，并在 Set5 数据集上完成 ×4 放大倍率测试。实验中首先实现了 ESPCN 模型，并取得了较稳定的结果：在 Set5 上达到 **29.2762 dB / 0.843973**，明显优于 Bicubic 基线的 **28.4292 dB / 0.823040**。

在此基础上，我进一步尝试引入简化版 SwinIR 结构，希望利用 Transformer 风格模型进一步提升结果。但实验表明，在当前实现和训练预算下，SwinIR 尝试版并未超过 ESPCN，其 Set5 平均指标为 **29.1670 dB / 0.840277**。因此可以得出结论：对于当前课程作业环境和时间成本而言，**ESPCN 是更合适的最终方案**，而 SwinIR 尝试版则作为模型扩展与对比实验保留在报告中。

总体来看，本实验完整实现了：

- DIV2K 训练；
- Set5 测试；
- Bicubic 基线对比；
- ESPCN 与 SwinIR 两种结构实验；
- PSNR / SSIM 指标分析。


## 10. 参考文献

1. Shi W, Caballero J, Huszár F, et al. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. CVPR, 2016.
2. Liang J, Cao J, Sun G, et al. SwinIR: Image Restoration Using Swin Transformer. ICCVW, 2021.
3. Agustsson E, Timofte R. NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. CVPR Workshops, 2017.