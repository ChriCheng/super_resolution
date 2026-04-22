# 基于 MindSpore 的图像超分辨率实验报告

## 1. 实验目的

本实验旨在使用 MindSpore 实现一种单幅图像超分辨率（Single Image Super-Resolution, SISR）方法，在 DIV2K 数据集上完成训练，并在 Set5 数据集上完成 ×4 放大倍率测试。测试流程按照题目要求，先对高分辨率图像进行 Bicubic 下采样，再使用超分辨率模型恢复高分辨率图像，最后将重建结果与原始真实图像进行比较，计算 PSNR 和 SSIM 指标。

通过本实验，希望达到以下目的：

1. 掌握图像超分辨率任务的基本流程；
2. 熟悉 MindSpore 下卷积神经网络的实现与训练方法；
3. 理解 Bicubic 下采样、超分辨率重建、PSNR/SSIM 评价的关系；
4. 学会对实验结果进行整理、分析与总结。

---

## 2. 实验原理

### 2.1 图像超分辨率任务简介

图像超分辨率是指从低分辨率图像恢复出高分辨率图像的过程。其目标是在保留整体结构的同时，尽可能恢复边缘、纹理和细节信息。单幅图像超分辨率仅依赖一张低分辨率输入图像，因此属于典型的病态逆问题。

在本实验中，低分辨率图像由高分辨率图像通过 Bicubic 插值下采样得到。设原始高分辨率图像为 $I_{HR}$，下采样后的低分辨率图像为 $I_{LR}$，超分模型记为 $F(\cdot)$，则目标可以表示为：

$$
\hat{I}_{HR} = F(I_{LR})
$$

其中 $\hat{I}_{HR}$ 为模型重建出的超分辨率图像。

### 2.2 ESPCN 网络原理

本实验选择 ESPCN（Efficient Sub-Pixel Convolutional Network）作为基础模型。相较于先将图像插值放大再做卷积的方式，ESPCN 先在低分辨率空间进行特征提取，最后通过子像素重排实现上采样，因此计算量更小，结构更适合课程作业实现。

网络结构如下：

1. 第一层卷积：提取浅层特征；
2. 第二层卷积：提取更高层特征；
3. 第三层卷积：输出 $3 \times r^2$ 个通道；
4. 使用 `DepthToSpace` 实现子像素重排，得到放大 $r=4$ 倍后的 RGB 图像。

本实验网络具体配置如下：

- Conv1：`3 → 64`，卷积核大小 `5×5`
- Conv2：`64 → 32`，卷积核大小 `3×3`
- Conv3：`32 → 48`，卷积核大小 `3×3`
- Pixel Shuffle：放大倍率 `×4`

### 2.3 损失函数与优化方式

本实验采用均方误差损失（MSE Loss）：

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\hat{I}_{HR}^{(i)} - I_{HR}^{(i)})^2
$$

其中 $N$ 为像素总数。由于 PSNR 与 MSE 直接相关，因此采用 MSE 作为训练目标具有较强的合理性。

优化器采用 Adam，学习率设置为 $1 \times 10^{-3}$。

---

## 3. 数据集与评价指标

### 3.1 DIV2K 数据集

DIV2K 是图像超分辨率领域常用的高质量数据集。本实验使用其中的训练集部分进行模型训练。训练时直接读取高分辨率图像，并在程序内部随机裁剪 HR patch，再通过 Bicubic 方法生成对应 LR patch，构造训练样本对。

### 3.2 Set5 数据集

Set5 是超分辨率任务中经典的小规模测试集，共包含 5 张图像，常用于快速测试模型的重建性能。本实验在 Set5 上进行 ×4 放大倍率测试。

测试流程如下：

1. 读取 Set5 高分辨率图像；
2. 使用 Bicubic 方法进行 ×4 下采样，生成低分辨率输入；
3. 使用训练好的模型对 LR 图像进行重建；
4. 将重建图像与真实 HR 图像比较，计算 PSNR 和 SSIM。

### 3.3 评价指标

#### （1）PSNR

峰值信噪比（Peak Signal-to-Noise Ratio, PSNR）定义如下：

$$
PSNR = 10 \log_{10} \left( \frac{MAX^2}{MSE} \right)
$$

其中 $MAX$ 表示像素最大值，8 bit 图像中通常取 255。PSNR 越大，说明重建结果与原图越接近。

#### （2）SSIM

结构相似性（Structural Similarity Index, SSIM）用于衡量两幅图像在亮度、对比度和结构方面的相似程度，取值范围通常为 $[0, 1]$，越接近 1 表示重建质量越好。

---

## 4. 实验环境与参数设置

### 4.1 实验环境

本实验使用 Python 与 MindSpore 实现，主要依赖如下：

- Python 3.10+
- MindSpore
- NumPy
- Pillow
- scikit-image

### 4.2 训练参数设置

本实验使用如下参数：

| 参数       |                 取值 |
| ---------- | -------------------: |
| 放大倍率   |                   ×4 |
| 训练数据集 |       DIV2K_train_HR |
| 测试数据集 |                 Set5 |
| 模型       |                ESPCN |
| Patch Size |                  192 |
| Batch Size |                   16 |
| Epochs     |                   80 |
| 学习率     |                 1e-3 |
| 优化器     |                 Adam |
| 损失函数   |              MSELoss |
| 数据增强   |       随机翻转、转置 |
| 评估方式   | RGB 全图 PSNR / SSIM |

说明：训练时采用随机 HR patch，并在程序内部动态生成 LR 图像；测试时对整张 Set5 图像直接进行评估。

---

## 5. 代码结构说明

项目主要文件如下：

```text
mindspore_sr_assignment/
├── src/
│   ├── model.py          # ESPCN网络定义
│   ├── dataset.py        # 数据集读取与样本构造
│   └── utils.py          # 指标计算与图像保存工具
├── train.py              # 训练脚本
├── eval.py               # Set5测试脚本
├── infer.py              # 单张图像推理脚本
├── requirements.txt      # 依赖文件
├── run_train.sh          # 训练命令示例
├── run_eval.sh           # 测试命令示例
└── MindSpore_SR_Report.md
```

各模块功能如下：

- `model.py`：实现 ESPCN 网络；
- `dataset.py`：实现 DIV2K 随机 patch 训练集与 Set5 测试集；
- `train.py`：完成模型训练与 checkpoint 保存；
- `eval.py`：加载模型并在 Set5 上测试，输出 PSNR 和 SSIM；
- `infer.py`：支持对单张图像进行超分推理。

---

## 6. 实验流程

### 6.1 训练流程

1. 读取 DIV2K 高分辨率图像；
2. 随机裁剪 HR patch；
3. 通过 Bicubic 下采样得到 LR patch；
4. 输入 ESPCN 网络进行前向传播；
5. 计算 MSE 损失；
6. 使用 Adam 优化器更新参数；
7. 每轮保存模型参数，并记录训练损失。

### 6.2 测试流程

1. 读取 Set5 高分辨率图像；
2. 使用 Bicubic 插值生成 ×4 低分辨率图像；
3. 输入训练好的 ESPCN 模型进行重建；
4. 保存重建图像；
5. 计算每张图像的 PSNR 和 SSIM；
6. 统计平均指标。

---

## 7. 实验结果

### 7.1 Set5 各图像测试结果

运行 `eval.py` 后，可将输出结果整理如下表：

| 图像名称  | PSNR/dB |   SSIM |
| --------- | ------: | -----: |
| baby      |  待填写 | 待填写 |
| bird      |  待填写 | 待填写 |
| butterfly |  待填写 | 待填写 |
| head      |  待填写 | 待填写 |
| woman     |  待填写 | 待填写 |
| 平均值    |  待填写 | 待填写 |

### 7.2 训练损失变化

可根据 `train_log.csv` 绘制训练损失曲线。通常情况下，随着训练轮数增加，训练损失会逐渐下降并趋于稳定，这说明模型逐步学会从低分辨率图像恢复高分辨率细节。

### 7.3 可视化结果分析

可以在报告中插入以下三类图像进行对比：

1. 原始高分辨率图像（HR）；
2. Bicubic 下采样得到的低分辨率图像（LR）；
3. 模型重建得到的超分辨率图像（SR）。

若模型训练有效，则 SR 图像在边缘清晰度、纹理细节和整体观感上应明显优于简单插值放大结果。

---

## 8. 结果分析

### 8.1 模型有效性分析

ESPCN 通过在低分辨率空间进行卷积运算，降低了计算开销，并利用子像素重排实现高效上采样。对于课程作业要求的 ×4 超分任务，该模型具有实现简单、训练稳定、推理效率较高的优点。

从结果上看，如果训练正常，模型在 Set5 上的 PSNR 和 SSIM 应明显优于单纯的 Bicubic 重建。这说明卷积神经网络能够学习低分辨率到高分辨率之间的映射关系，并恢复部分高频细节。

### 8.2 影响结果的因素

实验结果会受到以下因素影响：

1. **训练轮数**：训练轮数不足时，模型可能尚未收敛；
2. **Patch Size**：较大的 patch 有利于学习更多上下文信息，但会增加显存占用；
3. **Batch Size**：过小可能导致训练不稳定，过大则可能超过显存；
4. **评价方式**：本实验采用 RGB 全图直接评估，不做边界裁剪，因此结果可能与部分文献中的 Y 通道裁边结果存在差异；
5. **模型规模**：ESPCN 属于轻量级模型，效果虽然较好，但通常不如更深的残差类超分模型。

### 8.3 方法优缺点

**优点：**

- 实现简单，适合作业；
- 在 LR 空间计算，效率较高；
- 与 MindSpore 结合实现较为自然。

**缺点：**

- 模型较浅，对复杂纹理恢复能力有限；
- ×4 放大任务中容易出现细节不足、边缘偏平滑的问题；
- 指标和视觉效果通常不如更深层网络（如 EDSR、RCAN 等）。

---

## 9. 结论

本实验基于 MindSpore 实现了一个图像超分辨率系统，采用 ESPCN 网络，在 DIV2K 数据集上完成训练，并在 Set5 数据集上完成 ×4 放大倍率测试。实验流程严格按照题目要求，先使用 Bicubic 方法生成低分辨率图像，再用超分模型恢复高分辨率图像，最后通过 PSNR 和 SSIM 指标评估重建效果。

从实验设计上看，该方案结构清晰、实现难度适中，能够较好地满足课程作业要求。通过本实验，可以较系统地掌握图像超分辨率任务的数据构造方式、网络搭建方法、训练流程以及指标评价方式。

后续若要进一步提升性能，可以考虑更深的残差网络结构、感知损失、Y 通道评估方式以及更完善的数据增强策略。

---

## 10. 参考文献

[1] Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution. ECCV, 2014.

[2] Wenzhe Shi, Jose Caballero, Ferenc Huszar, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. CVPR, 2016.

[3] Eirikur Agustsson, Radu Timofte. NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. CVPR Workshops, 2017.