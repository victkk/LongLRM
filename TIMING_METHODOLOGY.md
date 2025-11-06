# Long-LRM 推理时间记录方法

## 概述

本文档说明 Long-LRM 中 `model_inference_time_sec` 的精确定义和测量方法。

---

## 时间记录范围

### ✅ 包含的操作（模型推理）

`model_inference_time_sec` **仅包含**以下核心模型前向传播操作：

1. **Tokenization** (`llrm.py:308`)
   - 将图像patch和相机信息编码为tokens
   - 通过 `self.tokenizer` 处理

2. **Global Token添加** (`llrm.py:309-313`)
   - 添加global tokens（如果使用）
   - LayerNorm处理

3. **Processor处理** (`llrm.py:315`)
   - Transformer/Mamba2层的前向传播
   - 这是模型的主要计算部分

4. **Token解码** (`llrm.py:320`)
   - 将tokens解码为3D高斯参数
   - 通过 `self.tokenDecoder` 处理

5. **高斯参数生成** (`llrm.py:322-342`)
   - 生成xyz, feature, scale, rotation, opacity
   - 像素对齐处理

### ❌ 不包含的操作

以下操作**不计入**推理时间：

1. **数据加载** (`dataset.py`)
   - 从磁盘读取图像
   - JSON文件解析

2. **相机信息预处理** (`llrm.py:280-295`)
   - Ray计算
   - Plucker坐标生成
   - Patchify操作

3. **高斯剪枝** (`llrm.py:356-373`)
   - 基于opacity的剪枝
   - 随机保留

4. **渲染** (`llrm.py:393-407`)
   - 高斯光栅化
   - 生成渲染图像

5. **损失计算** (`llrm.py:417-454`)
   - PSNR/SSIM/LPIPS计算

---

## 代码实现

### 时间记录位置

```python
# 开始计时（llrm.py:304-305）
torch.cuda.synchronize()  # 确保GPU操作完成
inference_start = time.time()

# ... 模型前向传播 ...

# 结束计时（llrm.py:353-354）
torch.cuda.synchronize()  # 确保GPU操作完成
inference_time = time.time() - inference_start
```

### 关键点

1. **GPU同步**：使用 `torch.cuda.synchronize()` 确保异步GPU操作完成后再记录时间
2. **精确边界**：
   - 开始：tokenization之前
   - 结束：生成完整高斯参数之后（但在剪枝之前）

---

## 测量结果说明

### 输出文件

时间信息保存在每个场景的 `metrics.csv` 文件中：

```csv
index, psnr, ssim, lpips
0, 24.123, 0.8456, 0.1234
...
mean, 24.567, 0.8512, 0.1198
gaussian_usage, 0.6234
num_gaussians_total, 524288
num_gaussians_active, 326789
model_inference_time_sec, 1.234
```

### 时间含义

- **单位**：秒（seconds）
- **精度**：毫秒级（受Python `time.time()` 限制）
- **测量对象**：单个batch的纯模型推理时间
- **batch size**：通常为1（每次处理1个场景）

### 典型值

根据不同配置和硬件，预期时间范围：

| 输入视角数 | 分辨率 | GPU | 预期时间 |
|-----------|--------|-----|---------|
| 6 views   | 256×448 | A100 | 0.5-1.5s |
| 32 views  | 540×960 | A100 | 2-5s |

**注意**：
- 第一个batch通常较慢（CUDA初始化、内核编译等）
- 后续batch会显著加速
- 使用gradient checkpointing会增加时间

---

## 与其他方法对比

在与SparseSplat等方法对比时，确保：

1. **时间范围一致**：都只计算模型推理时间
2. **GPU同步一致**：都使用 `torch.cuda.synchronize()`
3. **batch size一致**：通常都是batch=1
4. **预热一致**：都跳过第一个batch（warm-up）

---

## 完整时间分解

如需测量完整流程的各部分时间：

```python
# 数据加载时间（在 dataset.py 中）
data_load_start = time.time()
data = dataset[idx]
data_load_time = time.time() - data_load_start

# 模型推理时间（在 llrm.py 中，已实现）
model_inference_time  # 1-3秒

# 渲染时间（在 llrm.py render部分）
render_start = time.time()
renderings = self.render(...)
render_time = time.time() - render_start

# 总时间
total_time = data_load_time + model_inference_time + render_time
```

---

## 最后更新

- **日期**：2025-01-06
- **版本**：与 SparseSplat 对齐评测版本
- **文件**：`model/llrm.py:304, 353-354, 654`

---

## 参考

- 代码位置：`Long-LRM/model/llrm.py`
- 相关Issue：模型推理时间精确测量
- 对齐目标：SparseSplat 评测标准
