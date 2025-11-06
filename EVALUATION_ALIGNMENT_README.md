# Long-LRM 与 SparseSplat 评测对齐指南

本文档说明如何在 DL3DV 数据集上运行与 SparseSplat 完全对齐的 Long-LRM 评测。

## 概述

为了公平对比 Long-LRM 和 SparseSplat 的性能，我们需要确保两者使用：
- **相同的输入视角**（6个context views）
- **相同的测试视角**（50个target views）
- **相同的场景集合**（DL3DV-140评测集）

本指南提供了完整的数据准备和评测流程。

---

## 修改内容总结

### 1. 新增文件

| 文件路径 | 功能描述 |
|---------|---------|
| `data/convert_dl3dv_to_llrm.py` | 将DL3DV数据从Blender格式转换为Long-LRM的OpenCV格式 |
| `data/create_scene_list.py` | 生成场景列表文件用于评测 |
| `scripts/verify_alignment.py` | 验证数据转换和对齐正确性 |
| `scripts/eval_dl3dv_aligned.sh` | 评测启动脚本 |
| `scripts/compare_with_sparsesplat.py` | 生成对比分析报告 |
| `configs/dl3dv_eval_sparsesplat_aligned.yaml` | 对齐评测的配置文件 |

### 2. 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `data/dataset.py` | 添加了对SparseSplat固定视角索引的支持 |
| `model/llrm.py` | 在评测输出中添加了高斯基元数量统计 |

---

## 使用流程

### 步骤 1: 数据准备

假设您的DL3DV原始数据存储在其他服务器上，格式与 `dl3dv_test/` 样例一致。

#### 1.1 转换数据格式

将DL3DV数据从Blender格式转换为Long-LRM的OpenCV格式：

```bash
python data/convert_dl3dv_to_llrm.py \
    --input_dir /path/to/dl3dv_raw_data \
    --output_dir /path/to/dl3dv_converted \
    --image_subdir images_8
```

**说明**：
- `--input_dir`: DL3DV原始数据目录（包含所有场景子文件夹）
- `--output_dir`: 转换后数据的输出目录
- `--image_subdir`: 图像子目录名称（默认 `images_8`）

转换过程会：
- 读取每个场景的 `transforms.json`（Blender格式）
- 转换坐标系：Blender (Y-up) → OpenCV (Y-down)
- 生成 `opencv_cameras.json`（Long-LRM格式）
- 创建图像软链接（节省存储空间）

#### 1.2 生成场景列表文件

```bash
python data/create_scene_list.py \
    --index_file /data/zhangzicheng/workspace/SparseSplat-/SparseSplat/assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
    --data_dir /path/to/dl3dv_converted \
    --output_file data/dl3dv_eval_scenes.txt \
    --verify
```

**说明**：
- `--index_file`: SparseSplat的评测索引文件（指定使用哪些场景和视角）
- `--data_dir`: 步骤1.1转换后的数据目录
- `--output_file`: 输出的场景列表文件路径
- `--verify`: 验证所有场景是否存在

生成的 `dl3dv_eval_scenes.txt` 文件包含所有场景的 `opencv_cameras.json` 文件路径（每行一个）。

#### 1.3 验证数据对齐（可选但推荐）

使用样例场景验证转换是否正确：

```bash
# 首先转换样例场景
python data/convert_dl3dv_to_llrm.py \
    --input_dir /data/zhangzicheng/workspace/SparseSplat-/dl3dv_test \
    --output_dir /tmp/test_converted

# 验证对齐
python scripts/verify_alignment.py \
    --llrm_scene_json /tmp/test_converted/0a3e9c8e9b7713e77f45bf55edd1190c925028293aa0561feec54c826a0e6b98/opencv_cameras.json \
    --sparsesplat_index /data/zhangzicheng/workspace/SparseSplat-/SparseSplat/assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json \
    --scene_id 0a3e9c8e9b7713e77f45bf55edd1190c925028293aa0561feec54c826a0e6b98
```

**注意**：样例场景ID可能不在SparseSplat索引中，这是正常的。您应该使用索引文件中实际包含的场景进行验证。

---

### 步骤 2: 配置评测参数

编辑配置文件 `configs/dl3dv_eval_sparsesplat_aligned.yaml`：

```yaml
data_eval:
  # 更新为步骤1.2生成的文件路径
  data_path: "data/dl3dv_eval_scenes.txt"

  # SparseSplat索引文件路径（已预设，无需修改）
  use_sparsesplat_index: true
  sparsesplat_index_path: "/data/zhangzicheng/workspace/SparseSplat-/SparseSplat/assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json"

  # 分辨率设置（根据需要调整）
  resize_h: 256
  resize_w: 448

training:
  # 更新为您的预训练模型路径
  resume_ckpt: "checkpoints/dl3dv_i540_32input_8target/checkpoint_000010000.pt"
```

**关键参数说明**：
- `use_sparsesplat_index: true`: 启用固定视角索引模式
- `num_input_frames: 6`: 输入视角数量（对应context views）
- `num_target_frames: 50`: 测试视角数量（对应target views）
- `resize_h/resize_w`: 输入分辨率（主要对齐视角选择，分辨率可灵活调整）

---

### 步骤 3: 运行评测

#### 3.1 单GPU评测

```bash
bash scripts/eval_dl3dv_aligned.sh 0
```

#### 3.2 多GPU评测（推荐）

```bash
bash scripts/eval_dl3dv_aligned.sh 0,1,2,3
```

评测过程会自动：
1. 加载SparseSplat索引文件
2. 对每个场景使用固定的输入和测试视角
3. 渲染所有测试视角
4. 计算PSNR/SSIM/LPIPS指标
5. 保存高斯模型PLY文件
6. 记录高斯基元数量统计

---

### 步骤 4: 查看评测结果

评测完成后，结果保存在 `eval_results/dl3dv_eval_sparsesplat_aligned/`：

```
eval_results/dl3dv_eval_sparsesplat_aligned/
├── scene_hash_1/
│   ├── rendering/              # 渲染结果图像
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   ├── target/                 # Ground truth图像
│   │   ├── 0.png
│   │   ├── 1.png
│   │   └── ...
│   ├── input_images.png        # 输入视角拼接图
│   ├── gaussians_001.ply       # 3D高斯模型
│   ├── metrics.csv             # 评测指标
│   ├── input_frame_idx.txt     # 使用的输入帧索引
│   ├── target_frame_idx.txt    # 使用的测试帧索引
│   └── input_traj.mp4          # 输入轨迹可视化
├── scene_hash_2/
│   └── ...
└── summary.csv                 # 所有场景汇总
```

#### metrics.csv 格式示例

```csv
index, psnr, ssim, lpips
0, 24.123, 0.8456, 0.1234
1, 25.678, 0.8678, 0.1123
...
mean, 24.567, 0.8512, 0.1198
gaussian_usage, 0.6234
num_gaussians_total, 524288
num_gaussians_active, 326789
inference_time, 1.234
```

---

### 步骤 5: 生成对比报告

将Long-LRM结果与SparseSplat进行对比：

```bash
python scripts/compare_with_sparsesplat.py \
    --llrm_results eval_results/dl3dv_eval_sparsesplat_aligned/summary.csv \
    --output_dir comparison_results
```

生成的对比报告包括：
- `comparison_table.csv/md`: 指标对比表格
- `gaussian_statistics.txt`: 高斯基元统计
- `gaussian_distribution.png`: 高斯数量分布图
- `metrics_distribution.png`: 各指标分布箱线图
- `per_scene_metrics.csv`: 每个场景的详细指标

---

## 技术细节

### 坐标系转换

DL3DV使用Blender坐标系（Y-up），Long-LRM使用OpenCV坐标系（Y-down）。转换公式：

```python
blender_to_opencv = np.array([
    [1,  0,  0,  0],
    [0, -1,  0,  0],
    [0,  0, -1,  0],
    [0,  0,  0,  1]
])

c2w_opencv = blender_to_opencv @ c2w_blender
w2c_opencv = np.linalg.inv(c2w_opencv)
```

### 固定视角索引格式

SparseSplat索引JSON格式：

```json
{
  "scene_hash": {
    "context": [0, 19, 29, 33, 39, 49],  // 6个输入视角
    "target": [0, 1, 2, ..., 49]         // 50个测试视角
  }
}
```

### 高斯基元统计

- `num_gaussians_total`: 剪枝后的高斯总数
- `num_gaussians_active`: opacity > threshold 的有效高斯数
- `gaussian_usage`: 有效高斯比例

---

## 常见问题

### Q1: 预训练模型是32输入的，能否用于6输入评测？

**A**: 可以，但可能存在性能下降。预训练的32输入模型可以处理6个输入（泛化能力测试），但为了最佳性能，建议用6输入配置重新训练模型。

### Q2: 分辨率256x448与SparseSplat不一致会影响对比吗？

**A**: 主要影响是视角选择，分辨率可以灵活调整。但为了最公平的对比，建议检查SparseSplat实际使用的分辨率并保持一致。

### Q3: 某些场景在索引中但数据目录不存在怎么办？

**A**: 使用 `create_scene_list.py` 的 `--verify` 选项会自动跳过缺失的场景。确保您从服务器上获取了索引文件中所有场景的数据。

### Q4: 评测速度很慢怎么办？

**A**:
1. 使用多GPU并行评测：`bash scripts/eval_dl3dv_aligned.sh 0,1,2,3`
2. 调整batch size（配置文件中的`batch_size_per_gpu`）
3. 第一个batch推理较慢是正常的，后续会加速到约1秒/场景

---

## 输出示例

### 评测日志示例

```
Loaded SparseSplat index with 140 scenes
Found evaluation results for 140 scenes
Summary of evaluation results:
  psnr: 24.2134, num_scenes: 140
  ssim: 0.8512, num_scenes: 140
  lpips: 0.1198, num_scenes: 140
  num_gaussians_total: 524288.0, num_scenes: 140
  num_gaussians_active: 326789.0, num_scenes: 140
```

### 对比表格示例

| Metric | Long-LRM | SparseSplat | Difference |
|--------|----------|-------------|------------|
| PSNR ↑ | 24.213   | 25.456      | -1.243     |
| SSIM ↑ | 0.8512   | 0.8678      | -0.0166    |
| LPIPS ↓| 0.1198   | 0.1054      | +0.0144    |

---

## 致谢

本对齐评测方案基于：
- [Long-LRM](https://arthurhero.github.io/projects/llrm/index.html)
- [SparseSplat](https://github.com/...)
- [DL3DV Dataset](https://dl3dv-10k.github.io/DL3DV-10K/)

---

## 联系方式

如有问题，请参考：
- Long-LRM 原始README: `README.md`
- 问题追踪: 创建GitHub Issue

---

**最后更新**: 2025-01-06
