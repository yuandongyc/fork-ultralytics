# YOLO ONNX 模型测试脚本

本目录包含用于测试 YOLO ONNX 模型的脚本。

## 📁 文件说明

### 轻量级测试（推荐）
- **test_onnxruntime.py** - 纯 ONNXRuntime 推理，无需 ultralytics 依赖

### 完整功能测试（需要 ultralytics）
- **test_onnx_inference.py** - 详细的单图测试脚本
- **quick_test.py** - 快速测试脚本
- **batch_test.py** - 批量测试多张图片

## 🚀 使用方法

### 1. 轻量级测试（推荐）

**安装依赖：**
```bash
pip install onnxruntime opencv-python numpy
```

**运行测试：**
```bash
python test_onnxruntime.py
```

**优点：**
- ✅ 只需安装 ~50-150 MB（onnxruntime）
- ✅ 无需安装 ultralytics（~1GB）
- ✅ 更接近实际部署环境
- ✅ 推理性能更准确

**功能：**
- 加载 ONNX 模型
- 预处理图像
- ONNXRuntime 推理
- 后处理检测结果
- 绘制并保存结果

### 2. 完整功能测试

**安装依赖：**
```bash
pip install ultralytics[export]
```

**运行测试：**
```bash
# 快速测试
python quick_test.py

# 详细测试
python test_onnx_inference.py

# 批量测试
python batch_test.py
```

## 📋 配置说明

### test_onnxruntime.py 配置

```python
# 模型路径
model_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx")

# 测试图片路径
test_image_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets\zidane.jpg")

# 输出路径
output_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\my-tests\onnxruntime_result.jpg")
```

### 其他脚本配置

参见各脚本开头的配置部分。

## 📊 输出说明

### 控制台输出

```
============================================================
ONNXRuntime YOLO 模型推理测试
============================================================

✅ 模型: yolo26n.onnx
✅ 图像: zidane.jpg

📦 加载 ONNX 模型...
✅ 模型加载成功!
   输入名称: images
   输出名称: output0
   输入形状: [1, 3, 640, 640]

📷 读取图像...
✅ 图像尺寸: 640x480

🔧 预处理图像...
✅ 预处理完成，输入形状: (1, 3, 640, 640)

🔍 进行推理...
✅ 推理完成!
   输出形状: (1, 84, 8400)

📊 后处理输出...
✅ 检测到 2 个目标

检测结果:
------------------------------------------------------------
  1. person
     置信度: 0.890
     位置: (100, 50) -> (300, 450)

  2. sports ball
     置信度: 0.760
     位置: (400, 300) -> (450, 350)

------------------------------------------------------------

🎨 绘制检测结果...
✅ 结果已保存: onnxruntime_result.jpg

🎉 测试完成!
```

### 图片输出

标注后的图片会保存到 `my-tests` 目录：
- `onnxruntime_result.jpg` - ONNXRuntime 测试结果
- `result.jpg` - 详细测试结果
- `quick_result.jpg` - 快速测试结果
- `batch_results/` - 批量测试结果文件夹

## 🔧 依赖对比

| 测试脚本 | 依赖包 | 安装大小 |
|---------|---------|----------|
| **test_onnxruntime.py** | onnxruntime, opencv-python, numpy | ~50-150 MB |
| **其他脚本** | ultralytics[export] | ~1.8-3 GB |

## 📈 性能参考

基于 yolo26n.onnx 模型的预期性能：

- **推理时间**: ~10-30ms (CPU) / ~1-5ms (GPU)
- **模型大小**: ~6-8 MB
- **检测精度**: mAP50 ≈ 37-40%

## 🐛 故障排除

### 问题 1: onnxruntime 导入失败
```
ImportError: No module named 'onnxruntime'
```
**解决**: 安装 onnxruntime
```bash
pip install onnxruntime
```

### 问题 2: 模型加载失败
```
❌ 错误: 模型文件不存在
```
**解决**: 检查模型路径是否正确

### 问题 3: 推理失败
```
❌ 推理失败: ...
```
**解决**: 
- 检查图片格式是否支持
- 确认 onnxruntime 版本兼容
- 尝试重新导出 ONNX 模型

### 问题 4: 未检测到目标
```
⚠️  未检测到任何目标
```
**解决**:
- 检查图片内容是否包含可检测的目标
- 调整置信度阈值（修改 `conf_threshold` 参数）
- 尝试使用更大的模型（如 yolo26s）

## 📝 参数调整

### test_onnxruntime.py 参数

```python
# 预处理参数
input_size = 640  # 输入图像大小

# 后处理参数
conf_threshold = 0.25  # 置信度阈值
iou_threshold = 0.45   # NMS IOU 阈值
```

## 📚 更多信息

- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [OpenCV 文档](https://docs.opencv.org/)
- [Ultralytics 文档](https://docs.ultralytics.com/)