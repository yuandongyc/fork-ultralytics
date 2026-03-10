"""
纯 ONNXRuntime YOLO 模型推理测试
无需 ultralytics 依赖，轻量级推理
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

def preprocess_image(image, input_size=640):
    """预处理图像"""
    # 调整大小
    resized = cv2.resize(image, (input_size, input_size))
    
    # 归一化到 [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # 转换为 CHW 格式
    transposed = normalized.transpose(2, 0, 1)
    
    # 添加 batch 维度
    batched = np.expand_dims(transposed, axis=0)
    
    return batched

def postprocess_output(output, conf_threshold=0.25):
    """后处理输出"""
    # output shape: (1, 300, 6)
    # 6 = [x1, y1, x2, y2, confidence, class_id]
    
    predictions = output[0]  # (300, 6)
    
    # 提取边界框、置信度和类别ID
    boxes = predictions[:, :4]  # (300, 4)
    confidences = predictions[:, 4]  # (300,)
    class_ids = predictions[:, 5].astype(int)  # (300,)
    
    # 过滤低置信度
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # 过滤无效的边界框
    valid_mask = (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & \
                 (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    
    boxes = boxes[valid_mask]
    confidences = confidences[valid_mask]
    class_ids = class_ids[valid_mask]
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return boxes, confidences, class_ids

def draw_detections(image, boxes, scores, class_ids, class_names):
    """绘制检测结果"""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        
        # 确保类别ID在有效范围内
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"unknown_{class_id}"
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"{class_name}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                    (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def test_onnx_runtime():
    """测试 ONNXRuntime 推理"""
    
    # 配置
    model_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx")
    # test_image_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets\zidane.jpg")
    # test_image_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets\bus.jpg")
    test_image_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets\test_1.jpg")
    output_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\my-tests\onnxruntime_result.jpg")
    
    # COCO 类别名称（前80个）
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    print("=" * 60)
    print("ONNXRuntime YOLO 模型推理测试")
    print("=" * 60)
    print()
    
    # 检查文件
    if not model_path.exists():
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return
    
    if not test_image_path.exists():
        print(f"❌ 错误: 测试图像不存在: {test_image_path}")
        return
    
    print(f"✅ 模型: {model_path.name}")
    print(f"✅ 图像: {test_image_path.name}")
    print()
    
    # 加载模型
    print("📦 加载 ONNX 模型...")
    try:
        session = ort.InferenceSession(str(model_path))
        print("✅ 模型加载成功!")
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"   输入名称: {input_name}")
        print(f"   输出名称: {output_name}")
        print(f"   输入形状: {session.get_inputs()[0].shape}")
        print()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 读取图像
    print("📷 读取图像...")
    image = cv2.imread(str(test_image_path))
    if image is None:
        print(f"❌ 无法读取图像: {test_image_path}")
        return
    
    original_h, original_w = image.shape[:2]
    print(f"✅ 图像尺寸: {original_w}x{original_h}")
    print()
    
    # 预处理
    print("🔧 预处理图像...")
    input_size = 640
    input_data = preprocess_image(image, input_size)
    print(f"✅ 预处理完成，输入形状: {input_data.shape}")
    print()
    
    # 推理
    print("🔍 进行推理...")
    try:
        outputs = session.run([output_name], {input_name: input_data})
        print("✅ 推理完成!")
        print(f"   输出形状: {outputs[0].shape}")
        print()
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return
    
    # 后处理
    print("📊 后处理输出...")
    boxes, scores, class_ids = postprocess_output(outputs[0])
    
    # 缩放边界框到原始图像尺寸
    if len(boxes) > 0:
        scale_x = original_w / 640.0
        scale_y = original_h / 640.0
        
        boxes[:, 0] *= scale_x  # x1
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 3] *= scale_y  # y2
    
    print(f"✅ 检测到 {len(boxes)} 个目标")
    print()
    
    # 显示结果
    if len(boxes) > 0:
        print("检测结果:")
        print("-" * 60)
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids), 1):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            print(f"  {i}. {class_name}")
            print(f"     置信度: {score:.3f}")
            print(f"     位置: ({int(box[0])}, {int(box[1])}) -> ({int(box[2])}, {int(box[3])})")
            print()
        print("-" * 60)
        print()
    else:
        print("⚠️  未检测到任何目标")
        print()
    
    # 绘制结果
    print("🎨 绘制检测结果...")
    draw_detections(image, boxes, scores, class_ids, class_names)
    
    # 保存结果
    cv2.imwrite(str(output_path), image)
    print(f"✅ 结果已保存: {output_path}")
    print()
    
    print("🎉 测试完成!")

if __name__ == "__main__":
    test_onnx_runtime()