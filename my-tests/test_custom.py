"""
测试任意图像，支持自定义置信度阈值
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

def preprocess_image(image, input_size=640):
    resized = cv2.resize(image, (input_size, input_size))
    normalized = resized.astype(np.float32) / 255.0
    transposed = normalized.transpose(2, 0, 1)
    batched = np.expand_dims(transposed, axis=0)
    return batched

def postprocess_output(output, conf_threshold=0.25):
    """后处理输出"""
    predictions = output[0]
    
    boxes = predictions[:, :4]
    confidences = predictions[:, 4]
    class_ids = predictions[:, 5].astype(int)
    
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    valid_mask = (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & \
                 (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    
    boxes = boxes[valid_mask]
    confidences = confidences[valid_mask]
    class_ids = class_ids[valid_mask]
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return boxes, confidences, class_ids

def draw_detections(image, boxes, scores, class_ids, class_names):
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"unknown_{class_id}"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{class_name}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                    (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def test_with_threshold(image_path, threshold=0.25):
    """测试指定图像，使用指定阈值"""
    
    model_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx")
    output_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\my-tests\test_result.jpg")
    
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
    print(f"测试图像: {image_path.name}")
    print(f"置信度阈值: {threshold}")
    print("=" * 60)
    print()
    
    # 加载模型
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    original_h, original_w = image.shape[:2]
    print(f"✅ 图像尺寸: {original_w}x{original_h}")
    print()
    
    # 预处理
    input_data = preprocess_image(image)
    
    # 推理
    outputs = session.run([output_name], {input_name: input_data})
    
    # 后处理（使用指定阈值）
    boxes, scores, class_ids = postprocess_output(outputs[0], threshold)
    
    print(f"✅ 检测到 {len(boxes)} 个目标（阈值: {threshold}）")
    print()
    
    # 显示结果
    if len(boxes) > 0:
        print("检测结果:")
        print("-" * 60)
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids), 1):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            print(f"  {i}. {class_name}")
            print(f"     置信度: {score:.4f}")
            print(f"     位置: ({int(box[0])}, {int(box[1])}) -> ({int(box[2])}, {int(box[3])})")
            print()
        print("-" * 60)
        print()
        
        # 缩放边界框到原始图像尺寸
        scale_x = original_w / 640.0
        scale_y = original_h / 640.0
        
        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y
        
        # 绘制结果
        display_image = image.copy()
        draw_detections(display_image, boxes, scores, class_ids, class_names)
        
        cv2.imwrite(str(output_path), display_image)
        print(f"✅ 结果已保存: {output_path}")
        print()
    else:
        print("⚠️  未检测到任何目标")
        print("建议:")
        print("  1. 降低置信度阈值（如 0.1）")
        print("  2. 检查图像是否包含可检测的目标")
        print("  3. 使用更大的模型（如 yolo26s）")
        print()
    
    print("🎉 测试完成!")

if __name__ == "__main__":
    import sys
    
    # 默认测试图像
    default_image = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets\test_1.jpg")
    
    # 获取用户输入
    print("请选择测试选项:")
    print("1. 使用默认图像")
    print("2. 输入自定义图像路径")
    print("3. 测试不同阈值")
    
    choice = input("选择 (1/2/3): ").strip()
    
    if choice == "1":
        test_with_threshold(default_image, 0.1)  # 降低阈值到0.1
    elif choice == "2":
        image_path = input("请输入图像路径: ").strip()
        threshold = float(input("请输入置信度阈值 (0.0-1.0): ").strip())
        test_with_threshold(Path(image_path), threshold)
    elif choice == "3":
        print("测试不同阈值:")
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25]
        for threshold in thresholds:
            print(f"\n{'='*40}")
            print(f"阈值: {threshold}")
            print('='*40)
            test_with_threshold(default_image, threshold)
    else:
        print("无效选择，使用默认选项")
        test_with_threshold(default_image, 0.1)