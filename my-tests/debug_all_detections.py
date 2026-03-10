"""
调试版本：显示所有检测结果（不管置信度）
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

def postprocess_all_output(output):
    """显示所有检测结果，不管置信度"""
    predictions = output[0]
    
    boxes = predictions[:, :4]
    confidences = predictions[:, 4]
    class_ids = predictions[:, 5].astype(int)
    
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

def debug_test():
    model_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx")
    test_image_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets\test_1.jpg")
    output_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\my-tests\debug_result.jpg")
    
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
    print("调试：显示所有检测结果")
    print("=" * 60)
    print()
    
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    if not test_image_path.exists():
        print(f"❌ 错误: 图像文件不存在: {test_image_path}")
        return
    
    image = cv2.imread(str(test_image_path))
    if image is None:
        print(f"❌ 错误: 无法读取图像: {test_image_path}")
        print("   请检查:")
        print("   1. 文件路径是否正确")
        print("   2. 文件是否存在")
        print("   3. 文件格式是否支持")
        return
    
    original_h, original_w = image.shape[:2]
    print(f"✅ 图像尺寸: {original_w}x{original_h}")
    print()
    
    input_data = preprocess_image(image)
    outputs = session.run([output_name], {input_name: input_data})
    
    print(f"✅ 输出形状: {outputs[0].shape}")
    print()
    
    boxes, scores, class_ids = postprocess_all_output(outputs[0])
    
    print(f"✅ 检测到 {len(boxes)} 个目标（所有置信度）")
    print()
    
    if len(boxes) > 0:
        print("所有检测结果（按置信度排序）:")
        print("-" * 60)
        
        sorted_indices = np.argsort(scores)[::-1]
        for i, idx in enumerate(sorted_indices, 1):
            class_name = class_names[class_ids[idx]] if class_ids[idx] < len(class_names) else f"class_{class_ids[idx]}"
            print(f"  {i}. {class_name}")
            print(f"     置信度: {scores[idx]:.4f}")
            print(f"     位置: ({int(boxes[idx, 0])}, {int(boxes[idx, 1])}) -> ({int(boxes[idx, 2])}, {int(boxes[idx, 3])})")
            print()
        
        print("-" * 60)
        print()
        
        print("置信度统计:")
        print(f"  最大值: {np.max(scores):.4f}")
        print(f"  最小值: {np.min(scores):.4f}")
        print(f"  平均值: {np.mean(scores):.4f}")
        print(f"  中位数: {np.median(scores):.4f}")
        print()
        
        print("建议阈值:")
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        for threshold in thresholds:
            count = np.sum(scores > threshold)
            print(f"  阈值 {threshold:.2f}: {count} 个目标")
        print()
        
        if len(boxes) > 0:
            scale_x = original_w / 640.0
            scale_y = original_h / 640.0
            
            boxes[:, 0] *= scale_x
            boxes[:, 1] *= scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
            
            display_image = image.copy()
            draw_detections(display_image, boxes, scores, class_ids, class_names)
            
            cv2.imwrite(str(output_path), display_image)
            print(f"✅ 结果已保存: {output_path}")
            print()
    else:
        print("⚠️  没有检测到任何有效目标")
        print()
    
    print("🎉 调试完成!")

if __name__ == "__main__":
    debug_test()