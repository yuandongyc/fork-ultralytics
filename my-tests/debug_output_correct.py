"""
正确调试 ONNX 模型输出
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

def debug_onnx_output():
    """调试 ONNX 输出"""
    
    model_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx")
    test_image_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets\zidane.jpg")
    
    print("=" * 60)
    print("正确调试 ONNX 模型输出")
    print("=" * 60)
    print()
    
    # 加载模型
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # 读取图像
    image = cv2.imread(str(test_image_path))
    resized = cv2.resize(image, (640, 640))
    normalized = resized.astype(np.float32) / 255.0
    transposed = normalized.transpose(2, 0, 1)
    input_data = np.expand_dims(transposed, axis=0)
    
    # 推理
    outputs = session.run([output_name], {input_name: input_data})
    output = outputs[0]
    
    print(f"输出形状: {output.shape}")
    print()
    
    # 正确解析输出
    # output shape: (1, 300, 6)
    # 6 = [x1, y1, x2, y2, confidence, class_id]
    
    predictions = output[0]  # (300, 6)
    
    print("前10个检测结果:")
    print("-" * 60)
    for i in range(min(10, len(predictions))):
        x1, y1, x2, y2, conf, cls = predictions[i]
        print(f"{i+1}. 边界框: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")
        print(f"   置信度: {conf:.3f}")
        print(f"   类别ID: {int(cls)}")
        print()
    
    print("-" * 60)
    print()
    
    # 统计信息
    print("统计信息:")
    print("-" * 60)
    
    boxes = predictions[:, :4]
    confidences = predictions[:, 4]
    class_ids = predictions[:, 5].astype(int)
    
    print(f"边界框统计:")
    print(f"  x1 范围: [{np.min(boxes[:, 0]):.1f}, {np.max(boxes[:, 0]):.1f}]")
    print(f"  y1 范围: [{np.min(boxes[:, 1]):.1f}, {np.max(boxes[:, 1]):.1f}]")
    print(f"  x2 范围: [{np.min(boxes[:, 2]):.1f}, {np.max(boxes[:, 2]):.1f}]")
    print(f"  y2 范围: [{np.min(boxes[:, 3]):.1f}, {np.max(boxes[:, 3]):.1f}]")
    print()
    
    print(f"置信度统计:")
    print(f"  最小值: {np.min(confidences):.3f}")
    print(f"  最大值: {np.max(confidences):.3f}")
    print(f"  平均值: {np.mean(confidences):.3f}")
    print()
    
    print(f"类别ID统计:")
    print(f"  最小值: {np.min(class_ids)}")
    print(f"  最大值: {np.max(class_ids)}")
    print(f"  唯一值数量: {len(np.unique(class_ids))}")
    print()
    
    # 检查置信度是否需要sigmoid
    max_conf = np.max(confidences)
    if max_conf > 10:
        print("⚠️  置信度值较大，需要应用sigmoid")
        confidences_sigmoid = 1 / (1 + np.exp(-confidences))
        print(f"   sigmoid后最大值: {np.max(confidences_sigmoid):.3f}")
        print(f"   sigmoid后平均值: {np.mean(confidences_sigmoid):.3f}")
    else:
        print("✅ 置信度值在合理范围内")
    
    print()
    print("-" * 60)
    print()
    print("🎉 调试完成!")

if __name__ == "__main__":
    debug_onnx_output()