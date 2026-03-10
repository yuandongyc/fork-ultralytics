"""
调试 ONNX 模型输出
查看原始输出数据
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

def debug_onnx_output():
    """调试 ONNX 输出"""
    
    # 配置
    model_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx")
    test_image_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets\zidane.jpg")
    
    print("=" * 60)
    print("调试 ONNX 模型输出")
    print("=" * 60)
    print()
    
    # 加载模型
    print("📦 加载模型...")
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✅ 输入: {input_name}, 输出: {output_name}")
    print()
    
    # 读取图像
    print("📷 读取图像...")
    image = cv2.imread(str(test_image_path))
    resized = cv2.resize(image, (640, 640))
    normalized = resized.astype(np.float32) / 255.0
    transposed = normalized.transpose(2, 0, 1)
    input_data = np.expand_dims(transposed, axis=0)
    print(f"✅ 输入形状: {input_data.shape}")
    print()
    
    # 推理
    print("🔍 推理...")
    outputs = session.run([output_name], {input_name: input_data})
    output = outputs[0]
    print(f"✅ 输出形状: {output.shape}")
    print()
    
    # 分析输出
    print("📊 输出分析:")
    print("-" * 60)
    
    # 查看前几个anchor点的输出
    print("前5个anchor点的输出:")
    for i in range(min(5, output.shape[2])):
        print(f"\nAnchor {i}:")
        print(f"  边界框: {output[0, :4, i]}")
        print(f"  类别分数 (前5个): {output[0, 4:9, i]}")
        print(f"  类别分数 (最大): {np.max(output[0, 4:, i])}")
        print(f"  类别分数 (最小): {np.min(output[0, 4:, i])}")
        print(f"  类别分数 (平均): {np.mean(output[0, 4:, i])}")
    
    print()
    print("-" * 60)
    print()
    
    # 统计信息
    print("📈 统计信息:")
    print("-" * 60)
    
    # 边界框统计
    boxes = output[0, :4, :]
    print(f"边界框统计:")
    print(f"  最小值: {np.min(boxes)}")
    print(f"  最大值: {np.max(boxes)}")
    print(f"  平均值: {np.mean(boxes)}")
    print(f"  中位数: {np.median(boxes)}")
    print()
    
    # 类别分数统计
    scores = output[0, 4:, :]
    print(f"类别分数统计:")
    print(f"  最小值: {np.min(scores)}")
    print(f"  最大值: {np.max(scores)}")
    print(f"  平均值: {np.mean(scores)}")
    print(f"  中位数: {np.median(scores)}")
    print()
    
    # 检查是否需要sigmoid
    max_score = np.max(scores)
    print(f"最大类别分数: {max_score}")
    if max_score > 10:
        print("⚠️  分数值较大，可能需要应用sigmoid")
    elif max_score > 1:
        print("⚠️  分数值在1-10之间，可能需要sigmoid")
    else:
        print("✅ 分数值在0-1之间，可能已经是sigmoid后的")
    print()
    
    print("-" * 60)
    print()
    print("🎉 调试完成!")

if __name__ == "__main__":
    debug_onnx_output()