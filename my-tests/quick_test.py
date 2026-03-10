"""
简单的 YOLO ONNX 模型快速测试
"""

from ultralytics import YOLO
from pathlib import Path

def quick_test():
    """快速测试 ONNX 模型"""
    
    # 配置
    model_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx")
    test_image = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets\zidane.jpg")
    
    print("🚀 快速测试开始...")
    print(f"📦 模型: {model_path.name}")
    print(f"📷 图像: {test_image.name}")
    print()
    
    # 加载模型
    print("1️⃣ 加载模型...")
    model = YOLO(str(model_path))
    print(f"   ✅ 模型加载成功! 任务: {model.task}")
    
    # 推理
    print("2️⃣ 进行推理...")
    results = model(str(test_image))
    print(f"   ✅ 推理完成!")
    
    # 显示结果
    print("3️⃣ 检测结果:")
    result = results[0]
    boxes = result.boxes
    
    if len(boxes) == 0:
        print("   ⚠️  未检测到目标")
    else:
        print(f"   ✅ 检测到 {len(boxes)} 个目标:")
        for i, box in enumerate(boxes, 1):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            print(f"      {i}. {cls_name} (置信度: {conf:.2f})")
    
    # 保存结果
    print("4️⃣ 保存结果...")
    output_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\my-tests\quick_result.jpg")
    result.plot(str(output_path))
    print(f"   ✅ 结果已保存: {output_path.name}")
    
    print()
    print("🎉 测试完成!")

if __name__ == "__main__":
    quick_test()