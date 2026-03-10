"""
简单的 YOLO ONNX 模型图像识别测试脚本
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def test_onnx_inference():
    """测试 ONNX 模型推理"""
    
    # 配置参数
    model_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx")
    test_image_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets\zidane.jpg")
    output_dir = Path(r"D:\MyProjects\c_projects\fork-ultralytics\my-tests")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("YOLO ONNX 模型图像识别测试")
    print("=" * 60)
    
    # 检查文件是否存在
    if not model_path.exists():
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return
    
    if not test_image_path.exists():
        print(f"❌ 错误: 测试图像不存在: {test_image_path}")
        return
    
    print(f"✅ 模型文件: {model_path}")
    print(f"✅ 测试图像: {test_image_path}")
    print(f"✅ 输出目录: {output_dir}")
    print()
    
    # 加载 ONNX 模型
    print("📦 正在加载 ONNX 模型...")
    try:
        model = YOLO(str(model_path))
        print("✅ 模型加载成功!")
        print(f"   - 模型类型: {type(model)}")
        print(f"   - 任务类型: {model.task}")
        print(f"   - 类别数量: {len(model.names)}")
        print(f"   - 类别名称: {list(model.names.values())[:5]}...")
        print()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 读取测试图像
    print("📷 正在读取测试图像...")
    try:
        image = cv2.imread(str(test_image_path))
        if image is None:
            print(f"❌ 无法读取图像: {test_image_path}")
            return
        
        h, w = image.shape[:2]
        print(f"✅ 图像读取成功!")
        print(f"   - 图像尺寸: {w}x{h}")
        print(f"   - 图像通道: {image.shape[2]}")
        print()
    except Exception as e:
        print(f"❌ 图像读取失败: {e}")
        return
    
    # 进行推理
    print("🔍 正在进行推理...")
    try:
        results = model(str(test_image_path), verbose=True)
        print("✅ 推理完成!")
        print()
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return
    
    # 处理结果
    print("📊 推理结果:")
    print("-" * 60)
    
    result = results[0]
    boxes = result.boxes
    
    if len(boxes) == 0:
        print("⚠️  未检测到任何目标")
    else:
        print(f"✅ 检测到 {len(boxes)} 个目标:")
        print()
        
        for i, box in enumerate(boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            
            print(f"  目标 {i+1}:")
            print(f"    - 类别: {class_name} (ID: {class_id})")
            print(f"    - 置信度: {confidence:.3f}")
            print(f"    - 位置: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")
            print(f"    - 尺寸: {int(x2-x1)}x{int(y2-y1)}")
            print()
    
    print("-" * 60)
    print()
    
    # 绘制结果并保存
    print("🎨 正在绘制检测结果...")
    try:
        # 绘制检测结果
        annotated_image = result.plot()
        
        # 保存结果
        output_path = output_dir / "result.jpg"
        cv2.imwrite(str(output_path), annotated_image)
        print(f"✅ 结果已保存到: {output_path}")
        print()
    except Exception as e:
        print(f"❌ 结果保存失败: {e}")
        return
    
    # 显示统计信息
    print("📈 统计信息:")
    print("-" * 60)
    
    if len(boxes) > 0:
        # 按类别统计
        class_counts = {}
        for box in boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} 个")
        
        print()
        print(f"  平均置信度: {np.mean([box.conf[0].cpu().numpy() for box in boxes]):.3f}")
        print(f"  最高置信度: {np.max([box.conf[0].cpu().numpy() for box in boxes]):.3f}")
        print(f"  最低置信度: {np.min([box.conf[0].cpu().numpy() for box in boxes]):.3f}")
    
    print("-" * 60)
    print()
    print("✅ 测试完成!")

if __name__ == "__main__":
    test_onnx_inference()