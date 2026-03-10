"""
批量测试 YOLO ONNX 模型
"""

import cv2
from pathlib import Path
from ultralytics import YOLO
import time

def batch_test():
    """批量测试多张图片"""
    
    # 配置
    model_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx")
    test_images_dir = Path(r"D:\MyProjects\c_projects\fork-ultralytics\ultralytics\assets")
    output_dir = Path(r"D:\MyProjects\c_projects\fork-ultralytics\my-tests\batch_results")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("YOLO ONNX 模型批量测试")
    print("=" * 60)
    print()
    
    # 加载模型
    print("📦 加载模型...")
    model = YOLO(str(model_path))
    print(f"✅ 模型加载成功! 任务: {model.task}")
    print(f"   类别: {list(model.names.values())}")
    print()
    
    # 获取测试图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_images = [f for f in test_images_dir.iterdir() 
                  if f.suffix.lower() in image_extensions]
    
    if not test_images:
        print("❌ 未找到测试图片")
        return
    
    print(f"📷 找到 {len(test_images)} 张测试图片:")
    for img in test_images:
        print(f"   - {img.name}")
    print()
    
    # 批量测试
    total_detections = 0
    total_time = 0
    
    for i, image_path in enumerate(test_images, 1):
        print(f"🔍 测试 {i}/{len(test_images)}: {image_path.name}")
        print("-" * 60)
        
        # 推理
        start_time = time.time()
        results = model(str(image_path))
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # 处理结果
        result = results[0]
        boxes = result.boxes
        detections = len(boxes)
        total_detections += detections
        
        print(f"   ⏱️  推理时间: {inference_time*1000:.1f}ms")
        print(f"   🎯 检测数量: {detections}")
        
        if detections > 0:
            print(f"   📊 检测结果:")
            for j, box in enumerate(boxes, 1):
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                print(f"      {j}. {cls_name} (置信度: {conf:.2f})")
        
        # 保存结果
        output_path = output_dir / f"result_{image_path.stem}.jpg"
        result.plot(str(output_path))
        print(f"   💾 结果保存: {output_path.name}")
        print()
    
    # 统计总结
    print("=" * 60)
    print("📈 测试总结")
    print("=" * 60)
    print(f"   总图片数: {len(test_images)}")
    print(f"   总检测数: {total_detections}")
    print(f"   平均检测数: {total_detections/len(test_images):.1f}")
    print(f"   平均推理时间: {total_time/len(test_images)*1000:.1f}ms")
    print(f"   总推理时间: {total_time:.2f}s")
    print()
    print(f"   ✅ 所有结果已保存到: {output_dir}")
    print("🎉 批量测试完成!")

if __name__ == "__main__":
    batch_test()