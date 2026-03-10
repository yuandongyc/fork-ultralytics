"""
屏幕目标检测和操作
实时截屏并检测目标
"""

import cv2
import numpy as np
import onnxruntime as ort
import pyautogui
from pathlib import Path
import time

def preprocess_image(image, input_size=640):
    """预处理图像"""
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
        return [], [], []
    
    valid_mask = (boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & \
                 (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    
    boxes = boxes[valid_mask]
    confidences = confidences[valid_mask]
    class_ids = class_ids[valid_mask]
    
    if len(boxes) == 0:
        return [], [], []
    
    return boxes, confidences, class_ids

def draw_detections(image, boxes, scores, class_ids, class_names):
    """绘制检测结果"""
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

def get_screen_shot():
    """获取屏幕截图"""
    screenshot = pyautogui.screenshot()
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot

def scale_boxes_to_screen(boxes, screen_size, model_size=640):
    """将边界框缩放到屏幕尺寸"""
    screen_w, screen_h = screen_size
    scale_x = screen_w / model_size
    scale_y = screen_h / model_size
    
    scaled_boxes = boxes.copy()
    scaled_boxes[:, 0] *= scale_x
    scaled_boxes[:, 1] *= scale_y
    scaled_boxes[:, 2] *= scale_x
    scaled_boxes[:, 3] *= scale_y
    
    return scaled_boxes

def click_on_target(boxes, scores, class_ids, target_class_id, screen_size, model_size=640):
    """点击指定类别的目标"""
    scaled_boxes = scale_boxes_to_screen(boxes, screen_size, model_size)
    
    for i, (box, score, class_id) in enumerate(zip(scaled_boxes, scores, class_ids)):
        if class_id == target_class_id:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            print(f"点击目标: {target_class_id} 在 ({center_x}, {center_y})")
            pyautogui.click(center_x, center_y)
            return True
    
    return False

def screen_detection_demo():
    """屏幕检测演示"""
    
    model_path = Path(r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx")
    
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
    print("屏幕目标检测演示")
    print("=" * 60)
    print()
    
    # 加载模型
    print("📦 加载模型...")
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("✅ 模型加载成功!")
    print()
    
    # 获取屏幕尺寸
    screen_size = pyautogui.size()
    print(f"📺 屏幕尺寸: {screen_size}")
    print()
    
    print("🔍 开始检测 (按 Ctrl+C 停止)...")
    print()
    
    try:
        while True:
            # 截屏
            screenshot = get_screen_shot()
            
            # 预处理
            input_data = preprocess_image(screenshot)
            
            # 推理
            outputs = session.run([output_name], {input_name: input_data})
            
            # 后处理
            boxes, scores, class_ids = postprocess_output(outputs[0])
            
            # 缩放边界框到屏幕尺寸
            screen_h, screen_w = screenshot.shape[:2]
            scale_x = screen_w / 640.0
            scale_y = screen_h / 640.0
            
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 3] *= scale_y  # y2
            
            # 绘制结果
            display_image = screenshot.copy()
            draw_detections(display_image, boxes, scores, class_ids, class_names)
            
            # 显示检测结果
            if len(boxes) > 0:
                print(f"检测到 {len(boxes)} 个目标:")
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids), 1):
                    class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                    print(f"  {i}. {class_name}: {score:.3f}")
                print()
            
            # 显示图像
            cv2.imshow("Screen Detection", display_image)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 控制帧率
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    
    cv2.destroyAllWindows()
    print("\n🎉 演示完成!")

if __name__ == "__main__":
    screen_detection_demo()