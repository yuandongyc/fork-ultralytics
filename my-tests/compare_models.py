"""
比较两个 ONNX 模型的差异
"""

import onnx
from pathlib import Path

def compare_models(model1_path, model2_path):
    """比较两个 ONNX 模型"""
    
    print("=" * 60)
    print("ONNX 模型对比")
    print("=" * 60)
    print()
    
    # 加载模型
    model1 = onnx.load(model1_path)
    model2 = onnx.load(model2_path)
    
    print(f"模型1: {model1_path}")
    print(f"模型2: {model2_path}")
    print()
    
    # 文件大小
    size1_mb = Path(model1_path).stat().st_size / (1024 * 1024)
    size2_mb = Path(model2_path).stat().st_size / (1024 * 1024)
    
    print(f"文件大小:")
    print(f"  模型1: {size1_mb:.2f} MB")
    print(f"  模型2: {size2_mb:.2f} MB")
    print(f"  差异: {size1_mb - size2_mb:.2f} MB ({((size1_mb - size2_mb) / size1_mb * 100):.1f}%)")
    print()
    
    # 模型信息
    print("模型信息:")
    print(f"  模型1 - 节点数: {len(model1.graph.node)}, 输入数: {len(model1.graph.input)}, 输出数: {len(model1.graph.output)}")
    print(f"  模型2 - 节点数: {len(model2.graph.node)}, 输入数: {len(model2.graph.input)}, 输出数: {len(model2.graph.output)}")
    print()
    
    # 输入输出
    print("输入输出:")
    print(f"  模型1:")
    for inp in model1.graph.input:
        print(f"    输入: {inp.name}, 形状: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    for out in model1.graph.output:
        print(f"    输出: {out.name}, 形状: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
    
    print(f"  模型2:")
    for inp in model2.graph.input:
        print(f"    输入: {inp.name}, 形状: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    for out in model2.graph.output:
        print(f"    输出: {out.name}, 形状: {[d.dim_value for d in out.type.tensor_type.shape.dim]}")
    print()
    
    # 元数据
    print("元数据:")
    print(f"  模型1:")
    for key, value in model1.metadata_props:
        print(f"    {key}: {value}")
    print(f"  模型2:")
    for key, value in model2.metadata_props:
        print(f"    {key}: {value}")
    print()
    
    print("=" * 60)
    print("对比完成")
    print("=" * 60)

if __name__ == "__main__":
    model1 = r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n.onnx"
    model2 = r"D:\MyProjects\c_projects\fork-ultralytics\models\yolo26n-onnx\yolo26n_runtime.onnx"
    
    compare_models(model1, model2)