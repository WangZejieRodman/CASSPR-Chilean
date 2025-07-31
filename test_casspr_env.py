import torch
import MinkowskiEngine as ME
import open3d as o3d
import numpy as np

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")

# 测试MinkowskiEngine
try:
    coords = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32)
    feats = torch.tensor([[1], [2]], dtype=torch.float32)
    sparse_tensor = ME.SparseTensor(feats, coords)
    print("MinkowskiEngine测试成功:", sparse_tensor.shape)
except Exception as e:
    print(f"MinkowskiEngine测试失败: {e}")

# 测试CUDA算子
try:
    import models.transformer.cuda_ops.functions.sparse_ops as ops
    print("CUDA算子导入成功")
except Exception as e:
    print(f"CUDA算子导入失败: {e}")

# 测试其他关键依赖
try:
    import linear_attention_transformer
    import pytorch_metric_learning
    import torchtyping
    print("关键依赖导入成功")
except Exception as e:
    print(f"关键依赖导入失败: {e}")

print("CASSPR环境检查完成")
