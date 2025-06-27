# 尝试直接加载为模型对象
import torch

try:
    # 方法1：尝试直接加载
    model = torch.load("checkpoint/1.pt", map_location='cpu')
    if hasattr(model, '__call__'):
        print("成功直接加载为模型对象")
        model.eval()
    else:
        print("加载的不是模型对象，是:", type(model))
except Exception as e:
    print("直接加载失败:", e)

# 方法2：检查是否是torch.jit保存的模型
try:
    model = torch.jit.load("checkpoint/1.pt", map_location='cpu')
    print("成功加载为TorchScript模型")
    model.eval()
except Exception as e:
    print("TorchScript加载失败:", e)
