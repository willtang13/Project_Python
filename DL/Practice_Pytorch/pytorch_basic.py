#### Ref: https://medium.com/ching-i/pytorch-%E5%9F%BA%E6%9C%AC%E4%BB%8B%E7%B4%B9%E8%88%87%E6%95%99%E5%AD%B8-ac0e1ebfd7ec

#%%
import torch
import numpy as np
# %%
torch.tensor([[1, 2], [3, 4], [5, 6]])
# %%
torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64)
# %%
torch.zeros([2, 2])
# %%
torch.ones([3, 2]) 

# %% Setting GPU if available
if torch.cuda.is_available():
    cuda0 = torch.device('cuda:0')
    t1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64, device=cuda0)
else:
    print('Cuda is not available')
    cpu = torch.device('cpu')
    t1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64, device=cpu)

# %% Transfer between Tensor and Numpy 
numpy1 = t1.numpy()
print('numpy1:', numpy1)
print('type:', type(numpy1))
# %%
numpy2 = np.array([[1, 2, 3], [4, 5, 6]])
# 1st method
tensor1 = torch.tensor(numpy2)
print('tensor1 dtype: ', tensor1.dtype)
# 2nd method
tensor2 = torch.Tensor(numpy2)
print('tensor2 dtype: ', tensor2.dtype)
# 3rd method
tensor3 = torch.as_tensor(numpy2)
print('tensor3 dtype: ', tensor3.dtype)
# 4th method
tensor4 = torch.from_numpy(numpy2)
print('tensor4 dtype: ', tensor4.dtype)
# %% 
# 建立 Tensor 並設定 requires_grad 為 True
torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64, requires_grad=True)
# 建立隨機數值的 Tensor 並設定 requires_grad 為 True
torch.randn(2, 3, requires_grad=True)

# %% 計算梯度
# 建立隨機數值的 Tensor 並設定 requires_grad=True
x = torch.randn(2, 3, requires_grad=True)
y = torch.randn(2, 3, requires_grad=True)
z = torch.randn(2, 3, requires_grad=True)

# 計算式子
a = x * y
b = a + z
c = torch.sum(b)

# 計算梯度
c.backward()

# 查看 x 的梯度值
print(x.grad)

# 設定 detach()
d = x.detach()

# 查看是否追蹤 d 的梯度計算及數值?? 為何d回傳值不是None??
print('d requires grad: ', d.requires_grad)
print('d grad: ', d)
# %% 若使用 with torch.no_grad()，在其作用域範圍內的計算式變量 requires_grad 為 False。
# 建立隨機數值的 Tensor 並設定 requires_grad=True
x = torch.randn(2, 3, requires_grad=True)
print('set x requires_grad:', x.requires_grad)
y = torch.randn(2, 3, requires_grad=True)
z = torch.randn(2, 3, requires_grad=True)

with torch.no_grad():
    a = x * y
    b = a + z
    c = torch.sum(b)
    print('no grad c requires_grad: ', c.requires_grad)
    
# %%
