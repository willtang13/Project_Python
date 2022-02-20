# -*- coding: utf-8 -*-
#%%
import numpy as np
import matplotlib.pyplot as plt

print(np.__version__)
# %%
heterogeneous_list = [5566, 55.66, True, False, '5566']
for i in heterogeneous_list:
    print(type(i))
# %%
homogeneous_list = [1, 2, 3, 4, 5]
[i**2 for i in homogeneous_list]
# %%
arr = np.array([1, 2, 3, 4, 5])
arr**2
# %%
homogeneous_list  = [1, 2, 3, 4, 5]
type(homogeneous_list)
#%%
arr = np.array(homogeneous_list)
print(type(arr))
print(arr)
print(arr.dtype)
# %%
homogeneous_list  = [1, 2, 3, 4, 5]
arr = np.array(homogeneous_list, dtype=int)
print(arr.dtype)

# %%
np.zeros(5, dtype=int)
# %%
np.ones((2,2), dtype=float)
# %%
np.full((2,2), 5566, dtype=int)
# %%
np.arange(1, 10, 2)
# %%
np.linspace(1, 9, 5, dtype=int)
# %%
uniform_arr = np.random.random(10000)
normal_arr = np.random.normal(0, 1, 10000)
randint_arr = np.random.randint(1, 7, size=6)
print(uniform_arr)
print(normal_arr)
print(randint_arr)

# %%
fig = plt.figure()
ax = plt.axes()
ax.hist(uniform_arr, bins = 30)
plt.show()
# %%
fig = plt.figure()
ax = plt.axes()
ax.hist(normal_arr, bins = 30)
plt.show()

# %%
arr = np.array([[5, 5, 6, 6],
                [7, 7, 8, 8]])
print(arr.ndim)
print(arr.shape)
print(arr.size)
print(arr.dtype)
# %%
scalar = np.array(5566)
print(scalar)
print(scalar.ndim)
print(scalar.shape)

tensor = np.array([5, 5, 6, 6]*3).reshape(3, 2, 2)
print(tensor)
print(tensor.ndim)
print(tensor.shape)
# %%
matrix = np.array([5, 5, 6, 6]).reshape(2, 2)
I = np.eye(matrix.shape[0], dtype=int)
print(I)
# %%
vector = np.array([5, 5, 6, 6])
np.dot(vector, vector)
# %%
print(np.dot(matrix, I))
print(np.dot(I, matrix))
print(np.dot(matrix, matrix))
# %%
matrix = np.arange(6).reshape(2, 3)
print(matrix)
print(np.transpose(matrix))
print(matrix.T)
# %%
A = np.array([1, 2, 3, 4]).reshape(2, 2)
B = np.arange(5, 9, 1).reshape(2, 2)
A_inv = np.linalg.inv(A)
X = np.dot(A_inv, B)
print(X)

# %%
arr = np.array([55, 66, 56, 5566])
for i in range(arr.size):
    print(arr[i])

for i in range(arr.size):
    print(arr[-1-i])
# %%
np.random.seed(42)
arr = np.random.randint(1, 10, size=(3, 4))
print(arr)
print(arr[1, 1])
print(arr[2, -3])

# %%
arr = np.arange(10, 20)
print(arr[::])
print(arr[::2])
print(arr[:5])
print(arr[5:])
print(arr[::-1])
# %%
np.random.seed(0)
arr = np.random.randint(1, 100, size=(10,))
odd_indices = [0, 2, 8]
print(arr)
print(arr[odd_indices])

is_odd = [i % 2 == 1  for i in arr ]
print(arr)
print(is_odd)
print(arr[is_odd])
# %%
arr = np.arange(1, 10)
print(arr)
print(arr.shape)
print(arr.reshape(3, 3))
print(arr.reshape(3, 3).shape)

print(arr.reshape(3, -1))
print(arr.reshape(-1, 3))
# %%
arr = np.arange(1, 4)
print(arr.reshape(-1, 1))
print(arr.reshape(1, -1))

# %%
arr = np.arange(1, 10).reshape(3, 3)
print(arr.ravel())

# %% 複製陣列
arr = np.arange(1, 10)
mat = arr.reshape(3, 3) ## 切個或變形陣列不是複製，更改新陣列的內容，原生陣列也會改變
mat_c = arr.copy().reshape(3, 3) ## 透過複製的方式改變新陣列內容才不會影響原生陣列
mat[1, 1] = 5566
mat_c[2, 2] = 0
print(mat)
print(arr)
print(mat_c)
# %% 合併陣列
arr_a = np.arange(1, 5).reshape(2, 2)
arr_b = np.arange(5, 9).reshape(2, 2)
print(np.concatenate([arr_a, arr_b]))
print(np.concatenate([arr_a, arr_b], axis=1))
# %% 通用函式
long_arr = np.random.randint(1, 101, size=1000000)
%timeit [1/i for i in long_arr]
%timeit np.divide(1, long_arr)
# %% 轉換通用函式
def is_prime(x):
    div_cnt = 0
    for i in range(1, x + 1):
        if x % i == 0:
            div_cnt += 1
        if div_cnt > 2:
            break  
    return div_cnt == 2

is_prime_ufunc = np.vectorize(is_prime)
arr = np.arange(1, 12)
print("Is prime number:")
print(is_prime_ufunc(arr))
# %% 聚合函式
mat = np.arange(1, 16).reshape(3, 5)
print(mat)
print(np.sum(mat))
print(np.sum(mat, axis=0))
print(np.sum(mat, axis=1))

# %%
arr = np.arange(1, 16, dtype=float)
arr[-1] = np.NaN
print(arr)
print(np.sum(arr))
print(np.nansum(arr))

# %%
np.random.seed(42)
arr_0 = np.random.random((10,1))
arr_1 = 1 - arr_0
arr = np.concatenate([arr_0, arr_1], axis=1)
print(arr)

print(np.argmax(arr, axis=0))
print(np.argmax(arr, axis=1))

# %%
