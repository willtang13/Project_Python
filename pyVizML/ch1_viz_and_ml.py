# -*- coding: utf-8 -*-
#%%
from IPython.display import YouTubeVideo
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

#%%
# YouTubeVideo('Z8t4k0Q8e8Y', width=640, height=360)
# %%
arr = np.random.normal(size=10000)
arr
# %%
fig = plt.figure()
ax = plt.axes()
ax.hist(arr, bins=50)
plt.show
# %%
'''
函式
'''
eps = 1e-06
p = np.linspace(0 + eps, 1 - eps, 10000)
log_loss_0 = -np.log(1-p)
log_loss_1 = -np.log(p)
print(p)
print(log_loss_0)
print(log_loss_1)

fig = plt.figure()
ax = plt.axes()
ax.plot(p, log_loss_0, label='$y_{true}=0$')
ax.plot(p, log_loss_1, label='$y_{true}=1$')
ax.legend()
plt.show()
# %%
'''
Sigmod 數學式
'''
x = np.linspace(-6, 6, 1000)
fx = 1 / (1 + np.exp(-x))
fig = plt.figure()
ax = plt.axes()
ax.plot(x, fx)
plt.show()

# %%
'''
何謂機器學習
'''
