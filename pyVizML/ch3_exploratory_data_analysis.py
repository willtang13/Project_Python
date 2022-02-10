# -*- coding: utf-8 -*-
'''
Exploratory Data Analysis (EDA) 
'''

# %%
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# from tensorflow.keras import datasets
# %% Matlab 風格
x = np.linspace(0, np.pi*4, 100)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))
plt.show()

# %% 物件導向風格
fig, axes = plt.subplots(2, 1)
axes[0].plot(x, np.sin(x))
axes[1].plot(x, np.cos(x))
plt.show()

# %%
fig = plt.figure()
fig.canvas.get_supported_filetypes()
# %%
fig, axes = plt.subplots(2, 1)
axes[0].plot(x, np.sin(x))
axes[1].plot(x, np.cos(x))
fig.savefig('my_figre.png')
# %%
for f in os.listdir():
    if '.png' in f:
        print(f)
# %% 散佈圖
x = np.linspace(-2*np.pi, 2*np.pi)
f = np.sin(x)
fig = plt.figure()
ax = plt.axes()
ax.scatter(x, f)
plt.show()

# %% 橫條圖
np.random.seed(42)
random_integers = np.random.randint(1, 100, size=100)
n_odds = np.sum(random_integers % 2 == 0)
n_evens = np.sum(random_integers % 2 == 1)
y = np.array([n_odds, n_evens])
x = np.array([1, 2])
fig = plt.figure()
ax = plt.axes()
ax.barh(x, y)
plt.show()
# %% 等高線圖
x = np.linspace(-3.0, 3.0)
y = np.linspace(-2.0, 2.0)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = 2 * (Z1 - Z2)
fig = plt.figure()
ax = plt.axes()
ax.contour(X, Y, Z, cmap='RdBu')
plt.show()
# %%
fig = plt.figure()
ax = plt.axes()
ax.contourf(X, Y, Z, cmap='RdBu')
plt.show()
# %%
