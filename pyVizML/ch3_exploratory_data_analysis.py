# -*- coding: utf-8 -*-
'''
Exploratory Data Analysis (EDA) 
'''

# %%
from distutils.command.build_scripts import first_line_re
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tensorflow.keras import datasets
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
fig.savefig('my_figure.png')
# %%
for f in os.listdir():
    if '.png' in f:
        print(f)
# %% 散佈圖
x = np.linspace(-2*np.pi, 2*np.pi)
f = np.sin(x)
g = np.cos(x)
fig = plt.figure()
ax = plt.axes()
ax.scatter(x, f, label='sin(x)')
ax.scatter(x, g, label='cos(x)')
ax.legend()
plt.show()

# %% 橫條圖
font_path = r'C:/Windows/Fonts/msjh.ttf' # 設定中文字體
tc_font = FontProperties(fname=font_path) # 設定中文字體
np.random.seed(42)
random_integers = np.random.randint(1, 100, size=100)
n_odds = np.sum(random_integers % 2 == 0)
n_evens = np.sum(random_integers % 2 == 1)
y = np.array([n_odds, n_evens])
x = np.array([1, 2])
fig = plt.figure()
ax = plt.axes()
ax.barh(x, y)
ax.set_title('Odd/even numbers in 100 random integers')
ax.set_xlabel('頻率', fontproperties=tc_font) # 設定中文字體
ax.set_ylabel('Type', rotation=0)
ax.set_yticks(x)
ax.set_yticklabels(['Odds', 'Evens'])
ax.set_xlim([0, 100])
for xi, yi in zip(x, y):
    ax.text(yi + 1, xi - 0.05, '{}'.format(yi))
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
CS = ax.contour(X, Y, Z, cmap='RdBu')
ax.clabel(CS, inline=1, fontsize=10)
plt.show()
# %%
fig = plt.figure()
ax = plt.axes()
CS = ax.contourf(X, Y, Z, cmap='RdBu')
fig.colorbar(CS)
plt.show()
# %% 顯示圖片
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data(path='mnist.npz')
first_picture = X_train[0, :, :]
first_picture.shape

fig = plt.figure()
ax = plt.axes()
ax.imshow(first_picture, cmap='Greys')
plt.show()

# %% 繪製子圖
fig, axes = plt.subplots(3, 5)
print(type(axes))
print(axes.shape)

# %%
from pyvizml import ImshowSubplots

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data(path='mnist.npz')
iss = ImshowSubplots(3, 5, (8, 6))
iss.im_show(X_train, y_train)
# 