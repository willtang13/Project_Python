# -*- coding: utf-8 -*-
# %%
from importlib_metadata import DeprecatedTuple
from pyvizml import CreateNBAData
from pyvizml import ImshowSubplots
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import datasets
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
# %%
cnd = CreateNBAData(2021)
player_stats = cnd.create_player_stats_df()

# %%
pos_dict = {
    0: 'G',
    1: 'F'
}
pos = player_stats['pos'].values
pos_binary = np.array([0 if p[0] == 'G' else 1 for p in pos])
X = player_stats[['apg', 'rpg']].values.astype(float)
y = pos_binary
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
# %% 定義深度學習模型的結構
model = models.Sequential([
    Input(X_train.shape[1]),
    layers.Dense(4, activation='sigmoid'),
    layers.Dense(1, activation='sigmoid')
])
model.summary()
# %% 定義評估指標
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
# 最適化係數向量
n_iters = 5
model.fit(X_train, y_train,
          validation_data=(X_valid, y_valid),
          epochs=n_iters)
# %%
model.get_weights()
# %%
from pyvizml import DeepLearning
dl = DeepLearning([2, 4, 1])
dl.fit(X_train, y_train)
# %%
resolution = 50
apg = player_stats['apg'].values.astype(float)
rpg = player_stats['rpg'].values.astype(float)
X1 = np.linspace(apg.min() - 0.5, apg.max() + 0.5, num=resolution).reshape(-1, 1)
X2 = np.linspace(rpg.min() - 0.5, rpg.max() + 0.5, num=resolution).reshape(-1, 1)
APG, RPG = np.meshgrid(X1, X2)
Y_hat = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        xx_ij = APG[i, j]
        yy_ij = RPG[i, j]
        X_plot = np.array([xx_ij, yy_ij]).reshape(1, -1)
        z = dl.predict(X_plot)[0]
        Y_hat[i, j] = z
# %%
fig, ax = plt.subplots()
CS = ax.contourf(APG, RPG, Y_hat, alpha=0.2, cmap='RdBu')
colors = ['red', 'blue']
unique_categories = np.unique(y)
for color, i in zip(colors, unique_categories):
    xi = apg[y == i]
    yi = rpg[y == i]
    ax.scatter(xi, yi, c=color, edgecolor='k', label="{}".format(pos_dict[i]), alpha=0.6)
ax.set_title("Decision boundary of Forwards vs. Guards")
ax.set_xlabel("Assists per game")
ax.set_ylabel("Rebounds per game")
ax.legend()
plt.show()
# %% MNIST Data

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
iss = ImshowSubplots(3, 5, (8, 6))
iss.im_show(X_train, y_train)# %%

# %% Fashion MNIST
# from pyvizml import ImshowSubplots
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
fashion_mnist_labels = {
    0: "T-shirt/top",  # index 0
    1: "Trouser",      # index 1
    2: "Pullover",     # index 2
    3: "Dress",        # index 3
    4: "Coat",         # index 4
    5: "Sandal",       # index 5
    6: "Shirt",        # index 6
    7: "Sneaker",      # index 7
    8: "Bag",          # index 8
    9: "Ankle boot"    # index 9
}
iss = ImshowSubplots(3, 5, (8, 6))
iss.im_show(X_train, y_train, label_dict=fashion_mnist_labels)
# %%
w, h = 28, 28
X_train, X_valid = X_train[5000:], X_train[:5000]
y_train, y_valid = y_train[5000:], y_train[:5000]
X_train = X_train.reshape(X_train.shape[0], w*h)
X_valid = X_valid.reshape(X_valid.shape[0], w*h)
X_test = X_test.reshape(X_test.shape[0], w*h)
y_train = utils.to_categorical(y_train, 10)
y_valid = utils.to_categorical(y_valid, 10)
y_test = utils.to_categorical(y_test, 10)
# %%
model = models.Sequential([
    Input(28*28),
    layers.Dense(2**8, activation='sigmoid'),
    layers.Dense(2**8, activation='sigmoid'),
    layers.Dense(2**8, activation='sigmoid'),
    layers.Dense(10, activation='softmax')
])
model.summary()
# %%
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
n_iters = 5
model.fit(X_train, y_train,
          validation_data=(X_valid, y_valid),
          epochs=n_iters)
# %%
y_hat = model.predict(X_test)
fig, axes = plt.subplots(3, 5, figsize=(12, 6))
for i, index in enumerate(np.random.choice(X_test.shape[0], size=15, replace=False)):
    row_idx = i % 3
    col_idx = i // 3
    axes[row_idx, col_idx].imshow(X_test[index].reshape(w, h), cmap="Greys")
    axes[row_idx, col_idx].set_xticks([])
    axes[row_idx, col_idx].set_yticks([])
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    axes[row_idx, col_idx].set_title(
        "{} ({})".format(fashion_mnist_labels[predict_index], fashion_mnist_labels[true_index]),
        color=("green" if predict_index == true_index else "red")
    )
plt.show()

# %%
score = model.evaluate(X_valid, y_valid)
loss, accuracy = score
print("Accuracy: {:.2f}%".format(accuracy*100))
# %%
