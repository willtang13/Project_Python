# -*- coding: utf-8 -*-
# %%
import enum
from pydoc import resolve
from pyVizML.pyvizml import LogitReg
from pyvizml import CreateNBAData
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# %% Define functions 
def plot_contour_filled(XX, YY, resolution=50):
    PROBA = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            xx_ij = XX[i, j]
            yy_ij = YY[i, j]
            X_plot = np.array([xx_ij, yy_ij]).reshape(1, -1)
            z = h.predict_proba(X_plot)[0, 1]
            PROBA[i, j] = z
    fig, ax = plt.subplots()
    CS = ax.contourf(XX, YY, PROBA, cmap='RdBu')
    ax.set_title("Probability of being predicted as a forward")
    ax.set_xlabel("Assists per game")
    ax.set_ylabel("Rebounds per game")
    fig.colorbar(CS, ax=ax)
    plt.show()
    
def plot_decision_boundary(XX, YY, x, y, target_vector, pos_dict, h, resolution=50):
    Y_hat = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            xx_ij = XX[i, j]
            yy_ij = YY[i, j]
            X_plot = np.array([xx_ij, yy_ij]).reshape(1, -1)
            z = h.predict(X_plot)
            Y_hat[i, j] = z
    fig, ax = plt.subplots()
    CS = ax.contourf(XX, YY, Y_hat, alpha=0.2, cmap='RdBu')
    colors = ['red', 'blue']
    unique_categories = np.unique(target_vector)
    for color, i in zip(colors, unique_categories):
        xi = x[target_vector == i]
        yi = y[target_vector == i]
        ax.scatter(xi, yi, c=color, edgecolor='k', label="{}".format(pos_dict[i]), alpha=0.6)
    ax.set_title("Decision boundary of Forwards vs. Guards")
    ax.set_xlabel("Assists per game")
    ax.set_ylabel("Rebounds per game")
    ax.legend()
    plt.show()
# %%
cnd = CreateNBAData(2021)
player_states = cnd.create_player_stats_df()
player_states['pos'].dtype
# %%
print(player_states['pos'].unique())
print(player_states['pos'].nunique())

pos_dict = {
    0: 'G',
    1: 'F'
}
pos = player_states['pos'].values
pos_binary = np.array([0 if p[0] == 'G' else 1 for p in pos])
print(np.unique(pos_binary))

# %%
X = player_states[['apg', 'rpg']].values.astype(float)
y = pos_binary
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.33, random_state=42)
h = LogisticRegression(C=1e06) # 預測器的正規化程度
h.fit(X_train, y_train)
print(h.intercept_)
print(h.coef_)

# %% 
p_hat = h.predict_proba(X_valid)
print(p_hat[:10, :]) #第 0 欄是預測為類別 0 的機率, 第 1 欄是預測為類別 1 的機率 

y_pred = np.argmax(p_hat, axis=1)
print(y_pred[:10])
y_pred_label = [pos_dict[n] for n in y_pred]
print(y_pred_label[:10])
# %%
resolution = 50
apg = player_states['apg'].values.astype(float)
rpg = player_states['rpg'].values.astype(float)
X1 = np.linspace(apg.min() - 0.5, apg.max() + 0.5, num=resolution)
X2 = np.linspace(rpg.min() - 0.5, rpg.max() + 0.5, num=resolution)
APG, RPG = np.meshgrid(X1, X2)


# %%
plot_contour_filled(APG, RPG, resolution)
# %% Decision Boundary
plot_decision_boundary(APG, RPG, apg, rpg, y, pos_dict, h, 50)
# %% 羅吉斯迴歸
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def plot_sigmoid():
    x = np.linspace(-6, 6, 100)
    y = sigmoid(x)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, y)
    ax.axvline(0, color='black')
    for i in [0.0, 0.5, 1.0]:
        ax.axhline(y = i, ls=':', color='k', alpha=0.5)
    ax.set_ytick([0.0, 0.5, 1.0])
    ax.set_xlim(-0.1, 1.1)
    ax.set_title('Sigmoid functioin')
    plt.show()

plot_sigmoid()
# %% 羅吉斯迴歸使用交叉熵（Cross-entropy）函式作為cost function
def plot_cross_entropy():
    epsilon = 1e-5
    h = np.linspace(epsilon, 1-epsilon)
    y1 = -np.log(h)
    y2 = -np.log(1 - h)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(h, y1)
    ax[0].set_title('$y=1$\n$-\log(\sigma(Xw))$')
    ax[0].set_xticks([0, 1])
    ax[0].set_xlabel('$\sigma(Xw)$')
    ax[1].plot(h, y2)
    ax[1].set_title('$y=0$\n$-\log(1-\sigma(Xw))$')
    ax[1].set_xticks([0, 1])
    ax[1].set_xlabel('$\sigma(Xw)$')
    plt.tight_layout()
    plt.show()

plot_cross_entropy()    
    
# %%
from pyvizml import LogitReg
h = LogitReg()
h.fit(X_train, y_train, 100000, 0.01)
print(h.intercept_)
print(h.coef_)
# %%
y_pred = h.predict(X_valid)
y_pred_label = [pos_dict[i] for i in y_pred]
y_pred_label[:10]


# %% 二元分類延伸至多元分類：One versus rest
pos = player_states['pos'].values
print(np.unique(pos))

# %%
unique_pos = player_states['pos'].unique()
pos_dict = {i: p for i, p in enumerate(unique_pos)}
pos_dict_reversed = {v: k for k, v in pos_dict.items()}
pos_multiple = player_states['pos'].map(pos_dict_reversed)
print(pos_dict)
print(pos_dict_reversed)
print(np.unique(pos_multiple))
# %%
X = player_states[['apg', 'rpg']].values.astype(float)
y = pos_multiple
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LogisticRegression(C=1e5, multi_class='ovr')
h.fit(X_train, y_train)
p_hat = h.predict_proba(X_valid)
p_hat[:10]

# %%
y_pred = np.argmax(p_hat, axis=1)
y_pred_label = [pos_dict[i] for i in y_pred]
print(y_pred_label[:10])
# %% 二元分類延伸至多元分類：Softmax 函式
h_sm = LogisticRegression(C=1e5, multi_class='multinomial')
h_sm.fit(X_train, y_train)
p_hat = h_sm.predict_proba(X_valid)
p_hat[:10]
# %%
y_pred = np.argmax(p_hat, axis=1)
y_pred[:10]
# %%
y_pred_label = [pos_dict[i] for i in y_pred]
y_pred_label[:10]

# %% 兩種表示類別向量的形式
le = LabelEncoder()
pos = player_states['pos'].values
pos_le = le.fit_transform(pos)
print(pos[:10])
print(pos_le[:10])

# %% 獨熱編碼（One-hot encoder）
ohe = OneHotEncoder()
pos_ohe = ohe.fit_transform(pos.reshape(-1, 1)).toarray()
print(pos[:10])
print(pos_ohe[:10])
# %%
