# -*- coding: utf-8 -*-
# %%
from black import Line
from pyvizml import CreateNBAData
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# %%
cnd = CreateNBAData(2020)
players = cnd.create_players_df()
y = players['weightKilograms'].values.astype(float)
print(y.dtype)
# %%
X = players['heightMeters'].values.astype(float).reshape(-1, 1)
y = players['weightKilograms'].values.astype(float)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LinearRegression()
h.fit(X_train, y_train)
print(h.intercept_) # 截距項
print(h.coef_)      # 係數項
# %%
# 預測
y_pred = h.predict(X_valid)
y_pred[:10]

# %% 
# 創建迴歸線的資料
X1 = np.linspace(X.min()-0.1, X.max()+0.1).reshape(-1, 1)
y_hat = h.predict(X1)

fig, ax = plt.subplots()
ax.scatter(X_train.ravel(), y_train, label='training')
ax.scatter(X_valid.ravel(), y_valid, label='valid')
ax.scatter(X_valid.ravel(), y_pred, label='predict')
ax.plot(X1.ravel(), y_hat, c='red', label='regression')
ax.legend()
plt.show()
# %%
from pyvizml import NormalEquation
h2 = NormalEquation()
h2.fit(X_train, y_train)
print(h2.intercept_)
print(h2.coef_)
# %% 
X0 = np.ones((10, 1))
X1 = np.arange(1, 11).reshape(-1, 1)
w = np.array([5, 6])
X_train = np.concatenate([X0, X1], axis=1)
y_train = np.dot(X_train, w)
print(X_train)
print(w)
print(y_train)
# %%
np.random.seed(42)
w = np.random.rand(2)
y_hat =np.dot(X_train, w)
print(y_hat)
# %%
m = y_train.size
j = ((y_hat - y_train).T.dot(y_hat - y_train)) / m
print(j)

# %%
gradients = (2/m) * np.dot(X_train.T, y_hat - y_train)
print(gradients)
# %%
learning_rate = 0.001
print(-learning_rate * gradients)
# %%
w -= learning_rate * gradients
# %%
y_hat = np.dot(X_train, w)
print(y_hat)
j = ((y_hat - y_train).T.dot(y_hat - y_train)) / m
print(j)

# %% 梯度遞減
from pyvizml import GradientDescent
h = GradientDescent(fit_intercept=False)
h.fit(X_train, y_train, epochs=20000, learning_rate=0.001)
print(h.intercept_)
print(h.coef_)
# %%
X = players['heightMeters'].values.astype(float).reshape(-1, 1)
y = players['weightKilograms'].values.astype(float)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.33, random_state=42)
hg = GradientDescent()
hg.fit(X_train, y_train, epochs=30000, learning_rate=0.01)
print(h.intercept_)
print(h.coef_)
y_pred = hg.predict(X_valid)
print(y_pred[:10])
# %% 標準化與進階的梯度遞減
train = pd.read_csv("https://kaggle-getting-started.s3-ap-northeast-1.amazonaws.com/house-prices/train.csv")
X = train['GrLivArea'].values.reshape(-1, 1)
y = train['SalePrice'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.33, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_)
print(lr.coef_)
# %%
h = GradientDescent()
h.fit(X_train, y_train, epochs=500000, learning_rate=1e-7)
# %%
print(h.intercept_)
print(h.coef_)
# %%
def plot_contour(X_train, y_train, w_history, w_0_min, w_0_max, w_1_min, w_1_max, w_0_star, w_1_star):
    m = X_train.shape[0]
    X0 = np.ones((m, 1), dtype=float)
    X_train = np.concatenate([X0, X_train], axis=1)
    resolution = 100
    W_0, W_1 = np.meshgrid(np.linspace(w_0_min, w_0_max, resolution), np.linspace(w_1_min, w_1_max, resolution))
    Z = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            w = np.array([W_0[i, j], W_1[i, j]])
            y_hat = np.dot(X_train, w)
            mse = ((y_hat - y_train).T.dot(y_hat - y_train)) / m
            Z[i, j] = mse
    epochs = len(w_history)
    w_0_history = []
    w_1_history = []
    for i in range(epochs):
        w_0_history.append(w_history[i][0])
        w_1_history.append(w_history[i][1])
    fig, ax = plt.subplots()
    CS = ax.contour(W_0, W_1, Z)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.plot(w_0_history, w_1_history, "-", color="blue")
    ax.scatter(w_0_star, w_1_star, marker="*", color="red")
    ax.set_xlabel("$w_0$")
    ax.set_ylabel("$w_1$", rotation=0)
    plt.show()

# %%
w_history = h._w_history
plot_contour(X_train, y_train, w_history, -5000, 35000, -10, 200, lr.intercept_, lr.coef_[0])
# %%
train = pd.read_csv("https://kaggle-getting-started.s3-ap-northeast-1.amazonaws.com/house-prices/train.csv")
X = train['GrLivArea'].values.reshape(-1, 1)
y = train['SalePrice'].values
mms = MinMaxScaler()
X_scaled = mms.fit_transform(X)
y = train['SalePrice'].values
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, train_size=0.33, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_)
print(lr.coef_)
# %%
h = GradientDescent()
h.fit(X_train, y_train, epochs=100000, learning_rate=0.01)
print(h.intercept_) # 截距項
print(h.coef_)      # 係數項
# %%
intercept_rescaled = h.intercept_ - (h.coef_ * mms.data_min_ / mms.data_range_)
coef_rescaled = h.coef_ / mms.data_range_
print(intercept_rescaled)
print(coef_rescaled)
# %%
ss = StandardScaler()
X_ss_scaled = ss.fit_transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(X_ss_scaled, y, train_size=0.33, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.intercept_)
print(lr.coef_)

# %%
h = GradientDescent()
h.fit(X_train, y_train, epochs=10000, learning_rate=0.001)
print(h.intercept_)
print(h.coef_)
# %%
intercept_ss_rescaled = h.intercept_ - h.coef_ * ss.mean_ / ss.scale_
coef_rescaled = h.coef_ / ss.scale_
print('==rescaled w by using standard scaler==')
print(intercept_rescaled)
print(coef_rescaled)
# %% Adaptive Gradient Descent
from pyvizml import AdaGrad
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.33, random_state=42)
h_agd = AdaGrad()
h_agd.fit(X_train, y_train, epochs=500000, learning_rate=100)
print(h_agd.intercept_)
print(h_agd.coef_)

w_history = h_agd._w_history
plot_contour(X_train, y_train, w_history, -5000, 35000, -10, 200, h_agd.intercept_, h_agd.coef_[0])
# %%
