# -*- coding:utf_8 -*-
# %%
from pyvizml import CreateNBAData
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# %%
cnd = CreateNBAData(season_year=2021)
player_stats = cnd.create_player_stats_df()
# %% 評估數值預測任務: mse, mae
X = player_stats['heightMeters'].values.astype(float).reshape(-1, 1)
y = player_stats['weightKilograms'].values.astype(float)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LinearRegression()
h.fit(X_train, y_train)
y_pred = h.predict(X_valid)
mse_valid = mean_squared_error(y_valid, y_pred)
print(mse_valid)
mae_valid = mean_absolute_error(y_valid, y_pred)
print(mae_valid)
# %% 評估類別預測任務: confusion matrix
X = player_stats[['apg', 'rpg']].values.astype(float)
pos_dict = {
    0: 'G',
    1: 'F'
}
pos = player_stats['pos'].values
y = np.array([0 if p[0] == 'G' else 1 for p in pos])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LogisticRegression()
h.fit(X_train, y_train)
y_pred = h.predict(X_valid)
cm = confusion_matrix(y_valid, y_pred)
print(cm)
# %% 
print(accuracy_score(y_valid, y_pred)) # accuracy = (TP+TN)/(TP+TN+FP+FN)
print(precision_score(y_valid, y_pred)) # precision = TP / (TP+FP)
print(recall_score(y_valid, y_pred)) # recall = TP / (TP+FN)
print(f1_score(y_valid, y_pred)) # f1 score = 2*(precision*recall)/(precision+recall)

# %% 自訂計算評估指標的類別
from pyvizml import ClfMetrics
pos = player_stats['pos'].values
y = np.array([0 if p[0] == 'G' else 1 for p in pos])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
h = LogisticRegression()
h.fit(X_train, y_train)
y_pred = h.predict(X_valid)

# %%
clf_metrics = ClfMetrics(y_valid, y_pred)
clf_metrics.confusion_matrix()
# %%
print(clf_metrics.accuracy_score())
print(clf_metrics.precision_score())
print(clf_metrics.recall_score())
print(clf_metrics.f1_score())

# %% 減少訓練誤差
shuffled_index = player_stats.index.values.copy()
np.random.seed(42)
np.random.shuffle(shuffled_index)
X = player_stats['heightMeters'].values.astype(float)[shuffled_index].reshape(-1, 1)
y = player_stats['weightKilograms'].values.astype(float)[shuffled_index]
kf = KFold(n_splits=5)
h = LinearRegression()
mse_scores = []
for train_index, valid_index in kf.split(X):
    X_train, y_train = X[train_index], y[train_index]
    X_valid, y_valid = X[valid_index], y[valid_index]
    h.fit(X_train, y_train)
    y_pred = h.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    mse_scores.append(mse)
mean_mse_scores = np.mean(mse_scores)
print(mse_scores)
print(mean_mse_scores)
# %%
fig = plt.figure()
ax = plt.axes()
ax.plot(mse_scores, marker='.', markersize=10)
ax.axhline(mean_mse_scores, color='red', linestyle='--')
ax.set_title('Use average MSE in KFold cross validation')
ax.set_xticks(range(5))
plt.show()
# %% XXXXXXXXXXXXX
X = player_stats['heightMeters'].values.astype(float).reshape(-1, 1)
X_plot = np.linspace(X.min() - 0.1, X.max() + 0.1).reshape(-1, 1)
y = player_stats['weightKilograms'].values.astype(float)
degrees = range(9)
y_plots = []
training_errors = []
for d in degrees:
    poly = PolynomialFeatures(d)
    X_poly = poly.fit_transform(X)
    X_train, X_valid, y_train, y_valid = train_test_split(X_poly, y, test_size=0.33, random_state=42)
    h = LinearRegression()
    h.fit(X_train, y_train)
    y_pred = h.predict(X_train)
    training_error = mean_squared_error(y_train, y_pred)
    training_errors.append(training_error)
    X_plot_poly = poly.fit_transform(X_plot)
    y_pred = h.predict(X_plot_poly)
    y_plots.append(y_pred)
# %%
x = X.ravel()
fig, axes = plt.subplots(3, 3, figsize=(12, 6), sharey=True)
for k, d, te, y_p in zip(range(9), degrees, training_errors, y_plots):
    i = k // 3
    j = k % 3
    x_p = X_plot.ravel() 
    axes[i, j].scatter(x, y, s=5, alpha=0.5)
    axes[i, j].plot(x_p, y_p, color="red")
    axes[i, j].set_ylim(60, 150)
    axes[i, j].set_title("Degree: {}\nTraining Error: {:.4f}".format(d, te))
plt.tight_layout()
plt.show()
# %% 減少訓練誤差與測試誤差的間距:  L2 正規化
X = player_stats['heightMeters'].values.astype(float).reshape(-1, 1)
y = player_stats['weightKilograms'].values.astype(float)
poly = PolynomialFeatures(9)
X_plot = np.linspace(X.min() - 0.1, X.max().max() + 0.1).reshape(-1, 1)
X_poly = poly.fit_transform(X)
X_plot_poly = poly.fit_transform(X_plot)
X_train, X_valid, y_train, y_valid = train_test_split(X_poly, y, test_size=0.33, random_state=42)
alphas = [0, 1, 10, 1e3, 1e5, 1e6, 1e7, 1e8, 1e9]
y_plots = []
for alpha in alphas:
    h = Ridge(alpha=alpha)
    h.fit(X_train, y_train)
    y_pred = h.predict(X_train)
    y_pred = h.predict(X_plot_poly)
    y_plots.append(y_pred)

# %%
x = X.ravel()
fig, axes = plt.subplots(3, 3, figsize=(12, 6), sharey=True)
for k, alpha, y_p in zip(range(9), alphas, y_plots):
    i = k // 3
    j = k % 3
    x_p = X_plot.ravel()
    axes[i, j].scatter(x, y, s=5, alpha=0.5)
    axes[i, j].plot(x_p, y_p, color="red")
    axes[i, j].set_ylim(60, 150)
    axes[i, j].set_title("L2 Regularization: {:.0f}".format(alpha))
plt.tight_layout()
plt.show()
# %%
