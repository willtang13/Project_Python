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
from pyvizml import CreateNBAData
cnd = CreateNBAData(2020)
player_stats = cnd.create_player_stats_df()
# %%

X = player_stats['heightMeters'].values.reshape(-1,1)
y = player_stats['weightKilograms'].values
lr = LinearRegression()
h = lr.fit(X,y)
print(h.predict(np.array([[1.90]]))[0]) # 預測身高 190 公分 NBA 球員的體重
print(h.predict(np.array([[1.98]]))[0]) # 預測身高 198 公分 NBA 球員的體重
print(h.predict(np.array([[2.03]]))[0]) # 預測身高 203 公分 NBA 球員的體重
# %%
unique_pos = player_stats['pos'].unique()
pos_dict = {i : p for i, p in enumerate(unique_pos)}
pos_dict_reversed = {v : k for k, v in pos_dict.items()}
print(pos_dict)
print(pos_dict_reversed)

#%%
X = player_stats[['apg', 'rpg']].values
pos = player_stats['pos'].map(pos_dict_reversed)
y = pos.values
logit = LogisticRegression()
h = logit.fit(X, y)
print(pos_dict[h.predict(np.array([[5, 5]]))[0]])
print(pos_dict[h.predict(np.array([[5, 10]]))[0]])
print(pos_dict[h.predict(np.array([[5, 15]]))[0]])

# %%
