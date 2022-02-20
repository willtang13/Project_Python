# -*- coding: utf-8 -*-
# %%
import numpy as py
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston, fetch_california_housing, make_classification
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
# %%
from pyvizml import CreateNBAData
cnb =CreateNBAData(2020)
players = cnb.create_players_df()
X = players['heightMeters'].values.reshape(-1, 1)
y = players['weightKilograms'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)

# %% 模組化
ss = StandardScaler()
lr = LinearRegression()

pipeline = Pipeline([('scaler', ss), ('lr', lr)])
print(type(pipeline))
# %% 一致性
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_valid)
# %% 檢查性
print(lr.intercept_)
print(lr.coef_)

# %% 不自行創建類別
print(type(lr.intercept_))
print(type(lr.coef_))
# %%
players.shape
X = players[['heightMeters', 'heightInches']].values.astype(float)
y = players['weightPounds'].values.astype(float)
# %% 載入玩具資料集
X, y = load_boston(return_X_y=True)
print(X.shape)
print(y.shape)

# %% 載入現實世界資料集
X, y = fetch_california_housing(return_X_y=True)
print(X.shape)
print(y.shape)
# %% 載入生成資料集
X, y = make_classification()
print(X.shape)
print(y.shape)

# %% 預處理 polyfit轉換器
X = players[['heightFeet','heightInches']].values.astype(int)
X_before_poly = X.copy()
poly = PolynomialFeatures()
X_after_poly = poly.fit_transform(X_before_poly) # 高次項特徵轉換器輸出的 X: x_0, x_1, x_2, x_1**2, x_1*x_2, x_2**2
print(X_before_poly.shape)
print(X_after_poly.shape)
# %% 預處理 標準化
X_before_scaled = X.copy()
ms = MinMaxScaler()
ss = StandardScaler()
X_after_ms = ms.fit_transform(X_before_scaled)
X_after_ss = ss.fit_transform(X_before_scaled)
print(X_before_scaled[:10, :])
print(X_after_ms[:10, :])
print(X_after_ss[:10, :])


# %%
X = players[['heightFeet', 'heightInches']].values.astype(int)
y = players['weightKilograms'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
# 初始化
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_valid)

# %%
players_train, players_valid = train_test_split(players, test_size=0.3, random_state=42)
players_train.iloc[:5, :4]
players_valid.iloc[:5, :4]
# %%
train = pd.read_csv("https://kaggle-getting-started.s3-ap-northeast-1.amazonaws.com/titanic/train.csv")
test = pd.read_csv("https://kaggle-getting-started.s3-ap-northeast-1.amazonaws.com/titanic/test.csv")
print(train.shape)
print(test.shape)

train.columns.difference(test.columns) # 差別在 Survived 這個目標向量
# %%
