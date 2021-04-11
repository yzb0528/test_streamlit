#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:15:21 2021

@author: yasuhirokanno
"""

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from sklearn.linear_model import HuberRegressor
from sklearn.datasets import make_regression

# デモデータの生成

rng = np.random.RandomState(0)
x, y, coef = make_regression( n_samples=200, n_features=1, noise=4.0, coef=True, random_state=0)
x[:4] = rng.uniform(10, 20, (4, 1))
y[:4] = rng.uniform(10, 20, 4)
df = pd.DataFrame({
    'x_axis': x.reshape(-1,),
    'y_axis': y
     }) 

# ロバスト回帰のパラメータを設定

epsilon = st.slider('Select epsilon', 
          min_value=1.00, max_value=10.00, step=0.01, value=1.35)

# ロバスト回帰実行

huber = HuberRegressor(epsilon=epsilon
    ).fit(
    df['x_axis'].values.reshape(-1,1), 
    df['y_axis'].values.reshape(-1,1)
    )

# 散布図の生成

plot = alt.Chart(df).mark_circle(size=40).encode(
    x='x_axis',
    y='y_axis',
    tooltip=['x_axis', 'y_axis']
).properties(
    width=500,
    height=500
).interactive()

# ロバスト線形回帰の係数を取得

a1 = huber.coef_[0]
b1 = huber.intercept_

# 回帰直線の定義域を指定

x_min = df['x_axis'].min()
x_max = df['x_axis'].max()

# 回帰直線の作成

points = pd.DataFrame({
    'x_axis': [x_min, x_max],
    'y_axis': [a1*x_min+b1, a1*x_max+b1],
})

line = alt.Chart(points).mark_line(color='steelblue').encode(
    x='x_axis',
    y='y_axis'
    ).properties(
    width=500,
    height=500
    ).interactive()

# グラフの表示

st.write(plot+line)