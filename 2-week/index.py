#!/usr/bin/env python3
import numpy as np
import pandas as pd
import datetime
import pandas_datareader.data as web
#from pandas import Series, Dataframe

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

import matplotlib.pyplot as plt
from matplotlib import style

# Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8, 7))
mpl.__version__

# Adjusting the style of matplotlib
style.use('ggplot')

# Search data from now to three years back
end = datetime.datetime.now()
start = end.replace(day=end.day-1, month=end.month, year=end.year - 3, hour=0, minute=0, second=0, microsecond=0)
print(start)
print(end)

#Get the data
df = web.DataReader("INTC", 'yahoo', start, end)

# Select column for calculations
close_px = df['Adj Close']

####################
# Ridge Regression
####################
window_size = 32
num_samples = len(df) - window_size
indices = np.arange(num_samples).astype(np.int)[:,None] + np.arange(window_size +1).astype(np.int)

# Create a 2D matrix such that we have 33 columns per example.
# The training samples will be the first 32 columns with the
# target variable being the last column.
# [[   0    1    2 ...   30   31   32]
#  [   1    2    3 ...   31   32   33]
#  [   2    3    4 ...   32   33   34]
data = close_px.values[indices]

X = data[:,:-1]
y = data[:,-1]

# Training and test split
split_ration = 0.7
ind_split = int(split_ration * num_samples)
X_train = X[:ind_split]
y_train = y[:ind_split]
X_test = X[ind_split:]
y_test = y[ind_split:]

# Train the model
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# Infer
y_pred_train_ridge = ridge_model.predict(X_train)
y_pred_ridge = ridge_model.predict(X_test)

df_ridge = df.copy()
df_ridge.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_ridge = df_ridge.iloc[window_size:ind_split] # Past 32 days unknonw
df_ridge['Adj Close Train'] = y_pred_train_ridge[:-window_size]
df_ridge.plot(label='INTC', figsize=(16,8), title='Adjusting Closing Price', grid=True)

# Same for the test
df_ridge = df.copy()
df_ridge.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_ridge = df_ridge.iloc[ind_split+window_size:] # Past 32 days we don't know yet
df_ridge['Adj Close Test (Ridge)'] = y_pred_ridge
df_ridge.plot(label='INTC (Ridge)', figsize=(16,8), title='Adjusted Closing Price (Ridge)', grid=True)



####################
# 2nd Lasso
####################
l_model = Lasso()
l_model.fit(X_train, y_train)

# Indef
y_pred_train_lasso = l_model.predict(X_train)
y_pred_lasso = l_model.predict(X_test)

# Plot
df_lasso = df.copy()
df_lasso.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_lasso = df_lasso.iloc[window_size:ind_split] # Past 32 days are unknown
df_lasso['Adj Close Train (Lasso)'] = y_pred_train_lasso[:-window_size]
df_lasso.plot(label='INTC (Lasso)', figsize=(16,8), title='Adjusted Closing Price (Lasso)', grid=True)

# Same for the test data
df_lasso = df.copy()
df_lasso.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_lasso = df_lasso.iloc[ind_split+window_size:] # Past 32 days we don't know yet
df_lasso['Adj Close Test'] = y_pred_lasso
df_lasso.plot(label='INTC (Lasso)', figsize=(16,8), title='Adjusted Closing Price (Lasso)', grid=True)

####################
# 3rd KNeighborsRegressor
####################
knr_model = KNeighborsRegressor()
knr_model.fit(X_train, y_train)

# Indef
y_pred_train_knr = knr_model.predict(X_train)
y_pred_knr = knr_model.predict(X_test)

# Plot
df_knr = df.copy()
df_knr.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_knr = df_knr.iloc[window_size:ind_split] # Past 32 days are unknown
df_knr['Adj Close Train (knr)'] = y_pred_train_knr[:-window_size]
df_knr.plot(label='INTC (knr)', figsize=(16,8), title='Adjusted Closing Price (knr)', grid=True)

# Same for the test data
df_knr = df.copy()
df_knr.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
df_knr = df_knr.iloc[ind_split+window_size:] # Past 32 days we don't know yet
df_knr['Adj Close Test'] = y_pred_knr
df_knr.plot(label='INTC (knr)', figsize=(16,8), title='Adjusted Closing Price (knr)', grid=True)

plt.legend()
plt.show()