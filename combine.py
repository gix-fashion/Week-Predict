# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA

from pandas import read_csv
import math
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM

year09 = pd.read_csv('2009-pro.csv')
year10 = pd.read_csv('2010-pro.csv')
year11 = pd.read_csv('2011-pro.csv')
year12 = pd.read_csv('2012-pro.csv')
year13 = pd.read_csv('2013-pro.csv')
year14 = pd.read_csv('2014-pro.csv')
year15 = pd.read_csv('2015-pro.csv')
year16 = pd.read_csv('2016-pro.csv')

df_new = pd.concat([year09,year10,year11,year12,year13,year14,year15,year16],ignore_index=True)
del df_new['Unnamed: 0']
del df_new['num_brown']
del df_new['num_yellow']
del df_new['num_red']
del df_new['num_kaqi']
del df_new['num_green']

scaler = preprocessing.StandardScaler()
index = ['max_price','min_price','num_disc','mean_price','num_nodisc','size_large','size_medium','size_small']
for i in index:

    df_new[i] = scaler.fit_transform(df_new[i].values.reshape(-1, 1))

#print df_new

#df_new = df_new.astype(float)
#print type(df_new)
date = pd.date_range('2009/1/1','2016/12/1',freq='MS')
#print len(data)

#print type(date)
date_new = pd.to_datetime(date)


df_new.insert(0,'date',date_new)
#print df_new
save = df_new.to_csv('result.csv')

'''
#dataset = df_new.sale_vol
#plt.plot(dataset)
#plt.show()

#autocorrelation_plot(dataset)
#plt.show()
# 从图中得到 ar参数起点10

#print type(df_new)
#print df_new.info()
print df_new
final = df_new.as_matrix()# dataframe to ndarray
#print final
np.insert(final,0,date,axis=1)


print type(final)

print final


numpy的插入
numpy.insert可以有三个参数（arr，obj，values），也可以有4个参数（arr，obj，values，axis）：
第一个参数arr是一个数组，可以是一维的也可以是多维的，在arr的基础上插入元素
第二个参数obj是元素插入的位置
第三个参数values是需要插入的数值
第四个参数axis是指示在哪一个轴上对应的插入位置进行插入

# fit model

X = final.sale_vol
size = int(len(X)*0.66)
train,test = X[0:size],X[size:len(X)]
pre = [X for X in train]
pred = list()
model = ARIMA(pre, order=(10,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

'''
