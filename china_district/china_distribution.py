import pandas as pd
import numpy as np


'''
###########################step1：按照省划分数据集



china = pd.read_csv('sales_total_sku_channel.csv',encoding='utf-8',low_memory=False)

#按省分组
op = china.groupby(['province']).sum()
#获取省的list

province = op._stat_axis.values.tolist()

#print (province)

['上海市', '云南省', '内蒙古自治区', '北京市', '吉林省', '四川省', '天津市', 
 '宁夏回族自治区', '安徽省', '山东省', '山西省', '广东省', '广西壮族自治区', '新疆维吾尔自治区',
 '江苏省', '江西省', '河北省', '河南省', '浙江省', '海南省', '湖北省', '湖南省', '甘肃省', '福建省',
 '贵州省', '辽宁省', '重庆市', '陕西省', '青海省', '黑龙江省']


#获取省份的数据
for i in province:
    new = china[china['province'].isin([i])]

    new.to_csv('district_data/%s.csv'%i)




#####################step2:得到多年来总销量>800的sku（sku的选择策略可以再优化）



df1 = pd.read_csv("/Users/rui/Documents/new-jeans/sale/ALL-sale.csv",encoding='UTF-8',low_memory=False)

df1 = df1[(df1.Quantity>0)&(df1.sale_price>0)]
#去掉缺失值
df1 = df1.dropna(axis = 0)
# 去掉不必要的数据列
df1 = df1.drop(['sale_no','channel_id','sale_price'],axis = 1)

df1['sale_date'] = pd.to_datetime(df1['sale_date'])
date = df1.set_index('sale_date')

#按周显示
output = date.to_period('w')

data = df1.groupby(['sku_id']).sum()
#多年来总销量 >800
data = data[data['Quantity'] > 800]

print (data)
data.to_csv('0519_sku_sale_situation.csv')



#################step3:得到各省step2得到的sku的周销量



#获取sku
df = pd.read_csv('0519_sku_sale_situation.csv')
df.reset_index()
#print (df)

id = np.array(df['sku_id'])
idlist = id.tolist()

#print (idlist)

all = pd.read_csv('district_data/黑龙江省.csv',encoding='utf-8',low_memory= False)
#2008-2018年的数据
all['sale_date'] = pd.to_datetime(all['sale_date'])
all = all.set_index('sale_date')
all = all.sort_values('sale_date')

#无折扣数量、折扣率、周内最高价格、周内最低价格、周内价格均值、周内价格标准差、周销量


all['discount'] = all['sale_price'] / all['tag_price']  # 增加折扣特征
all['discount'][all['discount'] < 1] = 0
for i in idlist:
    new = all[all['sku_id'].isin([i])]
    #print (new)
    #sale_price', 'Quantity', 'long', 'years', 'tag_price', 'designer',
           #'barcode'],

    new = new.drop(['sale_price', 'long', 'years', 'tag_price','designer','barcode'], axis=1)

    output = new.resample('w').sum()


    output['rate_dis'] = (output['Quantity'] - output['discount']) / output['Quantity']

    dd = output.pop('Quantity')
    output['week_sal'] = dd
    output.to_csv('HeiLongJiang/%s.csv'%i)

'''
# -*- coding: utf-8 -*-
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
import math
import numpy as np
import tensorflow as tf
#cpu_num = int(os.environ.get('CPU_NUM', 4))
#config = tf.ConfigProto(device_count={"CPU": cpu_num},
                       # inter_op_parallelism_threads=cpu_num,
                        #intra_op_parallelism_threads=cpu_num,
                       # log_device_placement=True)

#with tf.Session(config=config) as sess:
err = []
out_y_hat = []
final_pian = []

#某个省份sku周销量的数据

os.chdir('/Users/rui/PycharmProjects/project/AnHui')
file_chdir = os.getcwd()

for root, dirs, files in os.walk(file_chdir):
    for file in files:

        ddf = pd.read_csv(file, encoding='gb18030', header=0, index_col=0)
        # print(ddf)
        # print df.shape
        # del df[u'销售日期']
        ddf = ddf.drop(['Unnamed: 0','nation','county','channel_type','area','shopping_guide_quantity'], axis=1)
        # print df.head()
        ddf = ddf.fillna(0)
        # print (ddf.head())

        values = ddf.values
        values = values.astype('float32')
        scaler = MinMaxScaler(copy=False, feature_range=(0, 1))

        # print scaler

        scaled = scaler.fit_transform(values)
        # print scaled

        x = scaled[:, :2]
        # print (x)
        y = scaled[:, 2]
        # print (y)

        # x = df.iloc[:,:13]
        # y = df.iloc[:,14]

        train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.1)

        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        # design network
        model = Sequential()
        model.add(LSTM(40, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        # fit network
        history = model.fit(train_X, train_y, epochs=40, batch_size=6, validation_data=(test_X, test_y), verbose=2,
                            )
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        yhat = model.predict(test_X)
        # print 'yhat',yhat.shape
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # print test_X,test_X.shape

        inv_yhat = concatenate((test_X[:, 0:], yhat), axis=1)
        # print (inv_yhat)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 2]

        # print ('yhat:',inv_yhat)
        ffffi = []

        for i in inv_yhat:
            if i < 0:
                i = 0
            else:
                i = i
            ffffi.append(i)
        out_y_hat.append(ffffi)

        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_X[:, 0:], test_y), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 2]
        # print ('y:',inv_y)

        pian = []

        for k, h in zip(inv_yhat, inv_y):
            piancha = abs(k - h)
            pian.append(piancha)

            rlt_pian = np.var(pian)
        final_pian.append(rlt_pian)

        # fff = sum(final_pian)/
        # print error

        # error1 = sum(error) / len(inv_y)
        # print 'test error:',error1
        # err.append(error1)

# print ('err:',err)
# print ('var:',final_pian)
# print ('y_hat:',out_y_hat)

dic_var = {'var': final_pian}

out_dic_var = pd.DataFrame(dic_var)
# print (out_dic_var)

out_dic_yhat = pd.DataFrame(out_y_hat)

# print (out_dic_yhat)

out_dic_yhat.insert(0, 'var', out_dic_var['var'])
# print ('final:',out_dic_yhat)
out_dic_yhat.to_csv('0519_AnHui_predict.csv')



