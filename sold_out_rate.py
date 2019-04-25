import pandas as pd
import numpy as np

df = pd.read_csv('sale_sku_group_2017.csv',encoding = 'GBK',low_memory = False)
stock = pd.read_csv('sku_group.csv',encoding = 'GBK',low_memory= False)
df = df.fillna(0)
#print (df.shape)
#print (stock.shape)

num_sale = []
num_stock = []


for i in range(0,53):
    temp = df.copy(deep=True)

    sum_tmp = 0
    for j in range(0,i+1):
        sum_tmp = sum_tmp+temp[str(j)]
    rate = sum_tmp
    num_sale.append(rate)
    #定义累积销量的index 和 column名
    key = [i for i in range(1,54)]
    values = num_sale
    dic  = dict(zip(key,values))
    sale  = pd.DataFrame(dic)
    sale.index = temp['sku_id']


    #sale.T.to_csv('aaaaaaaaa.csv')

    sku = temp['sku_id']

    # 首先使用np.array()函数把DataFrame转化为np.ndarray()，
    # 再利用tolist()函数把np.ndarray()转为list
    lt = np.array(sku)
    list_sorted = lt.tolist()

    temp2  = stock.copy()
    temp2['sku_id'] = temp2['sku_id'].astype('category').cat.set_categories(list_sorted)
    # 结果
    temp2 = temp2.dropna(axis=0)
    #temp2 = temp2.set_index('sku_id')

    stk = temp2['Quantity']

    num_stock.append(stk)

    #修改列名和行名，使sst 和 sale 两个表相同
    key_stk = [i for i in range(1,54)]
    values_stk = num_stock
    dic_stk  = dict(zip(key_stk,values_stk))
    sst  = pd.DataFrame(dic_stk)
    sst.index =  temp.ix[1:sst.shape[0],'sku_id']


    #sst = pd.DataFrame(num_stock)
    #sstt = sst.T
    #print (sale.head())
    #print (sst.head())
    #print (sst.index)
    #sstt = sst.rename(index = sale.index)
    mon = (sale / sst) * 100
    #print (mon.head())
    #rlt = pd.DataFrame(mon)

    mon.to_csv('sold_out_rate.csv')

