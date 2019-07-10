# coding: utf-8
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

import math

def process(file):
    life = pd.DataFrame(columns=('year','company','season','wave','goodsyear','life_circle'))
    df_ori = pd.read_csv(file)
    #print (df_ori)

    df_ori = df_ori.dropna()
    dict = {}
    dict.update(dict.fromkeys(['第1波段','第一波段'], 1))
    dict.update(dict.fromkeys(['第二波段','第2波段'], 2))
    dict.update(dict.fromkeys(['第3波段','第三波段'], 3))
    dict.update(dict.fromkeys(['第4波段','第四波段'], 4))
    dict.update(dict.fromkeys(['第5波段','第五波段'], 5))
    dict.update(dict.fromkeys(['第6波段','第六波段'], 6))
    dict.update(dict.fromkeys(['第7波段','第七波段'], 7))
    dict.update(dict.fromkeys(['第8波段','第八波段'], 8))
    dict.update(dict.fromkeys(['第十六周', '第十九周'], 16))
    dict.update(dict.fromkeys(['第十七周', '第十九周'], 17))
    dict.update(dict.fromkeys(['第十八周', '第十九周'], 18))
    dict.update(dict.fromkeys(['第十九周', '第十九周'], 19))

    df_ori['波段'] = df_ori['波段'].map(dict)

    dic = {}
    dic.update(dict.fromkeys(['秋','秋薄'], '秋'))
    dic.update(dict.fromkeys(['春'], '春'))
    dic.update(dict.fromkeys(['夏'], '夏'))
    dic.update(dict.fromkeys(['冬'], '冬'))
    df_ori['季节'] = df_ori['季节'].map(dic)


    df_ori = df_ori[df_ori['小类'].isin(['长袖衬衫'])]
    df_ori = df_ori[~ df_ori['分公司'].isin(['总部'])]

    a = []
    life_circle = []
    year = ['2015','2016','2017','2018','2019']
    year2 = ['2015','2016','2017','2018','2019']
    company = ['华中1分公司','华中2分公司','华南分公司','西南分公司','华北分公司','西北分公司']
    season = ['春','夏','秋','冬']
    type = [1,2,3,4,5,6,7,8,16,17,18,19]
    for j in year:
        for jj in year2:
            for k in company:
                for m in season:
                    for n in type:
                        #print (j,k,m,n)
                        if(j!=2017) and (k!='西北分公司') and (m != '夏' )and( n!=3):
                            df = df_ori
                            df = df[df['年'].isin([j])]
                            df = df[df['商品年份'].isin([jj])]
                            df = df[df['分公司'].isin([k])]
                            df = df[df['季节'].isin([m])]
                            df = df[df['波段'].isin([n])]
                            #print (df)

                            df = df.sort_values(by=['年','周'],ascending=True)


                            df.to_csv('rf-v1.0-data.csv',index = 0 )

                            new = pd.read_csv('rf-v1.0-data.csv')



                            #new =pd.read_csv('new_data.csv')

                            if (new.shape)[0]>=8:
                                #for i in range((new.shape)[0]):
                                #统计生命周期长度
                                #打折 并且 销售量<top*0.2 则停止
                                sum_year_sale = max(new['本周销售数量'].values)
                                new['life_I'] = new['本周销售数量'] - sum_year_sale*0.2

                                new['discount'] = new['本周销售金额']/new['本周销售吊牌金额']
                                new = new[new['discount'] >= 0.99]
                                new = new[new['life_I']>0]
                                #print (new)
                                new['life_II'] = new['本周销售数量']/sum(new['本周销售数量'])

                                #print ('aa',new['life_II'].mean() + 1 * new['life_II'].std())
                                #print ('bb',new['life_II'])
                                alpha = new['life_II'].mean() + 1 * new['life_II'].std()
                                beita = new['life_II'].mean() - 1 * new['life_II'].std()
                                alpha_2 = new['life_II'].mean() + 2 * new['life_II'].std()
                                beita_2 = new['life_II'].mean() - 2 * new['life_II'].std()
                                alpha_3 = new['life_II'].mean() + 3 * new['life_II'].std()
                                beita_3 = new['life_II'].mean() - 3 * new['life_II'].std()
                                new = new[new['life_II'] < alpha]
                                new = new[new['life_II'] > beita]
                                new = new[new['life_II'] < alpha_2]
                                new = new[new['life_II'] > beita_2]
                                new = new[new['life_II'] < alpha_3]
                                new = new[new['life_II'] > beita_3]

                                #print (new)
                                #print ('aaaaa',j,jj,k,m,n,new.shape[0])

                                life = life.append(({'year': j, 'company': k, 'season': m, 'wave': n,
                                                     'goodsyear': jj, 'life_circle': new.shape[0]}), ignore_index=True)
                            #print (life)


    #final = pd.DataFrame(life_circle)
    life = life.dropna()
    life = life[life['life_circle']>0]

    print (life)
    life.to_csv('life_cycle_origin.csv', index=0)

def life(file):
    final = pd.DataFrame()
    lc = pd.read_csv(file)

    year = ['2015', '2016', '2017', '2018', '2019']
    year2 = ['2015', '2016', '2017', '2018', '2019']
    company = ['华中1分公司', '华中2分公司', '华南分公司', '西南分公司', '华北分公司', '西北分公司']
    season = ['春', '夏', '秋', '冬']
    type = [1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18, 19]

    for jj in year2:
        for k in company:
            for m in season:
                for n in type:
                    # print (j,k,m,n)

                        df = lc
                        df = df[df['goodsyear'].isin([jj])]
                        df = df[df['company'].isin([k])]
                        df = df[df['season'].isin([m])]
                        df = df[df['wave'].isin([n])]
                        #print (df)
                        if (df.shape[0]>1):

                            final = final.append(df.iloc[0,:])
    print (final)


#process('v1.1-jeans.csv')
#process('v1.1-shirts.csv')
#process('/Users/rui/PycharmProjects/paper/未命名文件夹/22中类销售/T恤销售.csv')
life('life_cycle_origin.csv')
