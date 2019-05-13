
all = pd.read_csv('sales_total_sku.csv',encoding='utf-8',low_memory= False)

#输出看特征
#print (all.ix[0,:])

df = pd.read_csv('sku_sale_situation_small.csv')
df.reset_index()
#print (df)

#获取sku的list
id = np.array(df['sku_id'])
idlist = id.tolist()

sku = idlist[:65]
#print (sku)

deng = pd.read_csv('ToDeng_sku_predict.csv')
#print (deng.head())

final = []
for i in sku:
    #这里一定要赋一个新的值，不然循环会出问题
    new= deng[deng['sku_id'].isin([i])]
    predict = new.iloc[0,3]


    data = all[all['sku_id'].isin([i])]
    #print (data)
    data = data.drop(['sale_price', 'long', 'years', 'tag_price','designer','barcode'], axis=1)

    #data = new.groupby(['channel_id']).sum()
    data['sale_date'] = pd.to_datetime(all['sale_date'])
    data = data.set_index('sale_date')
    #print (data)

    #output = data.resample('w').sum()
    #按周显示
    output = data.to_period('w')
    #按周和店铺分组
    op = output.groupby(['sale_date','channel_id']).sum()

    op = op.unstack()
    op = op.fillna(0)
    y = op.ix[-2,:]
    yy = pd.DataFrame(y)
    #重新reset为单级索引
    yy = yy.reset_index()

    a = sum(yy.ix[:,2])
    b = yy.ix[:,2]

    #每个店铺的销量占sku销量的比例
    yy['prop'] = b / a
    #print (yy)
    yy = yy.drop(yy.columns[[0,2]],axis=1)
    #print (yy)

    yy['sku_id'] = i
    #sku 的 下周销量
    yy['NextWeek_predict'] = round(predict)
    #销量分配
    yy['distribution_sale'] = yy['NextWeek_predict'] * yy['prop']
    yy['var'] = new.iloc[0,2]


    for index, row in yy.iterrows():
        final.append(row)

ff = pd.DataFrame(final)

ff.to_csv('distribution.csv')
