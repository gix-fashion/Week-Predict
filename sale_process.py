df = pd.read_csv("/Users/rui/Documents/new-jeans/sale/ALL-sale.csv",encoding='UTF-8',low_memory=False)

df = df1[(df1.Quantity>0)&(df1.sale_price>0)]
df = df1.dropna(axis = 0)
df = df1.drop(['sale_no','channel_id','sale_price'],axis = 1)
df = df1[df1['sale_date'].str.contains('2017')]

df['sale_date'] = pd.to_datetime(df1['sale_date']).apply(lambda x:x.strftime("%W"))
df.rename(columns={'sale_date':'sale_week'},inplace=True)
#使用sku 和 周 进行分组
data = df1.groupby(['sku_id','sale_week']).sum()
data.sort_values(by=['sale_week'],inplace=True)

#行列转置
data = data.unstack(1)
data.to_csv('sale_sku_group_2017.csv')
