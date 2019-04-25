df = pd.read_csv("/Users/rui/Documents/new-jeans/data_full/purchase_20190410104706.csv",encoding='gb18030',low_memory=False)

#对异常值进行处理
df = df[(df.Quantity>0)&(df.purchase_price>0)]
df = df.dropna(axis = 0)
# 去掉不必要的数据列
df = df.drop(['purchase_no','supplier_id','purchase_price'],axis = 1)
#df = df[df['purchase_date'].str.contains('2018')]
data = df.groupby('sku_id').sum()

data.to_csv('sku_group.csv')
