# Fabulous-Intern-Project

*****月销量预测：

1. pre_dataprocess.py 是对数据的预处理，特征提取。获取到2009-2016年“上海市上海市市辖区”“牛仔裤/牛仔长裤”每个月的数据。
包含特征：小尺寸的数量、中尺寸的数量、大尺寸的数量、有折扣、无折扣、灰色数量、紫色数量、蓝色数量、白色数量、绿色数量、红色数量、黄色数量、
黑色数量、棕色数量、最高价格、最低价格、价格均值、价格标准差、月销量
2. combine.py 是对数据的再次处理。对无固定区间的特征进行标准化、增加日期特征（取每月第一天）
尝试使用ARIMA模型拟合，ARIMA模型只与内变量有关，在这用于观察销量趋势，是否是平稳序列
3. result_origin 为pre_dataprocess.py 运行后得到的09-16年的数据
4. result 为combine.py 为最终数据
5. 下一步增加的维度：
颜色聚类？（目前颜色分的太粗暴，不能在多个数据集上普遍使用，影响精度）该月的节假日数量、该月的季节、最大销量在该月的第几天
6.LSTM用于预测。


*****单店单品周销量预测：

1.week-data.py 用于生成周销量数据，包含特征：“小尺寸的数量、中尺寸的数量、大尺寸的数量、无折扣数量、折扣率、灰色数量、蓝色数量、
黑色数量、周内最高价格、周内最低价格、周内价格均值、周内价格标准差、周销量”，最左侧index为销售日期。

2.lstm-week.py 用于周销量预测，error 在7%-20%之间，训练数据和测试数据损失曲线重合程度较好。

3.predict.csv 为30次预测结果

*****区域单品周销量预测：

1. 08-12-shanghai-week.csv 为上海地区牛仔裤周销量数据，包含特征：

  “小尺寸的数量、中尺寸的数量、大尺寸的数量、无折扣数量、折扣率、灰色数量、蓝色数量、黑色数量、周内最高价格、周内最低价格、周内价格均值、周内价格标准          差、除黑灰蓝颜色之外其他颜色数量、周销量”
  
2. shanghai-predict.csv 为30次预测结果
