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

   输入：某店牛仔裤 特征“小尺寸的数量、中尺寸的数量、大尺寸的数量、无折扣数量、折扣率、灰色数量、蓝色数量、黑色数量、周内最高价格、周内最低价格、周内价格均值、周内价格标准差、除黑灰蓝颜色之外其他颜色数量、周销量”
   输出：错误率
   算法：LSTM

1.week-data.py 用于生成周销量数据，包含特征：“小尺寸的数量、中尺寸的数量、大尺寸的数量、无折扣数量、折扣率、灰色数量、蓝色数量、
黑色数量、周内最高价格、周内最低价格、周内价格均值、周内价格标准差、周销量”，最左侧index为销售日期。

2.lstm-week.py 用于周销量预测，error 在7%-20%之间，训练数据和测试数据损失曲线重合程度较好。

3.predict.csv 为30次预测结果

*****区域单品周销量预测：

   输入：上海市牛仔裤 特征“小尺寸的数量、中尺寸的数量、大尺寸的数量、无折扣数量、折扣率、灰色数量、蓝色数量、黑色数量、周内最高价格、周内最低价格、周内价格均值、周内价格标准差、除黑灰蓝颜色之外其他颜色数量、周销量”
   输出：错误率
   算法：LSTM


1. 08-12-shanghai-week.csv 为上海地区牛仔裤周销量数据，包含特征：

  “小尺寸的数量、中尺寸的数量、大尺寸的数量、无折扣数量、折扣率、灰色数量、蓝色数量、黑色数量、周内最高价格、周内最低价格、周内价格均值、周内价格标准          差、除黑灰蓝颜色之外其他颜色数量、周销量”
  
2. shanghai-predict.csv 为30次预测结果

3. shanghai-week-data.py 为生成区域周销量数据。

******品牌单品售罄率特征：

1. 品牌单品售罄率的定义是：在一定时间内（周、月）每个单品（有独一无二的sku编号）售卖的数量占该单品总库存量的比例。

2. 售罄率特征目的：反映某单品是 畅销款 正常款 还是滞销款，提升销量预测的准确度。

3. 统计单品售罄率：

   1)stock_process.py

      使用'purchase_20190410104706.csv‘统计每个sku的库存，得到‘sku_group.csv’。
      eg: sku1(sku编码),54（该单品库存数量为54个）

   2)sale_process.py

      使用销量数据统计每个sku在某年（目前统计的为2017年）每周的销量，得到‘sale_sku_group_2017.csv’。
  
   3)sold_out_rate.py
      
      按时间累加某sku的销量，计算该sku占库存的比例，得到'sold_out_rate.csv'

********单品（sku）周销量预测：

    输入：单品（某个特定货号、尺码、颜色的单品）的特征：“折扣数量、折扣率、周销量” 
    输出：单品的周销量预测，预测值和实际值偏差的方差
    算法：LSTM

过程:

1)对sku进行清洗，选择出sku总销量>200的sku单品（并且周销量大于0的销售周数大于24）
2）统计单品的“折扣数量、折扣率、周销量”三个特征，（应有售罄率特征，但售罄率较多>100%,暂不输入），每个sku的周销量为一个独立的csv文件，进入模型。
3）通过模型输出单品的周销量预测，预测值和实际值偏差的方差








