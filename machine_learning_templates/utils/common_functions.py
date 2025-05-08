"""
机器学习常用函数说明文档
包含常用函数的中文解释和用法示例
"""

# NumPy数组操作函数
"""
np.random.rand(d0, d1, ..., dn)
    功能：生成一个[0,1)之间均匀分布的随机数组
    参数：每个参数表示对应维度的大小
    示例：np.random.rand(100, 3) 生成100行3列的随机矩阵

np.random.randint(low, high, size)
    功能：生成指定范围内的随机整数数组
    参数：最小值，最大值，数组形状
    示例：np.random.randint(0, 10, (3, 4)) 生成3行4列的0-9随机整数矩阵

np.concatenate((a1, a2, ...), axis)
    功能：沿指定轴连接数组
    参数：要连接的数组元组，连接的轴
    示例：np.concatenate((arr1, arr2), axis=0) 垂直连接两个数组

np.reshape(array, newshape)
    功能：改变数组的形状
    参数：数组，新的形状
    示例：arr.reshape(3, 4) 将数组重塑为3行4列

np.transpose(array)
    功能：转置数组
    参数：要转置的数组
    示例：np.transpose(arr) 或 arr.T 进行矩阵转置

np.dot(a, b)
    功能：矩阵乘法
    参数：两个要相乘的数组
    示例：np.dot(A, B) 计算矩阵A和B的乘积
"""

# 数据处理函数
"""
np.unique(array)
    功能：获取数组中的唯一值
    参数：输入数组
    示例：np.unique([1, 1, 2, 3, 3]) 返回[1, 2, 3]

np.where(condition)
    功能：返回满足条件的元素索引
    参数：条件表达式
    示例：np.where(arr > 0) 返回大于0的元素索引

np.argmax(array, axis)
    功能：返回最大值的索引
    参数：数组，计算轴
    示例：np.argmax(arr, axis=1) 返回每行最大值的索引

np.cumsum(array)
    功能：计算累积和
    参数：输入数组
    示例：np.cumsum([1, 2, 3]) 返回[1, 3, 6]
"""

# 统计函数
"""
np.percentile(array, q)
    功能：计算百分位数
    参数：数组，百分位数值
    示例：np.percentile(arr, 75) 计算75%分位数

np.cov(X, Y)
    功能：计算协方差矩阵
    参数：两个变量的观测值
    示例：np.cov(x, y) 计算x和y的协方差矩阵

np.corrcoef(X, Y)
    功能：计算相关系数矩阵
    参数：两个变量的观测值
    示例：np.corrcoef(x, y) 计算x和y的相关系数矩阵
"""

# 机器学习预处理函数
"""
sklearn.preprocessing.MinMaxScaler()
    功能：将特征缩放到指定范围(默认[0,1])
    用法：scaler.fit_transform(X)
    示例：MinMaxScaler().fit_transform(X)

sklearn.preprocessing.LabelEncoder()
    功能：将标签编码为数值
    用法：encoder.fit_transform(y)
    示例：LabelEncoder().fit_transform(['cat', 'dog', 'cat'])

sklearn.preprocessing.OneHotEncoder()
    功能：独热编码
    用法：encoder.fit_transform(X)
    示例：OneHotEncoder().fit_transform([[0], [1], [2]])
"""

# 模型评估函数
"""
sklearn.metrics.confusion_matrix()
    功能：计算混淆矩阵
    参数：真实标签，预测标签
    示例：confusion_matrix(y_true, y_pred)

sklearn.metrics.roc_curve()
    功能：计算ROC曲线
    参数：真实标签，预测概率
    示例：roc_curve(y_true, y_score)

sklearn.metrics.auc()
    功能：计算AUC值
    参数：x坐标，y坐标
    示例：auc(fpr, tpr)
"""

# 深度学习激活函数
"""
torch.nn.ReLU()
    功能：整流线性单元
    特点：max(0,x)
    用途：解决梯度消失问题

torch.nn.Sigmoid()
    功能：sigmoid激活函数
    特点：将输入压缩到(0,1)
    用途：二分类问题

torch.nn.Tanh()
    功能：双曲正切函数
    特点：将输入压缩到(-1,1)
    用途：处理梯度消失
"""

# 深度学习损失函数
"""
torch.nn.CrossEntropyLoss()
    功能：交叉熵损失
    用途：多分类问题
    示例：criterion = nn.CrossEntropyLoss()

torch.nn.MSELoss()
    功能：均方误差损失
    用途：回归问题
    示例：criterion = nn.MSELoss()

torch.nn.BCELoss()
    功能：二元交叉熵损失
    用途：二分类问题
    示例：criterion = nn.BCELoss()
"""

# 深度学习优化器
"""
torch.optim.SGD()
    功能：随机梯度下降
    参数：学习率，动量等
    示例：optim.SGD(model.parameters(), lr=0.01)

torch.optim.RMSprop()
    功能：RMSprop优化器
    特点：自适应学习率
    示例：optim.RMSprop(model.parameters(), lr=0.01)

torch.optim.Adagrad()
    功能：Adagrad优化器
    特点：参数自适应学习率
    示例：optim.Adagrad(model.parameters(), lr=0.01)
"""