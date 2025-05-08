"""
线性回归算法模板
用于回归问题的基础算法实现

算法流程：
1. 模型假设
   - 假设因变量与自变量呈线性关系
   - 包含可学习的权重和偏置项
   - y = wx + b

2. 参数估计
   - 使用最小二乘法
   - 最小化预测值与真实值的均方误差
   - 可通过正规方程或梯度下降求解

3. 模型训练
   - 计算预测值
   - 计算损失函数
   - 更新模型参数

4. 预测
   - 使用学习到的参数
   - 代入新的特征值
   - 得到预测结果

5. 模型评估
   - 计算均方误差(MSE)
   - 计算决定系数(R²)
   - 分析残差
"""

# 导入必要的库
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 生成示例数据
X = np.random.rand(100, 3)  # 100个样本，每个样本3个特征
y = X @ np.array([1.5, -2.0, 1.0]) + np.random.randn(100) * 0.1  # 线性关系加噪声

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建线性回归模型实例
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算模型性能
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 打印结果
print(f"均方误差 (MSE): {mse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")
print(f"模型系数(权重): {model.coef_}")
print(f"模型截距(偏置): {model.intercept_}")