"""
线性回归算法模板
功能：使用sklearn实现线性回归

算法原理：
1. 假设数据和标签之间存在线性关系：y = wx + b
2. 通过最小化均方误差来学习参数w和b
3. 可以使用梯度下降或正规方程求解

优点：
1. 模型简单，易于理解和实现
2. 计算效率高
3. 具有很好的可解释性
4. 可以作为其他算法的基准

使用场景：
1. 房价预测
2. 销量预测
3. 温度预测
4. 其他连续值预测任务
"""

# ================ 导入必要的库 ================
import numpy as np                          # 数值计算库
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.metrics import mean_squared_error, r2_score  # 评估指标
from sklearn.preprocessing import StandardScaler  # 标准化处理
import matplotlib.pyplot as plt             # 数据可视化
import pandas as pd                         # 数据处理库

# ================ 1. 数据生成/加载 ================
# 生成模拟数据
np.random.seed(42)  # 设置随机种子，确保结果可复现
X = 2 * np.random.rand(100, 1)  # 生成100个样本，每个样本1个特征
y = 4 + 3 * X + np.random.randn(100, 1)  # 生成目标值：y = 4 + 3x + 噪声

# ================ 2. 数据预处理 ================
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,      # 20%用于测试
    random_state=42     # 随机种子
)

# ================ 3. 模型训练 ================
# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 打印模型参数
print('模型参数：')
print(f'截距项(b)：{model.intercept_[0]:.4f}')
print(f'权重(w)：{model.coef_[0][0]:.4f}')

# ================ 4. 模型预测 ================
# 在测试集上进行预测
y_pred = model.predict(X_test)

# ================ 5. 模型评估 ================
# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f'\n均方误差(MSE)：{mse:.4f}')

# 计算R方值（决定系数）
r2 = r2_score(y_test, y_pred)
print(f'R方值(R²)：{r2:.4f}')

# ================ 6. 结果可视化 ================
# 绘制实际值vs预测值对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs 预测值')
plt.show()

# 绘制残差图
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')
plt.show()