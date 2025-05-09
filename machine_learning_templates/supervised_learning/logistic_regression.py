"""
逻辑回归算法模板
功能：使用sklearn实现逻辑回归分类

算法原理：
1. 在线性回归基础上应用sigmoid函数：P(y=1|x) = 1 / (1 + e^(-wx-b))
2. 通过最大似然估计学习参数
3. 使用梯度下降优化

优点：
1. 实现简单，计算效率高
2. 可解释性强
3. 不易过拟合
4. 可以输出概率

使用场景：
1. 二分类问题
2. 多分类问题（一对多）
3. 风险预测
4. 疾病诊断
"""

# ================ 导入必要的库 ================
import numpy as np
from sklearn.linear_model import LogisticRegression  # 逻辑回归模型
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.metrics import (
    accuracy_score,  # 准确率
    classification_report,  # 分类报告
    confusion_matrix,  # 混淆矩阵
    roc_curve, auc  # ROC曲线和AUC
)
import matplotlib.pyplot as plt  # 绘图
import seaborn as sns  # 统计可视化
import pandas as pd  # 数据处理

# ================ 1. 数据生成/加载 ================
# 生成模拟二分类数据
np.random.seed(0)
X = np.random.randn(100, 2)  # 100个样本，2个特征
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # 二分类标签

# 也可以加载真实数据
# data = pd.read_csv('your_data.csv')
# X = data[['feature1', 'feature2', ...]]
# y = data['target']

# ================ 2. 数据集划分 ================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 20%用于测试
    random_state=42,   # 随机种子
    stratify=y         # 保持标签比例
)

# ================ 3. 数据预处理 ================
# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================ 4. 模型创建和训练 ================
# 创建逻辑回归模型
model = LogisticRegression(
    C=1.0,               # 正则化强度的倒数
    random_state=42,     # 随机种子
    max_iter=1000        # 最大迭代次数
)

# 训练模型
model.fit(X_train_scaled, y_train)

# 打印模型参数
print('模型参数:')
print(f'截距 (b)：{model.intercept_[0]:.4f}')
print('系数 (w)：')
for i, coef in enumerate(model.coef_[0]):
    print(f'特征 {i+1}: {coef:.4f}')

# ================ 5. 模型预测 ================
# 预测类别
y_pred = model.predict(X_test_scaled)
# 预测概率
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ================ 6. 模型评估 ================
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'\n准确率: {accuracy:.4f}')

# 打印分类报告
print('\n分类报告:')
print(classification_report(y_test, y_pred))

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# ================ 7. 可视化 ================
# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('实际类别')
plt.show()

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('接收者操作特征曲线 (ROC)')
plt.legend(loc="lower right")
plt.show()

# ================ 8. 决策边界可视化 ================
# 仅适用于2D特征
def plot_decision_boundary(X, y, model, scaler):
    h = .02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 预测网格点的类别
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edcolors='black')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('逻辑回归决策边界')
    plt.show()

# 绘制决策边界
plot_decision_boundary(X, y, model, scaler)