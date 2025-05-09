"""
K近邻算法模板
功能：使用sklearn实现KNN分类器

算法原理：
1. 计算待预测样本与所有训练样本的距离
2. 选择K个最近的邻居
3. 对这K个邻居进行投票（分类）或平均（回归）

优点：
1. 算法简单直观
2. 无需训练过程
3. 对异常值不敏感
4. 适用于多分类问题

使用场景：
1. 分类问题
2. 回归问题
3. 推荐系统
4. 模式识别
"""

# ================ 导入必要的库 ================
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # KNN分类器
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.metrics import accuracy_score  # 准确率计算
from sklearn.datasets import load_iris  # 加载示例数据集
import matplotlib.pyplot as plt  # 数据可视化

# ================ 1. 数据加载 ================
# 加载iris数据集作为示例
iris = load_iris()
X = iris.data      # 特征矩阵
y = iris.target    # 目标变量

# ================ 2. 数据集划分 ================
# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,     # 30%用于测试
    random_state=42,   # 随机种子，确保结果可复现
    stratify=y         # 保持标签比例
)

# ================ 3. 模型创建和训练 ================
# 创建KNN分类器实例
knn = KNeighborsClassifier(
    n_neighbors=5,     # 设置K值为5
    weights='uniform', # 统一权重
    algorithm='auto',  # 自动选择最优算法
    metric='minkowski' # 距离度量方式
)

# 训练模型
knn.fit(X_train, y_train)

# ================ 4. 模型预测 ================
# 在测试集上进行预测
y_pred = knn.predict(X_test)

# ================ 5. 模型评估 ================
# 计算模型准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy:.2f}')

# ================ 6. K值影响分析 ================
# 测试不同K值对模型性能的影响
k_range = range(1, 31)
k_scores = []

# 遍历不同的K值
for k in k_range:
    # 创建并训练模型
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    # 预测并记录准确率
    y_pred = knn.predict(X_test)
    k_scores.append(accuracy_score(y_test, y_pred))

# ================ 7. 可视化分析 ================
# 绘制K值与准确率的关系图
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores)
plt.xlabel('K值')
plt.ylabel('准确率')
plt.title('不同K值对模型准确率的影响')
plt.grid(True)
plt.show()