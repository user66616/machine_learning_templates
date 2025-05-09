"""
决策树算法模板
功能：使用sklearn实现决策树分类器

算法原理：
1. 通过递归二分的方式构建树结构
2. 每个节点选择最优特征和阈值进行分割
3. 使用信息增益或基尼指数作为分割准则

优点：
1. 可解释性强
2. 可以处理非线性关系
3. 不需要特征缩放
4. 可以处理数值和类别特征

使用场景：
1. 分类问题
2. 回归问题
3. 特征重要性分析
4. 决策支持系统
"""

# ================ 导入必要的库 ================
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree  # 决策树模型和可视化
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.metrics import (
    accuracy_score,         # 准确率
    classification_report,  # 分类报告
    confusion_matrix       # 混淆矩阵
)
import matplotlib.pyplot as plt  # 数据可视化
import seaborn as sns          # 统计可视化
from sklearn.datasets import make_classification  # 生成分类数据

# ================ 1. 数据生成/加载 ================
# 生成模拟分类数据
X, y = make_classification(
    n_samples=1000,      # 样本数量
    n_features=20,       # 特征数量
    n_informative=15,    # 有信息量的特征数
    n_redundant=5,       # 冗余特征数
    random_state=42      # 随机种子
)

# ================ 2. 数据集划分 ================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20%用于测试
    random_state=42,     # 随机种子
    stratify=y           # 保持标签比例
)

# ================ 3. 模型创建和训练 ================
# 创建决策树分类器
model = DecisionTreeClassifier(
    criterion='gini',     # 分割准则：'gini'或'entropy'
    max_depth=5,         # 树的最大深度
    min_samples_split=2, # 分裂节点所需的最小样本数
    min_samples_leaf=1,  # 叶节点所需的最小样本数
    random_state=42      # 随机种子
)

# 训练模型
model.fit(X_train, y_train)

# ================ 4. 模型预测 ================
# 在测试集上进行预测
y_pred = model.predict(X_test)

# ================ 5. 模型评估 ================
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'准确率：{accuracy:.4f}')

# 打印分类报告
print('\n分类报告：')
print(classification_report(y_test, y_pred))

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# ================ 6. 特征重要性分析 ================
# 获取特征重要性
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    '特征': [f'特征{i}' for i in range(X.shape[1])],
    '重要性': importances
})
feature_importance = feature_importance.sort_values('重要性', ascending=False)

# 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['特征'], feature_importance['重要性'])
plt.xticks(rotation=45)
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('特征重要性分析')
plt.tight_layout()
plt.show()

# ================ 7. 决策树可视化 ================
plt.figure(figsize=(20,10))
plot_tree(model, 
         feature_names=[f'特征{i}' for i in range(X.shape[1])],
         class_names=['类别0', '类别1'],
         filled=True,    # 填充颜色
         rounded=True)   # 圆角
plt.title('决策树可视化')
plt.show()

# ================ 8. 混淆矩阵可视化 ================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('实际类别')
plt.show()

# ================ 9. 模型参数调优 ================
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 创建网格搜索对象
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,              # 5折交叉验证
    scoring='accuracy' # 使用准确率作为评估指标
)

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 打印最佳参数
print('\n最佳参数：')
print(grid_search.best_params_)
print(f'最佳得分：{grid_search.best_score_:.4f}')