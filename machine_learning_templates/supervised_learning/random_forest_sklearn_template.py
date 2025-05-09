"""
基于sklearn的随机森林分类器模板
功能：使用sklearn快速实现随机森林分类

算法原理：
1. 随机森林是一种集成学习方法，通过构建多个决策树并进行投票来进行分类
2. 每棵树使用随机抽样的数据和特征进行训练
3. 最终预测结果通过多数投票（分类）或平均（回归）得到

优点：
1. 抗过拟合能力强
2. 可以处理高维数据
3. 可以评估特征重要性
4. 训练速度快，易于并行化

使用场景：
1. 分类和回归问题
2. 特征选择
3. 异常检测
"""

# ================ 导入必要的库 ================
# sklearn.ensemble: 包含集成学习方法
from sklearn.ensemble import RandomForestClassifier
# make_classification: 用于生成模拟分类数据
from sklearn.datasets import make_classification
# 用于数据集划分、交叉验证和网格搜索
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# 用于模型评估的各种指标
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# 用于数据预处理的标准化和归一化
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# 用于处理缺失值
from sklearn.impute import SimpleImputer
# 科学计算和数据处理
import numpy as np
import matplotlib.pyplot as plt  # 绘图库
import seaborn as sns          # 统计数据可视化
import pandas as pd           # 数据分析处理

# ================ 1. 数据生成/加载 ================
# 方式1：使用make_classification生成示例数据
# make_classification参数说明：
# n_samples: 样本数量
# n_features: 特征数量
# n_informative: 信息特征的数量（对分类有用的特征数）
# n_redundant: 冗余特征的数量（由信息特征线性组合而来）
# random_state: 随机种子，确保结果可复现
# shuffle: 是否打乱样本顺序
X, y = make_classification(
    n_samples=1000,    # 生成1000个样本
    n_features=4,      # 每个样本有4个特征
    n_informative=2,   # 其中2个特征是有信息量的
    n_redundant=0,     # 不生成冗余特征
    random_state=0,    # 设置随机种子
    shuffle=False      # 不打乱数据顺序
)

# 方式2：加载真实数据
# X = pd.read_csv('your_features.csv')  # 加载特征数据
# y = pd.read_csv('your_labels.csv')    # 加载标签数据

# ================ 2. 数据集划分 ================
# train_test_split参数说明：
# test_size: 测试集占比
# random_state: 随机种子，确保划分结果可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,     # 20%数据用于测试
    random_state=0     # 设置随机种子
)

# ================ 3. 数据预处理 ================
# 处理缺失值
# SimpleImputer参数说明：
# strategy: 填充策略，可选'mean'(平均值),'median'(中位数),'most_frequent'(众数),'constant'(常数)
imputer = SimpleImputer(strategy='mean')  
X = imputer.fit_transform(X)  # 使用均值填充缺失值

# 特征缩放
# StandardScaler: 标准化，将特征转换为均值为0，方差为1的正态分布
# MinMaxScaler: 归一化，将特征缩放到[0,1]区间
scaler = StandardScaler()  
X = scaler.fit_transform(X)  # 对特征进行标准化

# ================ 4. 模型创建和训练 ================
# RandomForestClassifier参数说明：
# n_estimators: 决策树的数量
# max_depth: 树的最大深度，None表示不限制
# min_samples_split: 分裂内部节点所需的最小样本数
# min_samples_leaf: 叶节点所需的最小样本数
# random_state: 随机种子
rf_classifier = RandomForestClassifier(
    n_estimators=100,    # 使用100棵决策树
    max_depth=2,         # 每棵树的最大深度为2
    min_samples_split=2, # 分裂节点最少需要2个样本
    min_samples_leaf=1,  # 叶节点最少需要1个样本
    random_state=0       # 设置随机种子
)

# 训练模型
rf_classifier.fit(X_train, y_train)  # 使用训练集训练模型

# ================ 5. 交叉验证 ================
# cross_val_score: 进行k折交叉验证
# cv: 折数，这里使用5折交叉验证
cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
print("\n交叉验证得分:", cv_scores)  # 打印每折的得分
print("平均交叉验证得分: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

# ================ 6. 模型预测和评估 ================
# 预测单个样本
single_prediction = rf_classifier.predict([[0, 0, 0, 0]])  # 预测一个新样本
print(f'单个样本预测结果：{single_prediction}')

# 预测测试集
y_pred = rf_classifier.predict(X_test)  # 对测试集进行预测

# ================ 7. 特征重要性分析 ================
# feature_importances_: 返回每个特征的重要性得分
feature_importance = rf_classifier.feature_importances_
for i, importance in enumerate(feature_importance):
    print(f'特征 {i+1} 重要性: {importance:.4f}')

# 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.title('特征重要性')
plt.xlabel('特征索引')
plt.ylabel('重要性')
plt.show()

# ROC曲线绘制
# predict_proba: 返回预测为各个类别的概率
y_prob = rf_classifier.predict_proba(X_test)[:, 1]  # 获取预测为正类的概率
fpr, tpr, _ = roc_curve(y_test, y_prob)  # 计算假正率和真正率
roc_auc = auc(fpr, tpr)  # 计算AUC值

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('接收者操作特征曲线 (ROC)')
plt.legend(loc="lower right")
plt.show()

# ================ 8. 学习曲线分析 ================
# learning_curve: 用于分析模型的学习曲线
# train_sizes: 训练集大小的不同取值
# train_scores: 在训练集上的得分
# test_scores: 在验证集上的得分
train_sizes, train_scores, test_scores = learning_curve(
    rf_classifier, X, y, 
    cv=5,  # 5折交叉验证
    n_jobs=-1,  # 使用所有CPU核心
    train_sizes=np.linspace(0.1, 1.0, 10)  # 训练集大小从10%到100%
)

# 计算平均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='训练得分')
plt.plot(train_sizes, test_mean, label='交叉验证得分')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel('训练样本数')
plt.ylabel('得分')
plt.title('学习曲线')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# ================ 9. 特征选择 ================
# SelectFromModel: 基于模型的特征选择
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(rf_classifier, prefit=True)  # 使用随机森林进行特征选择
feature_idx = selector.get_support()  # 获取被选择的特征索引
selected_features = np.where(feature_idx)[0]  # 转换为数组索引
print("\n被选择的特征索引:", selected_features)

# ================ 10. 模型保存和加载 ================
import joblib

# 保存模型到文件
joblib.dump(rf_classifier, 'random_forest_model.joblib')

# 从文件加载模型
# loaded_model = joblib.load('random_forest_model.joblib')