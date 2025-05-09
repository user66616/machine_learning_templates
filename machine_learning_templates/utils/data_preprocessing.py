"""
数据预处理模板
功能：常用数据预处理方法的实现示例

包含功能：
1. 数据清洗
2. 特征缩放
3. 特征编码
4. 缺失值处理
5. 异常值处理
"""

# ================ 导入必要的库 ================
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,     # 标准化
    MinMaxScaler,      # 归一化
    LabelEncoder,      # 标签编码
    OneHotEncoder      # 独热编码
)
from sklearn.impute import SimpleImputer  # 缺失值处理

# ================ 1. 数据加载示例 ================
# 假设我们有一个示例数据集
data = pd.DataFrame({
    '年龄': [25, 30, np.nan, 40, 35],
    '收入': [50000, 60000, 75000, np.nan, 65000],
    '性别': ['男', '女', '男', '女', '男'],
    '学历': ['本科', '硕士', '本科', '博士', '硕士']
})

# ================ 2. 缺失值处理 ================
# 数值型特征缺失值填充
num_imputer = SimpleImputer(strategy='mean')  # 使用均值填充
num_cols = ['年龄', '收入']
data[num_cols] = num_imputer.fit_transform(data[num_cols])

# 类别型特征缺失值填充
cat_imputer = SimpleImputer(strategy='most_frequent')  # 使用众数填充
cat_cols = ['性别', '学历']
data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

# ================ 3. 特征缩放 ================
# 标准化
scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# 归一化示例
minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data[num_cols])

# ================ 4. 特征编码 ================
# 标签编码
label_encoder = LabelEncoder()
data['性别_编码'] = label_encoder.fit_transform(data['性别'])

# 独热编码
onehot_encoder = OneHotEncoder(sparse=False)
学历_encoded = onehot_encoder.fit_transform(data[['学历']])
学历_cols = [f'学历_{cat}' for cat in onehot_encoder.categories_[0]]
学历_df = pd.DataFrame(学历_encoded, columns=学历_cols)

# ================ 5. 异常值处理 ================
def detect_outliers(data, column, threshold=3):
    """
    使用Z-score方法检测异常值
    """
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    return z_scores > threshold

# 检测并处理异常值示例
for col in num_cols:
    outliers = detect_outliers(data, col)
    print(f'\n{col}的异常值索引：')
    print(data[outliers].index.tolist())

# ================ 6. 数据分布可视化 ================
import matplotlib.pyplot as plt
import seaborn as sns

# 数值特征分布图
plt.figure(figsize=(12, 5))
for i, col in enumerate(num_cols, 1):
    plt.subplot(1, 2, i)
    sns.histplot(data[col], kde=True)
    plt.title(f'{col}分布图')
plt.tight_layout()
plt.show()

# 类别特征分布图
plt.figure(figsize=(12, 5))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(1, 2, i)
    data[col].value_counts().plot(kind='bar')
    plt.title(f'{col}分布图')
plt.tight_layout()
plt.show()

# ================ 7. 相关性分析 ================
# 计算数值特征间的相关性
correlation = data[num_cols].corr()

# 绘制相关性热力图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('特征相关性热力图')
plt.show()

# 打印处理后的数据信息
print('\n处理后的数据信息：')
print(data.info())
print('\n数据预览：')
print(data.head())