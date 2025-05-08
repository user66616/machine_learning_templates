"""
主成分分析(PCA)算法模板
用于降维和特征提取

算法流程：
1. 数据预处理
   - 对原始数据进行标准化处理
   - 消除量纲影响，使各特征具有相同尺度

2. 计算协方差矩阵
   - 标准化后的数据计算协方差矩阵
   - 反映特征之间的相关性

3. 特征值分解
   - 计算协方差矩阵的特征值和特征向量
   - 特征值代表主成分的方差贡献
   - 特征向量代表主成分的方向

4. 选择主成分
   - 根据特征值大小排序
   - 选择前k个最大特征值对应的特征向量
   - k的选择基于累计方差贡献率

5. 数据转换
   - 原始数据投影到选定的特征向量上
   - 得到降维后的数据表示
"""

# 导入必要的库
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 生成示例数据
X = np.random.rand(100, 10)  # 100个样本，10个特征

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建PCA模型实例
n_components = 3  # 降维后的维度
model = PCA(n_components=n_components)

# 训练模型并转换数据
X_transformed = model.fit_transform(X_scaled)

# 计算解释方差比
explained_variance_ratio = model.explained_variance_ratio_

# 打印结果
print(f'降维后数据形状: {X_transformed.shape}')
print(f'各主成分解释方差比:')
for i, ratio in enumerate(explained_variance_ratio):
    print(f'- PC{i+1}: {ratio:.4f}')
print(f'累计解释方差比: {sum(explained_variance_ratio):.4f}')