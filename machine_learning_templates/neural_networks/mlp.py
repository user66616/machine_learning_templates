"""
多层感知机(MLP)模板
基础神经网络实现

算法流程：
1. 网络构建
   - 定义输入层
   - 配置隐藏层（数量和神经元）
   - 设置输出层

2. 前向传播
   - 线性变换
   - 激活函数
   - 层间连接

3. 反向传播
   - 计算损失
   - 求解梯度
   - 参数更新

4. 模型训练
   - 批量训练
   - 多轮迭代
   - 学习率调整

5. 预测
   - 数据预处理
   - 网络前向计算
   - 输出预测结果
"""

# 导入必要的库
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成示例数据
X = np.random.randn(100, 10)  # 100个样本，10个特征
y = (X.sum(axis=1) > 0).astype(int)  # 二分类标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建MLP模型实例
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 两个隐藏层，分别有64和32个神经元
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 打印结果
print(f"准确率: {accuracy:.4f}")
print(f"模型结构: {model.n_layers_} 层")
print(f"每层神经元数量: {[layer.shape[0] for layer in model.coefs_]}")