# 1. 导入所需库
import pandas as pd  # 数据处理和分析
import numpy as np   # 数值计算
from sklearn.ensemble import RandomForestRegressor  # 随机森林回归模型
from sklearn.metrics import mean_squared_error  # 评估指标MSE

# 2. 数据加载和初步检查
# 读取训练数据（1500条记录）和测试数据（500条记录）
train_data = pd.read_csv('stock_train_data.csv')  # 包含日期、开盘价、最高价、最低价、收盘价、成交量、调整后收盘价
test_data = pd.read_csv('stock_test_data.csv')    # 包含除收盘价外的其他字段

# 3. 数据预处理
# 将日期列转换为datetime格式以便后续处理
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# 按日期排序确保时间序列顺序正确
train_data = train_data.sort_values('date')
test_data = test_data.sort_values('date')

# 4. 特征工程（简化版）
# 定义用于训练的特征列（使用原始提供的特征）
features = ['open', 'high', 'low', 'volume', 'adj_close']

# 5. 准备训练数据
X_train = train_data[features]  # 特征矩阵
y_train = train_data['close']   # 目标值（收盘价）

# 6. 模型训练
# 初始化随机森林回归模型
model = RandomForestRegressor(
    n_estimators=100,  # 使用100棵树
    random_state=42    # 固定随机种子保证结果可复现
)

# 训练模型
model.fit(X_train, y_train)

# 7. 模型验证（可选）
# 在训练集最后20%数据上验证（按时间顺序）
train_size = int(len(X_train) * 0.8)
X_val = X_train[train_size:]
y_val = y_train[train_size:]

# 预测验证集并计算MSE
val_pred = model.predict(X_val)
mse = mean_squared_error(y_val, val_pred)
print(f"验证集MSE: {mse}")

# 8. 准备测试数据
X_test = test_data[features]  # 使用与训练集相同的特征

# 9. 进行预测
predictions = model.predict(X_test)  # 预测测试集收盘价

# 10. 生成提交文件
# 创建包含日期和预测收盘价的DataFrame
submission = pd.DataFrame({
    'id': test_data['date'].dt.strftime('%Y-%m-%d'),  # 格式化日期为YYYY-MM-DD
    'close': predictions  # 预测的收盘价
})

# 11. 保存结果
# 将结果保存为CSV文件，不包含索引列
submission.to_csv('submission.csv', index=False)

# 12. 结果验证
# 检查提交文件是否符合要求
print("\n提交文件验证：")
print(f"记录数量: {len(submission)} (应等于500)")
print(f"日期范围: {submission['id'].min()} 至 {submission['id'].max()}")
print("前5行示例:")
print(submission.head())
