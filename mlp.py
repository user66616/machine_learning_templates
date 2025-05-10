# -*- coding: utf-8 -*-
"""pytorch_multiclass.ipynb"""

# 导入必要的库
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1) 加载数据
df = pd.read_csv('/content/train.csv')

# 2) 删除不需要的列
df = df.drop(columns=['id','CustomerId','Surname'])

# 3) 假设'Exited'是一个多类标签，取值从0到K-1
y = df['Exited'].values
X = df.drop(columns='Exited')

# 4) 对分类特征进行编码
le_geo = LabelEncoder().fit(X['Geography'])  # 对地理位置进行编码
le_gen = LabelEncoder().fit(X['Gender'])     # 对性别进行编码
X['Geography'] = le_geo.transform(X['Geography'])
X['Gender']    = le_gen.transform(X['Gender'])

# 5) 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42, stratify=y
)

# 6) 标准化数据
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# 7) 构建PyTorch数据集
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备（GPU/CPU）

# 将numpy数组转换为PyTorch张量
X_train_t = torch.from_numpy(X_train).float().to(device)
y_train_t = torch.from_numpy(y_train).long().to(device)       # 注意使用.long()
X_test_t  = torch.from_numpy(X_test).float().to(device)
y_test_t  = torch.from_numpy(y_test).long().to(device)

# 创建数据加载器
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# 8) 确定类别数量
num_classes = len(np.unique(y_train))

# 9) 定义多层感知器(MLP)模型
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.3),    # 第一层：输入层到64个神经元
            nn.Linear(64, 32),    nn.ReLU(), nn.Dropout(0.3),     # 第二层：64到32个神经元
            nn.Linear(32, out_dim)                                 # 输出层：32到输出类别数
        )
    def forward(self, x):
        return self.net(x)

# 初始化模型、损失函数和优化器
model = MLP(X_train.shape[1], num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=1e-3)  # 使用SGD优化器

# 10) 训练循环
epochs = 200
for epoch in range(1, epochs+1):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()           # 清零梯度
        logits = model(xb)             # 前向传播
        loss   = criterion(logits, yb)  # 计算损失
        loss.backward()                 # 反向传播
        optimizer.step()                # 更新参数
        running_loss += loss.item() * xb.size(0)
    avg_loss = running_loss / len(train_ds)
    print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}")

# 11) 在测试集上评估模型
model.eval()  # 设置为评估模式
with torch.no_grad():
    logits = model(X_test_t)
    preds = logits.argmax(dim=1).cpu().numpy()
    y_true = y_test

print("\nClassification Report on Test Set:")
print(classification_report(y_true, preds, digits=4))

# 12) 对未见过的test.csv进行预测
test_df = pd.read_csv('/content/test.csv')
ids = test_df['id']
test_df = test_df.drop(columns=['CustomerId','Surname'])
test_df['Geography'] = le_geo.transform(test_df['Geography'])  # 使用之前的编码器转换地理位置
test_df['Gender']    = le_gen.transform(test_df['Gender'])    # 使用之前的编码器转换性别
X_sub = scaler.transform(test_df.drop(columns='id').values)   # 使用之前的标准化器
X_sub_t = torch.from_numpy(X_sub).float().to(device)

# 进行预测
model.eval()
with torch.no_grad():
    logits = model(X_sub_t)
    preds  = logits.argmax(dim=1).cpu().numpy()

# 保存预测结果
submission = pd.DataFrame({'id': ids, 'Exited': preds})
submission.to_csv('prediction.csv', index=False)
print("Saved prediction.csv")
