"""
卷积神经网络(CNN)模板
用于图像分类等任务

算法流程：
1. 网络构建
   - 卷积层提取特征
   - 池化层降维压缩
   - 全连接层分类决策

2. 前向传播
   - 输入数据通过卷积层
   - 特征图经过激活函数
   - 池化层降低维度
   - 全连接层输出预测

3. 反向传播
   - 计算损失函数
   - 计算梯度
   - 更新网络参数

4. 模型训练
   - 批量处理数据
   - 迭代优化参数
   - 验证模型性能

5. 预测推理
   - 输入新的图像
   - 通过训练好的网络
   - 输出分类结果
"""

# 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 生成示例数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集作为示例
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义CNN模型
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1600, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'准确率: {100 * correct / total:.2f}%')





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np

"""
CNN模板说明：
1. 包含完整的CNN实现流程
2. 每行代码都有详细注释
3. 使用模拟数据演示
4. 包含训练和评估过程
"""

# 1. 数据准备部分 --------------------------------------------------------
# 创建模拟数据 (32x32的RGB图像，1000个样本，10个类别)
num_samples = 1000
num_classes = 10

# 模拟输入数据 (batch, channels, height, width)
X = torch.randn(num_samples, 3, 32, 32)  # 3通道(RGB)，32x32像素
# 模拟标签数据
y = torch.randint(0, num_classes, (num_samples,)) 

# 划分训练集和测试集 (80%训练，20%测试)
train_size = int(0.8 * num_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建PyTorch数据集和数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 2. CNN模型定义 --------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # 卷积层1: 输入3通道，输出16通道，3x3卷积核
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 最大池化层: 2x2窗口，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 卷积层2: 输入16通道，输出32通道
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # 全连接层1
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 经过两次池化后图像尺寸为8x8
        # 全连接层2 (输出层)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # 第一组卷积+激活+池化
        x = self.conv1(x)  # 输出尺寸: (batch, 16, 32, 32)
        x = self.relu(x)
        x = self.pool(x)    # 输出尺寸: (batch, 16, 16, 16)
        
        # 第二组卷积+激活+池化
        x = self.conv2(x)   # 输出尺寸: (batch, 32, 16, 16)
        x = self.relu(x)
        x = self.pool(x)    # 输出尺寸: (batch, 32, 8, 8)
        
        # 展平特征图 (保留batch维度)
        x = x.view(x.size(0), -1)  # 输出尺寸: (batch, 32*8*8)
        
        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

# 3. 模型初始化 --------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes).to(device)

# 损失函数 (交叉熵损失，适用于多分类)
criterion = nn.CrossEntropyLoss()

# 优化器 (Adam优化器，学习率0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练过程 --------------------------------------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    
    for images, labels in train_loader:
        # 将数据移动到设备 (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播计算梯度
        optimizer.step()       # 更新参数
        
        running_loss += loss.item()
    
    # 打印每个epoch的统计信息
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# 5. 评估过程 --------------------------------------------------------
model.eval()  # 设置为评估模式
correct = 0
total = 0

with torch.no_grad():  # 不计算梯度，节省内存
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"测试集准确率: {accuracy:.2f}%")

# 6. 关键概念说明 ----------------------------------------------------
"""
CNN核心组件解释：
1. 卷积层 (Conv2d): 
   - 使用小滤波器在图像上滑动，提取局部特征
   - 参数: 输入通道数, 输出通道数, 卷积核大小, 步长, 填充

2. 池化层 (MaxPool2d):
   - 降采样，减少计算量，增加平移不变性
   - 常用最大池化，取窗口内最大值

3. 激活函数 (ReLU):
   - 引入非线性，增强模型表达能力
   - ReLU: f(x) = max(0, x)

4. 全连接层 (Linear):
   - 将学到的特征映射到样本标记空间
   - 通常在CNN最后几层使用

5. 输入输出尺寸计算:
   - 输出尺寸 = (输入尺寸 - 卷积核大小 + 2*填充) / 步长 + 1
   - 池化通常会使尺寸减半
"""