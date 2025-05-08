"""
Transformer模型模板
用于序列到序列的任务，如机器翻译

算法流程：
1. 输入处理
   - 词嵌入编码
   - 位置编码
   - 输入编码叠加

2. 编码器
   - 多头自注意力
   - 前馈神经网络
   - 残差连接和层归一化
   - 多层堆叠

3. 解码器
   - 掩码多头注意力
   - 编码器-解码器注意力
   - 前馈神经网络
   - 多层堆叠

4. 输出生成
   - 线性变换
   - Softmax分类
   - 生成目标序列

5. 模型训练
   - 教师强制训练
   - 损失计算
   - 参数优化
"""

# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import math

# 定义Transformer参数
d_model = 512    # 模型维度
nhead = 8        # 注意力头数
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
batch_size = 32
seq_length = 100

# 生成示例数据
src = torch.randint(0, 1000, (batch_size, seq_length))  # 源序列
tgt = torch.randint(0, 1000, (batch_size, seq_length))  # 目标序列

# 创建Transformer模型
model = nn.Transformer(
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    output = model(src, tgt)
    
    # 计算损失
    loss = criterion(output.view(-1, d_model), tgt.view(-1))
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

# 模型评估
model.eval()
with torch.no_grad():
    test_src = torch.randint(0, 1000, (1, seq_length))
    test_tgt = torch.randint(0, 1000, (1, seq_length))
    output = model(test_src, test_tgt)
    print(f'测试输出形状: {output.shape}')