"""
模型保存模板
功能：保存和加载机器学习模型

包含功能：
1. PyTorch模型保存和加载
2. Scikit-learn模型保存和加载
3. 模型检查点（checkpoint）保存
4. 完整训练状态保存
"""

# ================ 导入必要的库 ================
import torch          # PyTorch深度学习框架
import joblib        # 用于保存scikit-learn模型
import os            # 操作系统接口，用于文件和目录操作
import json          # JSON数据处理
from datetime import datetime  # 日期时间处理

# ================ 1. PyTorch模型保存 ================
def save_pytorch_model(model, save_dir, model_name=None):
    """
    保存PyTorch模型
    
    参数：
    - model: PyTorch模型实例
    - save_dir: 保存目录路径
    - model_name: 模型名称（可选）
    """
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    
    # 如果没有指定模型名称，使用时间戳生成
    if model_name is None:
        model_name = f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # 构建完整的模型保存路径
    model_path = os.path.join(save_dir, f'{model_name}.pth')
    # 保存模型状态字典
    torch.save(model.state_dict(), model_path)
    print(f'模型已保存到: {model_path}')

def load_pytorch_model(model, model_path):
    """
    加载PyTorch模型
    
    参数：
    - model: 预定义的模型结构实例
    - model_path: 模型权重文件路径
    """
    # 加载模型状态字典
    model.load_state_dict(torch.load(model_path))
    # 设置为评估模式
    model.eval()
    return model

# ================ 2. 保存训练检查点 ================
def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename=None):
    """
    保存训练检查点，包含完整训练状态
    
    参数：
    - model: PyTorch模型实例
    - optimizer: 优化器实例
    - epoch: 当前训练轮次
    - loss: 当前损失值
    - save_dir: 保存目录
    - filename: 文件名（可选）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成检查点文件名
    if filename is None:
        filename = f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
    
    # 构建完整的检查点保存路径
    checkpoint_path = os.path.join(save_dir, filename)
    
    # 创建检查点字典，包含所有需要保存的信息
    checkpoint = {
        'epoch': epoch,                                    # 当前轮次
        'model_state_dict': model.state_dict(),           # 模型状态
        'optimizer_state_dict': optimizer.state_dict(),   # 优化器状态
        'loss': loss                                      # 损失值
    }
    
    # 保存检查点
    torch.save(checkpoint, checkpoint_path)
    print(f'检查点已保存到: {checkpoint_path}')

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载训练检查点
    
    参数：
    - model: PyTorch模型实例
    - optimizer: 优化器实例
    - checkpoint_path: 检查点文件路径
    """
    # 加载检查点文件
    checkpoint = torch.load(checkpoint_path)
    # 恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    # 恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 获取保存的轮次和损失值
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss

# ================ 3. Scikit-learn模型保存 ================
def save_sklearn_model(model, save_dir, model_name=None):
    """
    保存Scikit-learn模型
    
    参数：
    - model: Scikit-learn模型实例
    - save_dir: 保存目录
    - model_name: 模型名称（可选）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成模型文件名
    if model_name is None:
        model_name = f'sklearn_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    # 构建完整的模型保存路径
    model_path = os.path.join(save_dir, f'{model_name}.joblib')
    # 使用joblib保存模型
    joblib.dump(model, model_path)
    print(f'模型已保存到: {model_path}')

def load_sklearn_model(model_path):
    """
    加载Scikit-learn模型
    
    参数：
    - model_path: 模型文件路径
    """
    # 使用joblib加载模型
    return joblib.load(model_path)

# ================ 4. 保存模型配置 ================
def save_model_config(config, save_dir, config_name=None):
    """
    保存模型配置信息
    
    参数：
    - config: 配置字典
    - save_dir: 保存目录
    - config_name: 配置文件名（可选）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成配置文件名
    if config_name is None:
        config_name = f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    # 构建完整的配置文件保存路径
    config_path = os.path.join(save_dir, config_name)
    
    # 将配置信息保存为JSON文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f'配置已保存到: {config_path}')

# ================ 使用示例 ================
if __name__ == '__main__':
    # PyTorch模型保存示例
    # model = YourPyTorchModel()
    # save_pytorch_model(model, 'saved_models')
    
    # 保存检查点示例
    # optimizer = torch.optim.Adam(model.parameters())
    # save_checkpoint(model, optimizer, epoch=10, loss=0.1, save_dir='checkpoints')
    
    # Scikit-learn模型保存示例
    # from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier()
    # save_sklearn_model(model, 'saved_models')
    
    # 保存配置示例
    # config = {
    #     'model_type': 'CNN',
    #     'learning_rate': 0.001,
    #     'batch_size': 32
    # }
    # save_model_config(config, 'configs')