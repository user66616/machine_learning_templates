"""
评估指标工具模板
包含常用的模型评估指标
"""

# 导入必要的库
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

def classification_metrics(y_true, y_pred):
    """
    分类模型评估指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
    返回:
        包含各项指标的字典
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

def regression_metrics(y_true, y_pred):
    """
    回归模型评估指标
    
    参数:
        y_true: 真实值
        y_pred: 预测值
    返回:
        包含各项指标的字典
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

def cross_validation_score(model, X, y, cv=5):
    """
    交叉验证评估
    
    参数:
        model: 模型实例
        X: 特征矩阵
        y: 标签
        cv: 折数
    返回:
        交叉验证分数
    """
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=cv)
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }