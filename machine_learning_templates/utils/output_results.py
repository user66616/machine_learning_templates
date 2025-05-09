"""
预测结果输出模板
功能：将模型预测结果保存为CSV文件

包含功能：
1. 预测结果保存
2. 概率值输出
3. 结果格式化
"""

# ================ 导入必要的库 ================
import pandas as pd  # 数据处理
import numpy as np   # 数值计算
from datetime import datetime  # 时间处理

def save_predictions(test_ids, predictions, output_path=None, probabilities=None, labels=None):
    """
    保存预测结果到CSV文件
    
    参数：
    - test_ids: 测试集ID或索引
    - predictions: 预测标签
    - output_path: 输出文件路径（可选）
    - probabilities: 预测概率（可选）
    - labels: 标签名称（可选）
    """
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 如果未指定输出路径，使用默认路径
    if output_path is None:
        output_path = f'predictions_{timestamp}.csv'
    
    # 创建结果字典
    results = {
        'ID': test_ids,
        'Prediction': predictions
    }
    
    # 如果有概率值，添加到结果中
    if probabilities is not None:
        if probabilities.shape[1] > 1:  # 多分类情况
            for i in range(probabilities.shape[1]):
                col_name = f'Probability_class_{i}' if labels is None else f'Probability_{labels[i]}'
                results[col_name] = probabilities[:, i]
        else:  # 二分类情况
            results['Probability'] = probabilities
    
    # 创建DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存到CSV
    results_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f'预测结果已保存到: {output_path}')
    
    return results_df

# ================ 使用示例 ================
if __name__ == '__main__':
    # 生成示例数据
    test_ids = np.arange(100)  # 测试集ID
    predictions = np.random.randint(0, 3, 100)  # 预测标签
    probabilities = np.random.rand(100, 3)  # 预测概率
    class_labels = ['类别A', '类别B', '类别C']  # 类别标签
    
    # 保存预测结果
    results = save_predictions(
        test_ids=test_ids,
        predictions=predictions,
        probabilities=probabilities,
        labels=class_labels,
        output_path='model_predictions.csv'
    )
    
    # 显示结果预览
    print('\n预测结果预览：')
    print(results.head())