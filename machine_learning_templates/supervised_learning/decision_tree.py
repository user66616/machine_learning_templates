"""
决策树算法实现模板
可用于分类和回归问题

算法流程：
1. 特征选择
   - 计算每个特征的信息增益或基尼指数
   - 选择最优划分特征和阈值
   - 目标是使子节点数据更加纯净

2. 树的生长
   - 根据选定特征将数据集分割
   - 对每个子集递归构建子树
   - 直到达到停止条件

3. 停止条件判断
   - 达到最大深度
   - 节点样本数小于阈值
   - 节点已经纯净

4. 叶节点决策
   - 分类问题：多数投票
   - 回归问题：平均值

5. 预测新样本
   - 从根节点开始遍历
   - 根据特征值选择分支
   - 直到达到叶节点
"""

import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        决策树节点
        
        参数:
            feature: 特征索引
            threshold: 分割阈值
            left: 左子树
            right: 右子树
            value: 叶节点值
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        """
        初始化决策树
        
        参数:
            max_depth: 最大树深度
            min_samples_split: 分裂所需的最小样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def fit(self, X, y):
        """
        训练决策树
        
        参数:
            X: 特征矩阵
            y: 标签
        """
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        """
        递归构建决策树
        
        参数:
            X: 特征矩阵
            y: 标签
            depth: 当前深度
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_labels == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
            
        # 寻找最佳分割
        feature_idxs = np.arange(n_features)
        best_feature, best_thresh = self._best_split(X, y, feature_idxs)
        
        # 创建子树
        left_idxs = X[:, best_feature] < best_thresh
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return Node(best_feature, best_thresh, left, right)
        
    def _most_common_label(self, y):
        """
        返回最常见的标签
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]
        
    def predict(self, X):
        """
        预测新数据
        
        参数:
            X: 特征矩阵
        返回:
            预测标签
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

def main():
    """
    示例用法
    """
    # 生成示例数据
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 2, 100)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    tree = DecisionTree(max_depth=5)
    tree.fit(X_train, y_train)
    
    # 预测和评估
    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"准确率: {accuracy:.4f}")

if __name__ == "__main__":
    main()