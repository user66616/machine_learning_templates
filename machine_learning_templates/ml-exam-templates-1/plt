# 导入必要的库
import pandas as pd                          # 数据处理
import numpy as np                           # 数值计算
import matplotlib.pyplot as plt              # 基础绘图
import seaborn as sns                        # 美化绘图
from sklearn.metrics import confusion_matrix # 混淆矩阵
from sklearn.preprocessing import StandardScaler # 标准化
from sklearn.decomposition import PCA             # 主成分分析
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# 设置图像风格
sns.set(style="whitegrid")                  # 设置 seaborn 样式
plt.rcParams["figure.figsize"] = (8, 5)     # 设置默认图像大小

# 假设数据集已加载为 df，且有 y_true, y_pred, y_proba 可用
# df = pd.read_csv("your_data.csv")

# -------------------------------
# 1️⃣ 数据概况与缺失值
# -------------------------------

print("📦 数据维度:", df.shape)                  # 输出数据的维度
display(df.head())                               # 显示前几行数据

display(df.describe(include="all"))              # 查看数值/类别型统计信息

missing = df.isnull().sum()                      # 统计每列缺失值数量
missing = missing[missing > 0]                   # 过滤出存在缺失的列
if not missing.empty:
    missing.sort_values().plot.barh(color="salmon")  # 缺失值水平条形图
    plt.title("🧩 缺失值分布")
    plt.xlabel("缺失数量")
    plt.ylabel("特征名")
    plt.show()
else:
    print("✅ 没有缺失值")

# -------------------------------
# 2️⃣ 类别型变量分布
# -------------------------------

cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns  # 获取类别型列

for col in cat_cols:
    plt.figure()
    sns.countplot(data=df, x=col, palette="pastel", order=df[col].value_counts().index)  # 按频数排序
    plt.title(f"📌 类别分布: {col}")
    plt.xlabel(col)
    plt.ylabel("计数")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 3️⃣ 数值型变量分布（直方图 + KDE）
# -------------------------------

num_cols = df.select_dtypes(include=np.number).columns   # 获取数值型列

for col in num_cols:
    plt.figure()
    sns.histplot(df[col].dropna(), kde=True, bins=30, color="skyblue")  # 带核密度的直方图
    plt.title(f"📈 数值分布: {col}")
    plt.xlabel(col)
    plt.ylabel("频数")
    plt.tight_layout()
    plt.show()

# -------------------------------
# 4️⃣ 数值型变量箱型图（检测异常值）
# -------------------------------

for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col], color="lightgreen")  # 箱型图用于查看异常值
    plt.title(f"📉 箱型图: {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# -------------------------------
# 5️⃣ 数值变量之间的相关性热力图
# -------------------------------

corr = df[num_cols].corr()  # 计算相关系数矩阵

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)  # 相关性热力图
plt.title("🔗 数值变量相关性热力图")
plt.tight_layout()
plt.show()

# -------------------------------
# 6️⃣ 分组箱型图：类别 vs 数值型特征
# -------------------------------

for cat in cat_cols:
    for num in num_cols:
        plt.figure()
        sns.boxplot(data=df, x=cat, y=num, palette="Set2")  # 按类别分组画箱型图
        plt.title(f"📎 {num} vs {cat}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

# -------------------------------
# 7️⃣ PairPlot（数值型变量成对关系）
# -------------------------------

pair_cols = num_cols[:5]  # 最多取5列避免图像太大

sns.pairplot(df[pair_cols], diag_kind="kde", corner=True)  # 画 pairplot 图
plt.suptitle("🧊 数值变量两两关系图", y=1.02)
plt.show()

# -------------------------------
# 8️⃣ PCA 降维可视化（适用于高维数值数据）
# -------------------------------

scaled = StandardScaler().fit_transform(df[num_cols].dropna())  # 标准化数值型特征

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)                          # PCA 降维到二维

pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])       # 结果转换为 DataFrame

if "target" in df.columns:
    pca_df["target"] = df["target"]                             # 若有 target 则加上标签

    plt.figure()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="target", palette="Set1")  # 按类别着色
    plt.title("🧩 PCA 可视化（带标签）")
    plt.tight_layout()
    plt.show()
else:
    plt.figure()
    sns.scatterplot(data=pca_df, x="PC1", y="PC2")              # 无标签版本
    plt.title("🧩 PCA 可视化（无标签）")
    plt.tight_layout()
    plt.show()

# -------------------------------
# 9️⃣ 混淆矩阵可视化（分类任务）
# -------------------------------

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)  # y_true 是真实标签，y_pred 是预测结果

# 可视化混淆矩阵
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")  # 蓝色热力图表示混淆矩阵
plt.xlabel("Predicted Label")                       # 横轴是预测标签
plt.ylabel("True Label")                            # 纵轴是真实标签
plt.title("📊 混淆矩阵 Confusion Matrix")
plt.tight_layout()
plt.show()

# -------------------------------
# 🔟 补充：ROC 和 PR 曲线（如果有 y_proba）
# -------------------------------

# ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_proba)  # 计算假正率和真正率
roc_auc = roc_auc_score(y_true, y_proba)  # AUC 分数

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")  # 绘制 ROC 曲线
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("📈 ROC 曲线")
plt.legend()
plt.grid(True)
plt.show()

# PR 曲线
precision, recall, _ = precision_recall_curve(y_true, y_proba)  # 计算 Precision 和 Recall
ap = average_precision_score(y_true, y_proba)                   # 平均精度

plt.figure()
plt.plot(recall, precision, label=f"AP = {ap:.2f}", color="teal")  # 绘制 PR 曲线
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("📉 Precision-Recall 曲线")
plt.legend()
plt.grid(True)
plt.show()
