import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 数据准备
print("=" * 40)
print("1. 数据准备与预处理")
print("=" * 40)
# 使用糖尿病数据集作为替代（若要使用Boston，请用 fetch_openml(name="boston", version=1)）
data = load_diabetes()
X, y = data.data, data.target
feature_names = data.feature_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 正则化前必须 Z-score 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. 建模与评估
print("\n" + "=" * 40)
print("2. 模型训练与评估 (CV=10)")
print("=" * 40)

# 定义 Alpha 搜索空间
alphas = np.logspace(-3, 2, 100)

# 初始化模型
models = {
    "Ridge": RidgeCV(alphas=alphas, cv=10),
    "Lasso": LassoCV(alphas=alphas, cv=10, max_iter=10000),
    "ElasticNet": ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=10, max_iter=10000)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    n_nonzero = np.sum(model.coef_ != 0)

    results[name] = {
        "model": model,
        "best_alpha": model.alpha_,
        "rmse": rmse,
        "n_nonzero": n_nonzero,
        "coef": model.coef_
    }

    print(f"[{name}]")
    print(f"  最优 Alpha: {model.alpha_:.4f}")
    print(f"  测试集 RMSE: {rmse:.4f}")
    print(f"  非零变量数: {n_nonzero} / {X.shape[1]}")
    if name == "ElasticNet":
        print(f"  最优 L1 Ratio: {model.l1_ratio_}")
    print("-" * 30)

# 3. 绘制系数路径图
print("\n" + "=" * 40)
print("3. 绘制系数路径图")
print("=" * 40)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Ridge 路径
coefs_ridge = []
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    coefs_ridge.append(ridge.coef_)
axes[0].plot(alphas, coefs_ridge)
axes[0].set_xscale('log')
axes[0].set_title('Ridge 系数路径图')
axes[0].set_xlabel('Alpha (log scale)')
axes[0].set_ylabel('Coefficients')

# Lasso 路径
coefs_lasso = []
for a in alphas:
    lasso = Lasso(alpha=a, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    coefs_lasso.append(lasso.coef_)
axes[1].plot(alphas, coefs_lasso)
axes[1].set_xscale('log')
axes[1].set_title('Lasso 系数路径图')
axes[1].set_xlabel('Alpha (log scale)')

# ElasticNet 路径 (固定 l1_ratio=0.5)
coefs_enet = []
for a in alphas:
    enet = ElasticNet(alpha=a, l1_ratio=0.5, max_iter=10000)
    enet.fit(X_train_scaled, y_train)
    coefs_enet.append(enet.coef_)
axes[2].plot(alphas, coefs_enet)
axes[2].set_xscale('log')
axes[2].set_title('ElasticNet 系数路径图 (l1_ratio=0.5)')
axes[2].set_xlabel('Alpha (log scale)')

plt.tight_layout()
plt.show()

# 4. 1-SE 法则讨论
print("\n" + "=" * 40)
print("4. 1-SE 法则分析")
print("=" * 40)

for name, res in results.items():
    model = res["model"]
    # 获取交叉验证的均方误差 (MSE)
    mse_mean = model.mse_path_.mean(axis=1)
    mse_std = model.mse_path_.std(axis=1)

    # 最优 alpha 对应的索引
    best_idx = np.argmin(mse_mean)
    best_alpha = model.alphas_[best_idx] if hasattr(model, 'alphas_') else model.alpha_
    best_mse = mse_mean[best_idx]

    # 计算 1-SE 阈值
    # 标准误 = 标准差 / sqrt(折数)
    sem = mse_std[best_idx] / np.sqrt(10)
    threshold = best_mse + sem

    # 寻找满足 1-SE 条件的最大 alpha (即最稀疏的模型)
    # 在 alphas 列表中找到最小的索引，使得 mse_mean <= threshold
    # 因为 alphas 是递增的，所以索引越小，alpha越小，模型越复杂
    idx_1se = np.where(mse_mean <= threshold)[0][0]
    alpha_1se = model.alphas_[idx_1se] if hasattr(model, 'alphas_') else model.alpha_

    # 用 1-SE 的 alpha 重新拟合模型并评估
    if name == "Ridge":
        model_1se = Ridge(alpha=alpha_1se)
    elif name == "Lasso":
        model_1se = Lasso(alpha=alpha_1se, max_iter=10000)
    else:
        model_1se = ElasticNet(alpha=alpha_1se, l1_ratio=model.l1_ratio_, max_iter=10000)

    model_1se.fit(X_train_scaled, y_train)
    y_pred_1se = model_1se.predict(X_test_scaled)
    rmse_1se = np.sqrt(mean_squared_error(y_test, y_pred_1se))
    n_nonzero_1se = np.sum(model_1se.coef_ != 0)

    print(f"[{name}] 1-SE 法则:")
    print(f"  最优模型 (Min MSE): Alpha={best_alpha:.4f}, MSE={best_mse:.4f}, 非零变量={res['n_nonzero']}")
    print(f"  1-SE 模型 (更稀疏): Alpha={alpha_1se:.4f}, MSE={mse_mean[idx_1se]:.4f}, 非零变量={n_nonzero_1se}")
    print(f"  结论: 是否选用更稀疏模型？", end="")
    if n_nonzero_1se < res['n_nonzero']:
        print("是，1-SE模型在牺牲极少精度的情况下减少了特征数量。")
    else:
        print("否，该数据集下1-SE模型并未带来明显的稀疏性收益。")
    print("-" * 30)