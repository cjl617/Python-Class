import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据加载与特征工程
# ============================================================
print("=" * 40)
print("1. 数据加载与特征工程（TMDB 电影数据）")
print("=" * 40)

df = pd.read_csv("netflix_movies_detailed_up_to_2025.csv")   # 换成你的实际文件名

# 1.1 目标变量：票房 revenue
df = df[df["revenue"] > 0].copy()
print(f"筛选后样本数: {len(df)}")

# 对数变换（重尾分布）
df["log_revenue"] = np.log1p(df["revenue"])
df["log_budget"] = np.log1p(df["budget"].clip(lower=0))
df["has_budget"] = (df["budget"] > 0).astype(int)

# 1.2 分类特征：多值字段取第一个主值
def first_token(s):
    return s.fillna("Unknown").astype(str).str.split(",").str[0].str.strip()

df["main_genre"] = first_token(df["genres"])
df["main_language"] = first_token(df["language"])
df["main_country"] = first_token(df["country"])

# 1.3 数值特征
num_cols = [
    "log_budget", "has_budget", "popularity", "vote_count",
    "vote_average", "release_year", "rating",
]

# 1.4 分类特征 One-Hot
cat_cols = ["main_genre", "main_language", "main_country", "type"]

X = pd.get_dummies(df[num_cols + cat_cols], drop_first=True, dtype=float)
y = df["log_revenue"].values

print(f"特征矩阵维度: {X.shape}")
print(f"目标变量: log1p(revenue)，原始 revenue 范围 "
      f"[{df['revenue'].min():,}, {df['revenue'].max():,}]")

# 1.5 划分 + 标准化
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 2. 模型训练与评估 (CV=10)
# ============================================================
print("\n" + "=" * 40)
print("2. Ridge / Lasso / ElasticNet 训练与评估 (CV=10)")
print("=" * 40)

alphas = np.logspace(-4, 3, 120)

models = {
    "Ridge": RidgeCV(alphas=alphas, cv=10),
    "Lasso": LassoCV(alphas=alphas, cv=10, max_iter=50000),
    "ElasticNet": ElasticNetCV(
        alphas=alphas, l1_ratio=[0.1, 0.5, 0.9, 1.0],
        cv=10, max_iter=50000, n_jobs=-1,
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    n_nonzero = int(np.sum(model.coef_ != 0))

    results[name] = {
        "model": model,
        "best_alpha": model.alpha_,
        "rmse_log": rmse_log,
        "n_nonzero": n_nonzero,
    }

    print(f"[{name}]")
    print(f"  最优 Alpha: {model.alpha_:.5f}")
    print(f"  测试集 RMSE (log 尺度): {rmse_log:.4f}")
    print(f"  非零变量数: {n_nonzero} / {X.shape[1]}")
    if name == "ElasticNet":
        print(f"  最优 L1 Ratio: {model.l1_ratio_}")
    print("-" * 30)

# ============================================================
# 3. 系数路径图
# ============================================================
print("\n" + "=" * 40)
print("3. 系数路径图")
print("=" * 40)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, name in zip(axes, ["Ridge", "Lasso", "ElasticNet"]):
    coefs = []
    for a in alphas:
        if name == "Ridge":
            m = Ridge(alpha=a)
        elif name == "Lasso":
            m = Lasso(alpha=a, max_iter=50000)
        else:
            m = ElasticNet(alpha=a, l1_ratio=0.5, max_iter=50000)
        m.fit(X_train_scaled, y_train)
        coefs.append(m.coef_)
    ax.plot(alphas, coefs, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_title(f"{name} 系数路径图" + (" (l1_ratio=0.5)" if name == "ElasticNet" else ""))
    ax.set_xlabel("Alpha (log scale)")
    ax.set_ylabel("Coefficients")

plt.tight_layout()
plt.show()

# ============================================================
# 4. 1-SE 法则（修正 ElasticNetCV 的 mse_path_ 维度）
# ============================================================
print("\n" + "=" * 40)
print("4. 1-SE 法则分析")
print("=" * 40)

for name, res in results.items():
    model = res["model"]
    mse_path = model.mse_path_

    # ElasticNetCV 的 mse_path_ 是 3 维 (n_alphas, n_folds, n_l1_ratio)
    # 先选出最优 l1_ratio 对应的切片，再沿折数求均值/标准差
    if mse_path.ndim == 3:
        best_l1 = np.argmin(mse_path.mean(axis=(0, 1)))
        mse_path = mse_path[:, :, best_l1]

    mse_mean = mse_path.mean(axis=1)
    mse_std = mse_path.std(axis=1)

    best_idx = np.argmin(mse_mean)
    best_mse = mse_mean[best_idx]
    sem = mse_std[best_idx] / np.sqrt(10)
    threshold = best_mse + sem

    # alphas_ 升序：满足条件的最后一个索引 = 最大的 alpha = 最稀疏
    valid = np.where(mse_mean <= threshold)[0]
    idx_1se = valid[-1] if len(valid) else best_idx
    alpha_1se = model.alphas_[idx_1se]

    if name == "Ridge":
        m1 = Ridge(alpha=alpha_1se)
    elif name == "Lasso":
        m1 = Lasso(alpha=alpha_1se, max_iter=50000)
    else:
        m1 = ElasticNet(alpha=alpha_1se, l1_ratio=model.l1_ratio_, max_iter=50000)

    m1.fit(X_train_scaled, y_train)
    rmse_1se = np.sqrt(mean_squared_error(y_test, m1.predict(X_test_scaled)))
    n_nonzero_1se = int(np.sum(m1.coef_ != 0))

    print(f"[{name}]")
    print(f"  Min-MSE: Alpha={res['best_alpha']:.5f}, "
          f"非零={res['n_nonzero']}, RMSE(log)={res['rmse_log']:.4f}")
    print(f"  1-SE:    Alpha={alpha_1se:.5f}, "
          f"非零={n_nonzero_1se}, RMSE(log)={rmse_1se:.4f}")
    print("  结论:", "1-SE 模型更稀疏且精度损失可忽略。"
          if n_nonzero_1se < res["n_nonzero"]
          else "该数据下 1-SE 未带来额外稀疏收益。")
    print("-" * 30)