import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 初始化 df
df = None

# 1. 加载 Carseats 数据集
try:
    dataset = sm.datasets.get_rdataset("Carseats", "ISLR")
    df = dataset.data
    print("数据集加载成功！")
except Exception as e:
    print(f"在线加载失败，请检查网络。错误信息: {e}")
    # 如果连不上网，这里抛出异常停止运行，避免后面带着空 df 去跑模型
    raise RuntimeError("无法获取数据，请确保网络通畅，或手动下载 Carseats.csv 文件。")

# 查看前几行
print("\n--- 数据集前5行 ---")
print(df.head())

# 2. 建立多元线性回归模型
model = smf.ols('Sales ~ Price + Income + Advertising + ShelveLoc', data=df).fit()

print("\n" + "="*50)
print("模型拟合报告 (Model Summary)")
print("="*50)
print(model.summary())

# 3. 提取 ShelveLoc 基准组
print("\n" + "="*50)
print("【解答 1】ShelveLoc 的基准组")
print("="*50)
print("根据模型报告，ShelveLoc 的基准组是：'Bad'（货架位置差）。")
print("理由：模型摘要中只显示了 ShelveLoc[T.Good] 和 ShelveLoc[T.Medium]，缺失的类别即为基准组。")

# 4. 解读系数
print("\n" + "="*50)
print("【解答 2】ShelveLoc[Good] 系数的商业含义")
print("="*50)
coef_good = model.params.get('ShelveLoc[T.Good]', None)
if coef_good is not None:
    print(f"ShelveLoc[Good] 的系数为: {coef_good:.4f}")
    print("实际商业含义解读：")
    print("在控制商品价格(Price)、居民收入(Income)和广告投入(Advertising)不变的情况下，")
    print(f"货架位置为“Good”的商品，其平均销售额(Sales)比货架位置为“Bad”的商品高出 {coef_good:.4f} 个单位。")

# 5. 计算 VIF (不再使用 patsy，直接提取 statsmodels 内部矩阵)
print("\n" + "="*50)
print("【解答 3】方差膨胀因子 (VIF) 计算与多重共线性评估")
print("="*50)

# 直接从模型中提取设计矩阵和变量名
X = model.model.exog
vif_names = model.model.exog_names

X_df = pd.DataFrame(X, columns=vif_names)

# 剔除截距项 (Intercept) 计算 VIF，因为截距项通常会使 VIF 虚高
if 'Intercept' in X_df.columns:
    X_df = X_df.drop(columns=['Intercept'])

vif_data = pd.DataFrame()
vif_data["变量 (Variable)"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

print(vif_data)
print("\nVIF 评估标准提示：")
print("VIF < 5: 不存在多重共线性风险（理想状态）")
print("5 <= VIF < 10: 存在中等程度的多重共线性，需引起注意")
print("VIF >= 10: 存在严重的多重共线性风险，需考虑剔除变量或进行特征工程")