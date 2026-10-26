import os
import time
import warnings
import argparse

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold,
    LeaveOneOut, cross_val_score,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification, fetch_openml, load_digits
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score,
)

# imbalanced-learn 可选
try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("[警告] 未安装 imbalanced-learn，案例⑤将跳过 SMOTE/欠采样对比")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

RESULTS = {}   # 汇总所有案例的关键指标


# =========================================================
# 通用工具
# =========================================================
def make_onehot():
    """兼容不同 sklearn 版本的 OneHotEncoder"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def evaluate(name, y_true, y_pred, y_score=None, pos_label=1):
    """统一评估：打印分类报告 + 返回关键指标字典"""
    print(f"\n{'=' * 62}")
    print(f"[模型] {name}")
    print(f"{'=' * 62}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    print(f"混淆矩阵:\n{confusion_matrix(y_true, y_pred)}")

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_pos": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "precision_pos": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    }
    if y_score is not None:
        out["auc_roc"] = roc_auc_score(y_true, y_score)
        out["auc_pr"] = average_precision_score(y_true, y_score)
        print(f"AUC-ROC: {out['auc_roc']:.4f}  |  AUC-PR: {out['auc_pr']:.4f}")
    print(f"Acc: {out['accuracy']:.4f}  |  "
          f"Recall+: {out['recall_pos']:.4f}  |  "
          f"Precision+: {out['precision_pos']:.4f}  |  "
          f"F1+: {out['f1_pos']:.4f}")
    return out


def try_read_csv(url, **kwargs):
    """读取 URL，失败返回 None"""
    try:
        return pd.read_csv(url, **kwargs)
    except Exception as e:
        print(f"[下载失败] {url}\n  原因：{e}")
        return None


# =========================================================
# 案例 ① Spambase
# =========================================================
def load_spambase():
    """UCI Spambase：57 特征 + 1 标签，无表头"""
    candidates = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        "https://archive.ics.uci.edu/static/public/94/spambase.zip",
    ]
    for url in candidates:
        print(f"[Spambase] 尝试下载：{url}")
        df = try_read_csv(url, header=None)
        if df is not None and df.shape[1] == 58:
            print(f"[Spambase] 加载成功，形状：{df.shape}")
            return df.iloc[:, :57].values, df.iloc[:, 57].astype(int).values
    print("[Spambase] 在线下载失败，请手动下载 spambase.data 放到 ./data/ 下")
    return None, None


def run_spambase():
    print("\n" + "#" * 62)
    print("# 案例 ① Spambase 垃圾邮件分类")
    print("#" * 62)

    X, y = load_spambase()
    if X is None:
        return None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ---------- 线性核 ----------
    t0 = time.time()
    lin_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)),
    ])
    lin_pipe.fit(X_tr, y_tr)
    t_lin = time.time() - t0
    res_lin = evaluate("Linear SVM (C=1)", y_te, lin_pipe.predict(X_te))
    res_lin["train_time_s"] = round(t_lin, 2)

    # ---------- RBF 核 + 网格搜索 ----------
    rbf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "svm__C": [1, 10, 100],
        "svm__gamma": [0.001, 0.01, 0.1],
    }
    gs = GridSearchCV(
        rbf_pipe, param_grid, cv=3,
        scoring="accuracy", n_jobs=-1, verbose=0,
    )
    t0 = time.time()
    gs.fit(X_tr, y_tr)
    t_rbf = time.time() - t0
    print(f"\nRBF 最优参数：{gs.best_params_}（网格搜索耗时 {t_rbf:.1f}s）")
    res_rbf = evaluate(f"RBF SVM {gs.best_params_}", y_te, gs.predict(X_te))
    res_rbf["train_time_s"] = round(t_rbf, 2)

    RESULTS["Spambase"] = {"linear": res_lin, "rbf": res_rbf}
    print("\n[结论] 课件参考：线性 92.3%，RBF 约 93.5%。"
          "文本词频特征线性可分性较好，RBF 提升有限但训练时间明显增加。")
    return RESULTS["Spambase"]


# =========================================================
# 案例 ② German Credit
# =========================================================
GERMAN_COLS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property", "age", "other_plans",
    "housing", "existing_credits", "job", "num_liable", "telephone",
    "foreign_worker", "class",
]
GERMAN_CAT_COLS = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "personal_status", "other_debtors", "property",
    "other_plans", "housing", "job", "telephone", "foreign_worker",
]
GERMAN_NUM_COLS = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_liable",
]


def load_german():
    """UCI Statlog German Credit：1000 条，20 特征，标签 1=good 2=bad"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    print(f"[German] 尝试下载：{url}")
    df = try_read_csv(url, sep=r"\s+", header=None)
    if df is None or df.shape[1] != 21:
        print("[German] 在线下载失败，请手动下载 german.data 放到 ./data/ 下")
        return None, None

    df.columns = GERMAN_COLS
    # 标签：1=good → 0；2=bad → 1（正类=坏客户）
    y = (df["class"] == 2).astype(int).values
    X = df.drop(columns=["class"])
    print(f"[German] 加载成功，形状：{X.shape}，坏客户占比：{y.mean() * 100:.1f}%")
    return X, y


def run_german():
    print("\n" + "#" * 62)
    print("# 案例 ② German Credit 信用违约预测")
    print("#" * 62)

    X, y = load_german()
    if X is None:
        return None

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # 预处理：类别列独热，数值列标准化
    preprocess = ColumnTransformer([
        ("cat", make_onehot(), GERMAN_CAT_COLS),
        ("num", StandardScaler(), GERMAN_NUM_COLS),
    ])

    # ---------- 基线 RBF SVM ----------
    base_pipe = Pipeline([
        ("prep", preprocess),
        ("svm", SVC(kernel="rbf", C=5, gamma=0.1,
                    random_state=RANDOM_STATE)),
    ])
    t0 = time.time()
    base_pipe.fit(X_tr, y_tr)
    t_base = time.time() - t0
    res_base = evaluate(
        "RBF SVM 基线 (C=5, gamma=0.1)",
        y_te, base_pipe.predict(X_te), base_pipe.decision_function(X_te),
    )
    res_base["train_time_s"] = round(t_base, 2)

    # ---------- 加权 RBF SVM ----------
    weighted_pipe = Pipeline([
        ("prep", preprocess),
        ("svm", SVC(kernel="rbf", C=5, gamma=0.1,
                    class_weight={0: 1, 1: 3},   # 少数类（坏客户）加权
                    random_state=RANDOM_STATE)),
    ])
    t0 = time.time()
    weighted_pipe.fit(X_tr, y_tr)
    t_w = time.time() - t0
    res_w = evaluate(
        "RBF SVM + class_weight={0:1, 1:3}",
        y_te, weighted_pipe.predict(X_te), weighted_pipe.decision_function(X_te),
    )
    res_w["train_time_s"] = round(t_w, 2)

    RESULTS["German Credit"] = {"baseline": res_base, "weighted": res_w}
    print("\n[结论] 课件参考 AUC≈0.83。小样本 + 混合特征，RBF 优于线性；"
          "class_weight 提升坏客户 Recall，但会牺牲部分 Precision。")
    return RESULTS["German Credit"]


# =========================================================
# 案例 ③ Leukemia 基因表达
# =========================================================
def load_leukemia():
    """优先本地 CSV；否则用结构等价的高维小样本合成数据"""
    for fname in ["leukemia_gene_expression.csv", "leukemia.csv", "data/leukemia.csv"]:
        if os.path.exists(fname):
            print(f"[Leukemia] 读取本地文件：{fname}")
            df = pd.read_csv(fname)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            # 若是字符串标签（ALL/AML），转 0/1
            if y.dtype == object:
                uniq = sorted(set(y))
                y = np.array([uniq.index(v) for v in y])
            print(f"[Leukemia] 形状：{X.shape}，类别分布：{np.bincount(y.astype(int))}")
            return X, y.astype(int)

    print("[Leukemia] 未找到本地基因表达数据，使用结构等价的合成数据（n≪d）")
    X, y = make_classification(
        n_samples=72, n_features=2000, n_informative=50,
        n_redundant=0, n_clusters_per_class=1,
        flip_y=0.02, class_sep=0.8, random_state=RANDOM_STATE,
    )
    print(f"[Leukemia] 合成数据形状：{X.shape}，类别分布：{np.bincount(y)}")
    return X, y


def run_leukemia():
    print("\n" + "#" * 62)
    print("# 案例 ③ Leukemia 基因表达分类（高维小样本）")
    print("#" * 62)

    X, y = load_leukemia()

    # 线性 SVM + 留一交叉验证
    lin_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)),
    ])
    t0 = time.time()
    loo = LeaveOneOut()
    scores = cross_val_score(lin_svm, X, y, cv=loo, scoring="accuracy")
    t_loo = time.time() - t0
    print(f"\n线性 SVM 留一交叉验证：{scores.mean():.4f} ± {scores.std():.4f}"
          f"（{len(scores)} 折，耗时 {t_loo:.1f}s）")

    # 顺带对比 RBF（同 LOOCV）
    rbf_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)),
    ])
    scores_rbf = cross_val_score(rbf_svm, X, y, cv=loo, scoring="accuracy")
    print(f"RBF  SVM 留一交叉验证：{scores_rbf.mean():.4f} ± {scores_rbf.std():.4f}")

    RESULTS["Leukemia"] = {
        "linear_loocv": round(scores.mean(), 4),
        "rbf_loocv": round(scores_rbf.mean(), 4),
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
    }
    print("\n[结论] 课件参考：线性 SVM 留一交叉验证 100%，KNN 85%，决策树 78%。"
          "n≪d 时线性核更稳、更可解释；RBF 在超高维下容易过拟合。")
    return RESULTS["Leukemia"]


# =========================================================
# 案例 ④ MNIST 0 vs 1
# =========================================================
def load_mnist_01():
    """优先 fetch_openml 下载 MNIST；失败则用 sklearn 8x8 digits 降级"""
    print("[MNIST] 尝试通过 fetch_openml 加载 mnist_784（首次约 55MB）...")
    try:
        try:
            mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        except TypeError:
            mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        X = mnist.data.astype(np.float32)
        y = mnist.target.astype(int)
        mask = np.isin(y, [0, 1])
        X, y = X[mask], y[mask]
        X = X / 255.0
        print(f"[MNIST] 0 vs 1 加载成功：{X.shape}，类别分布：{np.bincount(y)}")
        return X, y, "mnist_784"
    except Exception as e:
        print(f"[MNIST] 下载/加载失败：{e}")
        print("[MNIST] 降级到 sklearn 内置 8×8 digits 数据集做 0 vs 1 演示")
        digits = load_digits()
        mask = np.isin(digits.target, [0, 1])
        X = digits.data[mask] / 16.0
        y = digits.target[mask]
        print(f"[Digits] 0 vs 1：{X.shape}，类别分布：{np.bincount(y)}")
        return X, y, "digits_8x8"


def run_mnist():
    print("\n" + "#" * 62)
    print("# 案例 ④ MNIST 0 vs 1 手写数字二分类")
    print("#" * 62)

    X, y, source = load_mnist_01()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ---------- 线性 SVM：PCA 降维 + LinearSVC ----------
    lin_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=100, random_state=RANDOM_STATE)),
        ("svm", LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)),
    ])
    t0 = time.time()
    lin_pipe.fit(X_tr, y_tr)
    t_lin = time.time() - t0
    res_lin = evaluate("LinearSVC + PCA(100)", y_te, lin_pipe.predict(X_te))
    res_lin["train_time_s"] = round(t_lin, 2)

    # ---------- RBF SVM：用 30% 训练子集控制时间 ----------
    X_tr_sub, _, y_tr_sub, _ = train_test_split(
        X_tr, y_tr, train_size=0.3, stratify=y_tr, random_state=RANDOM_STATE
    )
    rbf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma=0.01, random_state=RANDOM_STATE)),
    ])
    t0 = time.time()
    rbf_pipe.fit(X_tr_sub, y_tr_sub)
    t_rbf = time.time() - t0
    res_rbf = evaluate(
        f"RBF SVM (30% 训练子集, n={len(y_tr_sub)})",
        y_te, rbf_pipe.predict(X_te),
    )
    res_rbf["train_time_s"] = round(t_rbf, 2)

    RESULTS["MNIST 0vs1"] = {
        "source": source,
        "linear": res_lin,
        "rbf": res_rbf,
    }
    print(f"\n[结论] 课件参考：线性 98.1%（3s），RBF 99.4%（45s）。"
          f"当前数据源 = {source}。像素特征本身接近线性可分，"
          f"RBF 用训练时间换最后约 1% 准确率。")
    return RESULTS["MNIST 0vs1"]


# =========================================================
# 案例 ⑤ Credit Card Fraud
# =========================================================
def load_fraud():
    """优先本地 creditcard.csv；否则用合成不平衡数据"""
    path = os.path.join(DATA_DIR, "creditcard.csv")
    if os.path.exists(path):
        print(f"[Fraud] 读取本地文件：{path}")
        df = pd.read_csv(path)
        X = df.drop(columns=["Class"]).values
        y = df["Class"].values
        print(f"[Fraud] 形状：{X.shape}，欺诈占比：{y.mean() * 100:.4f}%")
        return X, y, "real"

    print("[Fraud] 未找到 ./data/creditcard.csv，使用合成极端不平衡数据演示")
    X, y = make_classification(
        n_samples=20000, n_features=30, n_informative=10, n_redundant=5,
        weights=[0.99, 0.01], flip_y=0.001, random_state=RANDOM_STATE,
    )
    print(f"[Fraud] 合成数据形状：{X.shape}，欺诈占比：{y.mean() * 100:.4f}%")
    return X, y, "synthetic"


def run_fraud():
    print("\n" + "#" * 62)
    print("# 案例 ⑤ Credit Card Fraud 信用卡欺诈检测")
    print("#" * 62)

    X, y, source = load_fraud()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # ---------- 方案 A：欠采样 + class_weight ----------
    if HAS_IMBLEARN:
        # 把多数类降到 5000，少数类全保留
        rus = RandomUnderSampler(
            sampling_strategy={0: 5000, 1: int(y_tr.sum())},
            random_state=RANDOM_STATE,
        )
        X_rus, y_rus = rus.fit_resample(X_tr, y_tr)
        print(f"[Fraud] 欠采样后训练集：{X_rus.shape}，"
              f"欺诈占比：{y_rus.mean() * 100:.2f}%")
    else:
        X_rus, y_rus = X_tr, y_tr

    svm_weighted = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    class_weight={0: 1, 1: 20},
                    random_state=RANDOM_STATE)),
    ])
    t0 = time.time()
    svm_weighted.fit(X_rus, y_rus)
    t_w = time.time() - t0
    res_w = evaluate(
        "RBF SVM + class_weight (欠采样)",
        y_te, svm_weighted.predict(X_te),
        svm_weighted.decision_function(X_te),
    )
    res_w["train_time_s"] = round(t_w, 2)

    # ---------- 方案 B：SMOTE + RBF SVM ----------
    res_smote = None
    if HAS_IMBLEARN:
        smote = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)
        X_sm, y_sm = smote.fit_resample(X_rus, y_rus)
        print(f"[Fraud] SMOTE 后训练集：{X_sm.shape}，"
              f"欺诈占比：{y_sm.mean() * 100:.2f}%")

        svm_smote = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                        random_state=RANDOM_STATE)),
        ])
        t0 = time.time()
        svm_smote.fit(X_sm, y_sm)
        t_s = time.time() - t0
        res_smote = evaluate(
            "SMOTE + RBF SVM",
            y_te, svm_smote.predict(X_te),
            svm_smote.decision_function(X_te),
        )
        res_smote["train_time_s"] = round(t_s, 2)

    RESULTS["Credit Fraud"] = {
        "source": source,
        "weighted": res_w,
        "smote": res_smote,
    }
    print("\n[结论] 课件参考：标准 RBF SVM Recall 0.42 → class_weight=100 后 0.81"
          " → SMOTE 后 0.85。极端不平衡下 accuracy 无意义，"
          "必须看 Recall / F1 / AUC-PR。")
    return RESULTS["Credit Fraud"]


# =========================================================
# 汇总
# =========================================================
def print_summary():
    print("\n\n" + "=" * 62)
    print("全部案例关键指标汇总")
    print("=" * 62)
    for case, res in RESULTS.items():
        print(f"\n▶ {case}")
        for model_name, m in res.items():
            if not isinstance(m, dict):
                continue
            if "accuracy" not in m:
                continue
            line = (f"  - {model_name:36s} "
                    f"Acc={m['accuracy']:.4f}  "
                    f"Recall+={m['recall_pos']:.4f}  "
                    f"F1+={m['f1_pos']:.4f}")
            if "auc_pr" in m:
                line += f"  AUC-PR={m['auc_pr']:.4f}"
            if "train_time_s" in m:
                line += f"  ({m['train_time_s']}s)"
            print(line)
        # Leukemia 特殊结构
        if "linear_loocv" in res:
            print(f"  - 线性 LOOCV={res['linear_loocv']:.4f}  "
                  f"RBF LOOCV={res['rbf_loocv']:.4f}  "
                  f"(n={res['n_samples']}, d={res['n_features']})")


# =========================================================
# 主入口
# =========================================================
CASE_MAP = {
    "spam": run_spambase,
    "german": run_german,
    "leukemia": run_leukemia,
    "mnist": run_mnist,
    "fraud": run_fraud,
}


def main():
    parser = argparse.ArgumentParser(description="SVM 真实数据分类实战")
    parser.add_argument(
        "--cases", nargs="+",
        default=["spam", "german", "leukemia", "mnist", "fraud"],
        choices=list(CASE_MAP.keys()) + ["all"],
        help="要运行的案例，默认全部",
    )
    args = parser.parse_args()

    cases = list(CASE_MAP.keys()) if "all" in args.cases else args.cases

    t_all = time.time()
    for c in cases:
        try:
            CASE_MAP[c]()
        except Exception as e:
            print(f"\n[案例 {c} 运行出错] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print_summary()
    print(f"\n总耗时：{time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()