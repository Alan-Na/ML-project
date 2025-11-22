# ensemble_softmax_mlp_v3.py
# -*- coding: utf-8 -*-

import re, math, random
import numpy as np
import pandas as pd

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- Config ----------------
CSV_PATH     = "training_data_conflict_removed.csv"
TARGET_COL   = "label"
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15

# Softmax 方案 3：Softmax 和 MLP 完全共享“结构化特征工程”
SOFTMAX_C_GRID       = [1.0, 0.5, 0.2, 0.1, 0.05]
SOFTMAX_REFIT_TRAIN  = True  # 选好 C 后是否在 train 上重训一遍

# 用于共享预处理器内文本 TF-IDF 的参数
TFIDF_MIN_DF   = 2
TFIDF_NGRAM    = (1, 2)
TFIDF_MAX_FEAT = None      # 这里共享 SVD 后维度本身受控，可不强制上限
TFIDF_MAX_DF   = 1.0       # 在 shared preprocessor 中通常不用太严格的上限

# SVD + MLP 侧
SVD_N_COMP   = 100
BATCH_SIZE   = 64
MAX_EPOCHS   = 50
PATIENCE     = 6
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_P    = 0.20
GRAD_CLIP    = 1.0

N_RUNS       = 200
SEED_BASE    = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

w_lr  = 0.4
w_mlp = 0.6

# === 关键列名 ===
LIKERT_ACADEMIC = "How likely are you to use this model for academic tasks?"
LIKERT_SUBOPT_FREQ = "Based on your experience, how often has this model given you a response that felt suboptimal?"
LIKERT_EXPECT_REF = "How often do you expect this model to provide responses with references or supporting evidence?"
LIKERT_VERIFY_FREQ = "How often do you verify this model's responses?"

MULTI_BEST = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
MULTI_SUBOPT = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"

ORDINAL_MAP_ACADEMIC = {
    "1 — Not at all likely": 1,
    "2 — Unlikely": 2,
    "3 — Neutral / Unsure": 3,
    "4 — Likely": 4,
    "5 — Very likely": 5,
}
ORDINAL_MAP_FREQ = {
    "1 — Never": 1,
    "2 — Rarely": 2,
    "3 — Sometimes": 3,
    "4 — Often": 4,
    "5 — Very often": 5,
}
ORDINAL_COL_MAPS = {
    LIKERT_ACADEMIC: ORDINAL_MAP_ACADEMIC,
    LIKERT_SUBOPT_FREQ: ORDINAL_MAP_FREQ,
    LIKERT_EXPECT_REF: ORDINAL_MAP_FREQ,
    LIKERT_VERIFY_FREQ: ORDINAL_MAP_FREQ,
}

# 多选题关键字（用于 multi-hot）
TASK_KEYWORDS = {
    "math": "Math computations",
    "coding": "Writing or debugging code",
    "data": "Data processing or analysis",
    "draft": "Drafting professional text",
    "writing": "Writing or editing essays",
    "explain": "Explaining complex concepts simply",
    "convert": "Converting content between formats",
}

# ---------------- Utils ----------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_df(path):
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"缺少目标列 {TARGET_COL}")
    if "student_id" not in df.columns:
        df["student_id"] = np.arange(len(df))
        print("[WARN] 未找到 student_id，退化为按行分组。")
    return df

def detect_id_like_columns(df, max_unique_ratio=0.9):
    id_cols = []
    pat = re.compile(r"(?:^|[_\-])(?:id|uuid|guid)(?:$|[_\-])", re.I)
    for c in df.columns:
        if pat.search(str(c)):
            id_cols.append(c)
    return id_cols

def detect_text_like_columns(df, candidate_cols):
    text_cols = []
    for c in candidate_cols:
        s = df[c].astype(str)
        if s.str.len().mean() >= 30 or (s.nunique(dropna=True)/max(1, len(s)) >= 0.2):
            text_cols.append(c)
    return text_cols

def join_text_columns(X):
    if isinstance(X, pd.DataFrame):
        return X.astype(str).fillna("").agg(" ".join, axis=1)
    X = pd.DataFrame(X)
    return X.astype(str).fillna("").agg(" ".join, axis=1)

def build_preprocessor(num_cols, cat_cols, text_cols, svd_seed):
    """
    共享预处理器：
    - 数值：median 填充
    - 类别：常量填充 + OHE
    - 文本：join -> TfidfVectorizer -> TruncatedSVD
    - 整体：MaxAbsScaler
    """
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", ohe)
    ])
    text_pipe = Pipeline([
        ("join", FunctionTransformer(join_text_columns, validate=False)),
        ("tfidf", TfidfVectorizer(
            min_df=TFIDF_MIN_DF,
            ngram_range=TFIDF_NGRAM,
            max_df=TFIDF_MAX_DF,
            max_features=TFIDF_MAX_FEAT
        )),
        ("svd", TruncatedSVD(n_components=SVD_N_COMP, random_state=svd_seed))
    ])
    preprocess = ColumnTransformer(
        [("num", num_pipe, num_cols),
         ("cat", cat_pipe, cat_cols),
         ("text", text_pipe, text_cols)],
        remainder="drop",
        sparse_threshold=0.0    # 强制输出 dense
    )
    return Pipeline([("preprocess", preprocess), ("scale", MaxAbsScaler())])

# Likert & 多选（这里保留 encode_ordinal_likert 但 v3 不强制使用）
def encode_ordinal_likert(df):
    for col, mapping in ORDINAL_COL_MAPS.items():
        if col not in df.columns:
            continue
        df[col] = df[col].map(lambda x: mapping.get(str(x).strip(), np.nan))
    return df

def expand_multi_select(df, col, prefix):
    """
    多选题拆分成 multi-hot：
    - prefix_math, prefix_coding, ...
    - 并删除原长文本列
    """
    if col not in df.columns:
        return df
    s = df[col].fillna("")
    for key, pat in TASK_KEYWORDS.items():
        new_col = f"{prefix}_{key}"
        df[new_col] = s.str.contains(pat, regex=False).astype("float32")
    df = df.drop(columns=[col])
    return df

# ---------------- Dataset & MLP ----------------
class NPDataset(Dataset):
    def __init__(self, X, y):
        self.X = np.asarray(X, np.float32)
        self.y = np.asarray(y, np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, p=DROPOUT_P):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, 128),    nn.ReLU(), nn.Dropout(p),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def eval_model_with_probs(model, loader):
    model.eval()
    y_true, y_pred = [], []
    probs_all = []
    for xb, yb in loader:
        xb = torch.tensor(xb, dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(yb, dtype=torch.long, device=DEVICE)
        logits = model(xb)
        probs  = torch.softmax(logits, dim=1)
        y_true.extend(yb.cpu().numpy().tolist())
        y_pred.extend(torch.argmax(probs, 1).cpu().numpy().tolist())
        probs_all.append(probs.cpu().numpy())
    probs_all = np.vstack(probs_all) if probs_all else np.zeros((0, 0), dtype=np.float32)
    return np.array(y_true), np.array(y_pred), probs_all

# ---------------- 分组划分 ----------------
def grouped_split_with_all_classes(y, groups, test_size, seed, max_tries=50):
    all_classes = np.unique(y)
    tries = 0
    rng = np.random.RandomState(seed)
    last = None
    while True:
        rs = int(rng.randint(0, 2**31 - 1))
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr_idx, te_idx = next(gss.split(np.zeros(len(y)), y, groups))
        last = (tr_idx, te_idx)
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        if (len(np.unique(y_tr)) == len(all_classes)) and (len(np.unique(y_te)) == len(all_classes)):
            return tr_idx, te_idx
        tries += 1
        if tries >= max_tries:
            print("[WARN] 尝试多次仍未覆盖全部类别，使用最后一次切分。")
            return last

# ---------------- Softmax（LR）相关工具 ----------------
def _align_proba(P_raw, clf_classes, classes):
    """把 LR 输出概率矩阵对齐到全局 classes 顺序"""
    cls_lr = list(map(str, clf_classes))
    col_map = {c: i for i, c in enumerate(cls_lr)}
    P = np.zeros((P_raw.shape[0], len(classes)), dtype=np.float32)
    for j, c in enumerate(classes):
        if c in col_map:
            P[:, j] = P_raw[:, col_map[c]]
        else:
            P[:, j] = 0.0
    row_sum = P.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return P / row_sum

def _train_lr_with_val(X_tr, y_tr, X_val, y_val, seed, c_grid):
    """
    在共享特征上训练 LogisticRegression：
    - 仅在 train 上 fit；
    - 用 val_set 选择最佳 C；
    - 可选在 train 上用最佳 C 重训一次。
    """
    best, best_c, best_acc = None, None, -1.0
    for C in c_grid:
        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=C,
            max_iter=3000,
            random_state=seed
        )
        clf.fit(X_tr, y_tr)
        pred_val = clf.predict(X_val)
        acc_val = accuracy_score(y_val, pred_val)
        if acc_val > best_acc + 1e-12:
            best_acc = acc_val
            best_c = C
            best = clf
   
    if SOFTMAX_REFIT_TRAIN and best_c is not None:
        # 合并训练集和验证集
        X_train_full = np.vstack([X_tr, X_val])
        y_train_full = np.concatenate([y_tr, y_val])
        
        best = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=best_c, # 使用选出来的最佳 C
            max_iter=3000,
            random_state=seed
        ).fit(X_train_full, y_train_full)
    return best, best_c, best_acc

# ---------------- 共享特征工程：Softmax & MLP 共用 ----------------
def _compute_shared_features(df, train_idx, val_idx, test_idx, seed):
    """
    在 train 子集上拟合 shared preprocessor，并对 train/val/test 统一 transform。
    - 丢掉 ID-like 列（含 student_id）
    - 多选题展开为 multi-hot（并删除原问句列）
    - Likert 保持字符串形式（交给 OHE）
    - 文本列自动检测并走 TF-IDF→SVD
    """
    dfm = df.copy()

    # 丢掉 ID-like 列（含 student_id）
    id_cols = detect_id_like_columns(dfm.drop(columns=[TARGET_COL], errors="ignore"))
    if "student_id" in dfm.columns:
        id_cols = list(set(id_cols) | {"student_id"})
    if id_cols:
        dfm = dfm.drop(columns=id_cols, errors="ignore")

    # 多选题展开为 multi-hot，并删除原列
    dfm = expand_multi_select(dfm, MULTI_BEST,  prefix="best")
    dfm = expand_multi_select(dfm, MULTI_SUBOPT, prefix="subopt")

    X = dfm.drop(columns=[TARGET_COL])
    y = dfm[TARGET_COL].astype(str)

    # 仅用 train 子集来决定列类型
    X_tr  = X.iloc[train_idx]; y_tr  = y.iloc[train_idx]
    X_val = X.iloc[val_idx];   y_val = y.iloc[val_idx]
    X_te  = X.iloc[test_idx];  y_te  = y.iloc[test_idx]

    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    obj_cols_tr = [c for c in X_tr.columns if c not in num_cols]
    text_cols = detect_text_like_columns(X_tr, obj_cols_tr)
    cat_cols  = [c for c in obj_cols_tr if c not in text_cols]

    pre = build_preprocessor(num_cols, cat_cols, text_cols, svd_seed=seed)
    pre.fit(X_tr, y_tr)

    X_tr_np  = pre.transform(X_tr)
    X_val_np = pre.transform(X_val)
    X_te_np  = pre.transform(X_te)

    meta = {
        "pre": pre,
        "X_tr_np":  X_tr_np,
        "X_val_np": X_val_np,
        "X_te_np":  X_te_np,
        "y_tr":  y_tr.values,
        "y_val": y_val.values,
        "y_te":  y_te.values,
    }
    return meta

def softmax_from_shared_features(meta, classes, seed):
    """
    在共享特征上训练 Softmax（LR）：
    - 仅用 train 特征 + y_tr；
    - 用 val 特征 + y_val 选 C；
    - 返回对 train/test 的预测概率和 acc。
    """
    X_tr_np  = meta["X_tr_np"]
    X_val_np = meta["X_val_np"]
    X_te_np  = meta["X_te_np"]
    y_tr     = meta["y_tr"]
    y_val    = meta["y_val"]
    y_te     = meta["y_te"]

    clf, best_c, _ = _train_lr_with_val(X_tr_np, y_tr, X_val_np, y_val, seed, SOFTMAX_C_GRID)

    P_tr = _align_proba(clf.predict_proba(X_tr_np), clf.classes_, classes)
    P_te = _align_proba(clf.predict_proba(X_te_np), clf.classes_, classes)

    pred_tr = np.array([classes[i] for i in np.argmax(P_tr, axis=1)])
    pred_te = np.array([classes[i] for i in np.argmax(P_te, axis=1)])
    acc_tr  = accuracy_score(y_tr, pred_tr)
    acc_te  = accuracy_score(y_te, pred_te)
    return P_tr, P_te, acc_tr, acc_te

def mlp_from_shared_features(meta, classes, seed):
    """
    在共享特征上训练 MLP：
    - train 上训练，val 上 early stopping；
    - 最后在 train/test 上评估。
    """
    seed_all(seed)

    X_tr_np  = meta["X_tr_np"]
    X_val_np = meta["X_val_np"]
    X_te_np  = meta["X_te_np"]
    y_tr     = meta["y_tr"]
    y_val    = meta["y_val"]
    y_te     = meta["y_te"]

    cls2idx = {c: i for i, c in enumerate(classes)}
    y_tr_np  = np.array([cls2idx[s] for s in y_tr],  np.int64)
    y_val_np = np.array([cls2idx[s] for s in y_val], np.int64)
    y_te_np  = np.array([cls2idx[s] for s in y_te],  np.int64)

    model = MLP(in_dim=X_tr_np.shape[1], n_classes=len(classes), p=DROPOUT_P).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(NPDataset(X_tr_np,  y_tr_np), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(NPDataset(X_val_np, y_val_np), batch_size=BATCH_SIZE, shuffle=False)

    best_val = math.inf
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = torch.tensor(xb, dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(yb, dtype=torch.long, device=DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # 验证集 early stopping
        with torch.no_grad():
            model.eval()
            val_losses = []
            for xb, yb in val_loader:
                xb = torch.tensor(xb, dtype=torch.float32, device=DEVICE)
                yb = torch.tensor(yb, dtype=torch.long, device=DEVICE)
                val_losses.append(criterion(model(xb), yb).item() * len(yb))
            val_loss = float(np.sum(val_losses)) / max(1, len(y_val_np))
            scheduler.step(val_loss) #新增此行以及更换Adam为AdamW

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 在 train + test 上评估（仍然只用 train 训练）
    train_full = DataLoader(NPDataset(X_tr_np, y_tr_np), batch_size=BATCH_SIZE, shuffle=False)
    y_tr_true, y_tr_pred, P_tr = eval_model_with_probs(model, train_full)

    test_loader = DataLoader(NPDataset(X_te_np, y_te_np), batch_size=BATCH_SIZE, shuffle=False)
    y_te_true, y_te_pred, P_te = eval_model_with_probs(model, test_loader)

    acc_tr = accuracy_score(y_tr_true, y_tr_pred)
    acc_te = accuracy_score(y_te_true, y_te_pred)
    return P_tr, P_te, acc_tr, acc_te

# ---------------- 主循环 ----------------
def run_once(seed):
    df = load_df(CSV_PATH)
    y_all = df[TARGET_COL].astype(str)
    groups = df["student_id"].astype(str)

    # 先切 train/test
    tr_idx, te_idx = grouped_split_with_all_classes(y_all, groups, TEST_SIZE, seed)

    # 再在 train 部分切出 val
    y_tr_all = y_all.iloc[tr_idx]
    groups_tr = groups.iloc[tr_idx]
    idx_tr_in, idx_val_in = grouped_split_with_all_classes(y_tr_all, groups_tr, VAL_SIZE, seed + 1)
    tr_idx_final = np.array(tr_idx)[idx_tr_in]
    val_idx_final = np.array(tr_idx)[idx_val_in]

    # 仅用训练集确定类别集合
    classes = sorted(y_all.iloc[tr_idx_final].unique())

    # 构建并拟合共享预处理器，生成 train/val/test 的统一特征
    meta = _compute_shared_features(df, tr_idx_final, val_idx_final, te_idx, seed)

    # Softmax（在共享特征上的 LR）
    P_lr_tr, P_lr_te, acc_lr_tr, acc_lr_te = softmax_from_shared_features(
        meta, classes, seed
    )

    # MLP（在共享特征上的 MLP）
    P_mlp_tr, P_mlp_te, acc_mlp_tr, acc_mlp_te = mlp_from_shared_features(
        meta, classes, seed
    )

    # 简单 0.5 / 0.5 集成
    P_tr = w_lr * P_lr_tr + w_mlp * P_mlp_tr
    P_te = w_lr * P_lr_te + w_mlp * P_mlp_te

    pred_tr = np.array([classes[i] for i in np.argmax(P_tr, axis=1)])
    pred_te = np.array([classes[i] for i in np.argmax(P_te, axis=1)])
    y_tr = df.loc[tr_idx_final, TARGET_COL].astype(str).values
    y_te = df.loc[te_idx, TARGET_COL].astype(str).values

    acc_tr = accuracy_score(y_tr, pred_tr)
    acc_te = accuracy_score(y_te, pred_te)
    return acc_tr, acc_te, (acc_lr_tr, acc_lr_te), (acc_mlp_tr, acc_mlp_te)

def main():
    ens_tr, ens_te = [], []
    lr_tr, lr_te = [], []
    mlp_tr, mlp_te = [], []

    for i in range(N_RUNS):
        seed = SEED_BASE + i
        seed_all(seed)
        a_tr, a_te, (l_tr, l_te), (m_tr, m_te) = run_once(seed)
        ens_tr.append(a_tr); ens_te.append(a_te)
        lr_tr.append(l_tr);  lr_te.append(l_te)
        mlp_tr.append(m_tr); mlp_te.append(m_te)

    print("\n=== Variant 3: Softmax & MLP 共享结构化特征工程 ===")
    print(f"Runs = {N_RUNS}")
    print(f"[Ensemble] Train acc mean = {np.mean(ens_tr):.4f} | std = {np.std(ens_tr):.4f}")
    print(f"[Ensemble] Test  acc mean = {np.mean(ens_te):.4f} | std = {np.std(ens_te):.4f}")
    print(f"[Softmax ] Train acc mean = {np.mean(lr_tr):.4f} | Test acc mean = {np.mean(lr_te):.4f}")
    print(f"[MLP     ] Train acc mean = {np.mean(mlp_tr):.4f} | Test acc mean = {np.mean(mlp_te):.4f}")

if __name__ == "__main__":
    main()
