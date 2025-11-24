import re, math, random
import numpy as np
import pandas as pd

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

# ---------------- Config ----------------
CSV_PATH     = "training_data_clean.csv"
TARGET_COL   = "label"
TEST_SIZE    = 0.20
VAL_SIZE     = 0.15

SOFTMAX_C_GRID       = [1.0, 0.5, 0.2, 0.1, 0.05]
SOFTMAX_REFIT_TRAIN  = True 

TFIDF_MIN_DF   = 2
TFIDF_NGRAM    = (1, 2)
TFIDF_MAX_FEAT = None     
TFIDF_MAX_DF   = 1.0      

# SVD + MLP
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

# Ensemble 
w_lr  = 0.4
w_mlp = 0.6

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
        raise ValueError(f"lack of {TARGET_COL}")
    if "student_id" not in df.columns:
        df["student_id"] = np.arange(len(df))
        print("[WARN] no student_id")
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
        sparse_threshold=0.0  
    )
    return Pipeline([("preprocess", preprocess), ("scale", MaxAbsScaler())])

def encode_ordinal_likert(df):
    for col, mapping in ORDINAL_COL_MAPS.items():
        if col not in df.columns:
            continue
        df[col] = df[col].map(lambda x: mapping.get(str(x).strip(), np.nan))
    return df

def expand_multi_select(df, col, prefix):
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
            print("[WARN] use last one")
            return last

def _align_proba(P_raw, clf_classes, classes):
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
        X_train_full = np.vstack([X_tr, X_val])
        y_train_full = np.concatenate([y_tr, y_val])

        best = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=best_c,
            max_iter=3000,
            random_state=seed
        ).fit(X_train_full, y_train_full)
    return best, best_c, best_acc

def _compute_shared_features(df, train_idx, val_idx, test_idx, seed):
    dfm = df.copy()

    # drop id column
    id_cols = detect_id_like_columns(dfm.drop(columns=[TARGET_COL], errors="ignore"))
    if "student_id" in dfm.columns:
        id_cols = list(set(id_cols) | {"student_id"})
    if id_cols:
        dfm = dfm.drop(columns=id_cols, errors="ignore")

    dfm = expand_multi_select(dfm, MULTI_BEST,  prefix="best")
    dfm = expand_multi_select(dfm, MULTI_SUBOPT, prefix="subopt")

    X = dfm.drop(columns=[TARGET_COL])
    y = dfm[TARGET_COL].astype(str)

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
    X_tr_np  = meta["X_tr_np"]
    X_val_np = meta["X_val_np"]
    y_tr     = meta["y_tr"]
    y_val    = meta["y_val"]

    clf, best_c, _ = _train_lr_with_val(X_tr_np, y_tr, X_val_np, y_val, seed, SOFTMAX_C_GRID)

    P_tr  = _align_proba(clf.predict_proba(X_tr_np),  clf.classes_, classes)
    P_val = _align_proba(clf.predict_proba(X_val_np), clf.classes_, classes)

    pred_tr  = np.array([classes[i] for i in np.argmax(P_tr, axis=1)])
    pred_val = np.array([classes[i] for i in np.argmax(P_val, axis=1)])
    acc_tr   = accuracy_score(y_tr,  pred_tr)
    acc_val  = accuracy_score(y_val, pred_val)
    return P_tr, P_val, acc_tr, acc_val

def mlp_from_shared_features(meta, classes, seed):
    seed_all(seed)

    X_tr_np  = meta["X_tr_np"]
    X_val_np = meta["X_val_np"]
    y_tr     = meta["y_tr"]
    y_val    = meta["y_val"]

    cls2idx = {c: i for i, c in enumerate(classes)}
    y_tr_np  = np.array([cls2idx[s] for s in y_tr],  np.int64)
    y_val_np = np.array([cls2idx[s] for s in y_val], np.int64)

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

        with torch.no_grad():
            model.eval()
            val_losses = []
            for xb, yb in val_loader:
                xb = torch.tensor(xb, dtype=torch.float32, device=DEVICE)
                yb = torch.tensor(yb, dtype=torch.long, device=DEVICE)
                val_losses.append(criterion(model(xb), yb).item() * len(yb))
            val_loss = float(np.sum(val_losses)) / max(1, len(y_val_np))
            scheduler.step(val_loss)

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

    train_full = DataLoader(NPDataset(X_tr_np, y_tr_np), batch_size=BATCH_SIZE, shuffle=False)
    y_tr_true, y_tr_pred, P_tr = eval_model_with_probs(model, train_full)

    val_full   = DataLoader(NPDataset(X_val_np, y_val_np), batch_size=BATCH_SIZE, shuffle=False)
    y_val_true, y_val_pred, P_val = eval_model_with_probs(model, val_full)

    acc_tr  = accuracy_score(y_tr_true,  y_tr_pred)
    acc_val = accuracy_score(y_val_true, y_val_pred)
    return P_tr, P_val, acc_tr, acc_val

def run_once(seed):
    df = load_df(CSV_PATH)
    y_all = df[TARGET_COL].astype(str)
    groups = df["student_id"].astype(str)

    tr_idx, te_idx = grouped_split_with_all_classes(y_all, groups, TEST_SIZE, seed)

    y_tr_all = y_all.iloc[tr_idx]
    groups_tr = groups.iloc[tr_idx]
    idx_tr_in, idx_val_in = grouped_split_with_all_classes(y_tr_all, groups_tr, VAL_SIZE, seed + 1)
    tr_idx_final = np.array(tr_idx)[idx_tr_in]
    val_idx_final = np.array(tr_idx)[idx_val_in]

    classes = sorted(y_all.iloc[tr_idx_final].unique())

    meta = _compute_shared_features(df, tr_idx_final, val_idx_final, te_idx, seed)

    P_lr_tr, P_lr_val, acc_lr_tr, acc_lr_val = softmax_from_shared_features(
        meta, classes, seed
    )

    P_mlp_tr, P_mlp_val, acc_mlp_tr, acc_mlp_val = mlp_from_shared_features(
        meta, classes, seed
    )

    P_tr  = w_lr * P_lr_tr  + w_mlp * P_mlp_tr
    P_val = w_lr * P_lr_val + w_mlp * P_mlp_val

    pred_tr  = np.array([classes[i] for i in np.argmax(P_tr,  axis=1)])
    pred_val = np.array([classes[i] for i in np.argmax(P_val, axis=1)])
    y_tr  = df.loc[tr_idx_final,  TARGET_COL].astype(str).values
    y_val = df.loc[val_idx_final, TARGET_COL].astype(str).values

    # Ensemble：acc & macro-F1（train / val）
    acc_tr  = accuracy_score(y_tr,  pred_tr)
    acc_val = accuracy_score(y_val, pred_val)

    f1_ens_tr  = f1_score(y_tr,  pred_tr,  average="macro")
    f1_ens_val = f1_score(y_val, pred_val, average="macro")

    # Ensemble：validation per-class precision & recall
    prec_val, rec_val, _, _ = precision_recall_fscore_support(
        y_val, pred_val, labels=classes, average=None, zero_division=0
    )

    # Softmax & MLP macro-F1（validation）
    pred_lr_val = np.array([classes[i] for i in np.argmax(P_lr_val, axis=1)])
    pred_mlp_val = np.array([classes[i] for i in np.argmax(P_mlp_val, axis=1)])

    f1_lr_val  = f1_score(y_val, pred_lr_val,  average="macro")
    f1_mlp_val = f1_score(y_val, pred_mlp_val, average="macro")

    return (acc_tr, acc_val, f1_ens_tr, f1_ens_val,
            prec_val, rec_val,
            (acc_lr_tr,  acc_lr_val,  f1_lr_val),
            (acc_mlp_tr, acc_mlp_val, f1_mlp_val),
            classes)

def main():
    ens_tr_acc, ens_val_acc = [], []
    ens_tr_f1,  ens_val_f1  = [], []
    lr_tr_acc,  lr_val_acc  = [], []
    mlp_tr_acc, mlp_val_acc = [], []

    # per-class precision/recall (validation, ensemble)
    prec_runs = []
    rec_runs  = []
    class_names = None

    for i in range(N_RUNS):
        seed = SEED_BASE + i
        seed_all(seed)
        (a_tr, a_val, f_tr, f_val,
         prec_val, rec_val,
         (l_tr, l_val, l_f1),
         (m_tr, m_val, m_f1),
         classes) = run_once(seed)

        ens_tr_acc.append(a_tr);   ens_val_acc.append(a_val)
        ens_tr_f1.append(f_tr);    ens_val_f1.append(f_val)
        lr_tr_acc.append(l_tr);    lr_val_acc.append(l_val)
        mlp_tr_acc.append(m_tr);   mlp_val_acc.append(m_val)

        prec_runs.append(prec_val)
        rec_runs.append(rec_val)
        if class_names is None:
            class_names = classes

    prec_runs = np.stack(prec_runs)  # (N_RUNS, n_classes)
    rec_runs  = np.stack(rec_runs)

    mean_prec = prec_runs.mean(axis=0)
    std_prec  = prec_runs.std(axis=0)
    mean_rec  = rec_runs.mean(axis=0)
    std_rec   = rec_runs.std(axis=0)

    print(f"Runs = {N_RUNS}")
    print(f"[Ensemble] Train acc mean = {np.mean(ens_tr_acc):.4f} | std = {np.std(ens_tr_acc):.4f}")
    print(f"[Ensemble] Val   acc mean = {np.mean(ens_val_acc):.4f} | std = {np.std(ens_val_acc):.4f}")
    print(f"[Ensemble] Val   F1  mean = {np.mean(ens_val_f1):.4f} | std = {np.std(ens_val_f1):.4f}")

    print(f"[Softmax ] Train acc mean = {np.mean(lr_tr_acc):.4f} | Val acc mean = {np.mean(lr_val_acc):.4f}")
    print(f"[MLP     ] Train acc mean = {np.mean(mlp_tr_acc):.4f} | Val acc mean = {np.mean(mlp_val_acc):.4f}")

    print("\n[Ensemble] Validation per-class Precision (mean ± std):")
    for c, m, s in zip(class_names, mean_prec, std_prec):
        print(f"  {c:>10}: {m:.4f} ± {s:.4f}")

    print("\n[Ensemble] Validation per-class Recall (mean ± std):")
    for c, m, s in zip(class_names, mean_rec, std_rec):
        print(f"  {c:>10}: {m:.4f} ± {s:.4f}")

if __name__ == "__main__":
    main()
