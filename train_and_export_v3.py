"""
train_and_export_v3.py

基于 训练脚本的完整训练以及导出脚本。
- 跑一次和原来 run_once/main 一样的流程（单个 seed）；
- 训练 Softmax(LR) + MLP；
- 保存：
    * 之前的调试用文件：preprocessor.pkl / softmax_lr.pkl / mlp_state_dict.pt / classes.pkl
    * 额外导出一个  model_params_v3.pkl  给 pred.py 使用（纯 numpy 参数）
"""

import os
import math
import pickle   # 导出给 pred.py 用的参数
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

import final_model as m


MODEL_DIR = "model_softmax_mlp_v3"
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
LR_MODEL_PATH     = os.path.join(MODEL_DIR, "softmax_lr.pkl")
MLP_STATE_PATH    = os.path.join(MODEL_DIR, "mlp_state_dict.pt")
CLASSES_PATH      = os.path.join(MODEL_DIR, "classes.pkl")
META_PATH         = os.path.join(MODEL_DIR, "training_meta.pkl")

# 给 pred.py 用的参数文件，放在当前脚本所在目录
PARAMS_OUT_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "model_params_v3.pkl")


def train_mlp_on_shared_features(meta, classes, seed):
    m.seed_all(seed)

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

    model = m.MLP(in_dim=X_tr_np.shape[1], n_classes=len(classes), p=m.DROPOUT_P).to(m.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=m.LR, weight_decay=m.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(m.NPDataset(X_tr_np,  y_tr_np), batch_size=m.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(m.NPDataset(X_val_np, y_val_np), batch_size=m.BATCH_SIZE, shuffle=False)

    best_val = math.inf
    best_state = None
    patience_left = m.PATIENCE

    for epoch in range(1, m.MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = torch.tensor(xb, dtype=torch.float32, device=m.DEVICE)
            yb = torch.tensor(yb, dtype=torch.long, device=m.DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), m.GRAD_CLIP)
            opt.step()

        # 验证集 early stopping
        with torch.no_grad():
            model.eval()
            val_losses = []
            for xb, yb in val_loader:
                xb = torch.tensor(xb, dtype=torch.float32, device=m.DEVICE)
                yb = torch.tensor(yb, dtype=torch.long, device=m.DEVICE)
                val_losses.append(criterion(model(xb), yb).item() * len(yb))
            val_loss = float(np.sum(val_losses)) / max(1, len(y_val_np))
            scheduler.step(val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = m.PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 在 train + test 上评估（仍然只用 train 训练）
    train_full = DataLoader(m.NPDataset(X_tr_np, y_tr_np), batch_size=m.BATCH_SIZE, shuffle=False)
    y_tr_true, y_tr_pred, P_tr = m.eval_model_with_probs(model, train_full)

    test_loader = DataLoader(m.NPDataset(X_te_np, y_te_np), batch_size=m.BATCH_SIZE, shuffle=False)
    y_te_true, y_te_pred, P_te = m.eval_model_with_probs(model, test_loader)

    acc_tr = accuracy_score(y_tr_true, y_tr_pred)
    acc_te = accuracy_score(y_te_true, y_te_pred)
    return model, P_tr, P_te, acc_tr, acc_te


def export_params_for_pred(meta, lr_clf, mlp_model, classes,
                           out_path=PARAMS_OUT_PATH,
                           w_lr=m.w_lr, w_mlp=m.w_mlp):
    """
    把 sklearn 的 preprocessor + LR、PyTorch 的 MLP
    拆成 pred.py 需要的纯 numpy 参数
    """
    pre = meta["pre"]                          # Pipeline(preprocess -> scale)
    preprocess = pre.named_steps["preprocess"] # ColumnTransformer
    scaler = pre.named_steps["scale"]          # MaxAbsScaler

    # 取出 num / cat / text 三个子 pipeline 和列名
    name_to_tr = {name: (trans, cols) for name, trans, cols in preprocess.transformers_}

    # --- 数值部分 ---
    num_pipe, num_cols = name_to_tr["num"]
    num_imputer = num_pipe.named_steps["imputer"]
    num_medians = num_imputer.statistics_.astype("float32")

    # --- 类别部分 ---
    cat_pipe, cat_cols = name_to_tr["cat"]
    cat_imputer = cat_pipe.named_steps["imputer"]
    cat_fill_values = [str(x) for x in cat_imputer.statistics_]
    ohe = cat_pipe.named_steps["ohe"]
    cat_categories = [np.array(c, dtype=object) for c in ohe.categories_]

    # --- 文本部分 ---
    text_pipe, text_cols = name_to_tr["text"]
    tfidf = text_pipe.named_steps["tfidf"]
    svd = text_pipe.named_steps["svd"]

    vocab = {str(k): int(v) for k, v in tfidf.vocabulary_.items()}
    idf = tfidf.idf_.astype("float32")
    ngram_min, ngram_max = tfidf.ngram_range
    svd_components = svd.components_.astype("float32")

    # --- MaxAbsScaler ---
    max_abs = scaler.max_abs_.astype("float32")

    # --- Softmax LR 参数 ---
    lr_coef = lr_clf.coef_.astype("float32")        # (n_classes, n_features)
    lr_intercept = lr_clf.intercept_.astype("float32")

    # --- MLP 参数 ---
    state = mlp_model.state_dict()
    W1 = state["net.0.weight"].detach().cpu().numpy().astype("float32")
    b1 = state["net.0.bias"].detach().cpu().numpy().astype("float32")
    W2 = state["net.3.weight"].detach().cpu().numpy().astype("float32")
    b2 = state["net.3.bias"].detach().cpu().numpy().astype("float32")
    W3 = state["net.6.weight"].detach().cpu().numpy().astype("float32")
    b3 = state["net.6.bias"].detach().cpu().numpy().astype("float32")

    params = {
        "target_col": m.TARGET_COL,
        "classes": [str(c) for c in classes],
        "w_lr": float(w_lr),
        "w_mlp": float(w_mlp),

        "num_cols": list(num_cols),
        "num_medians": num_medians,

        "cat_cols": list(cat_cols),
        "cat_fill_values": cat_fill_values,
        "cat_categories": cat_categories,

        "text_cols": list(text_cols),
        "tfidf_vocabulary": vocab,
        "tfidf_idf": idf,
        "tfidf_ngram_min": int(ngram_min),
        "tfidf_ngram_max": int(ngram_max),
        "svd_components": svd_components,

        "maxabs_scale": max_abs,

        "lr_coef": lr_coef,
        "lr_intercept": lr_intercept,

        "mlp_W1": W1,
        "mlp_b1": b1,
        "mlp_W2": W2,
        "mlp_b2": b2,
        "mlp_W3": W3,
        "mlp_b3": b3,
    }

    with open(out_path, "wb") as f:
        pickle.dump(params, f)

    print(f"\n[export_params_for_pred] 已导出 pred.py 使用的参数到: {out_path}")


def train_and_export(seed=None, model_dir=MODEL_DIR):
    """
    跑一次完整训练，并把模型导出。
    """
    if seed is None:
        seed = m.SEED_BASE

    os.makedirs(model_dir, exist_ok=True)
    print(f"[INFO] 使用 seed = {seed}")
    print(f"[INFO] 模型将保存到目录: {model_dir}")

    # 1. 读取数据 & 分组划分 train/val/test
    df = m.load_df(m.CSV_PATH)
    y_all = df[m.TARGET_COL].astype(str)
    groups = df["student_id"].astype(str)

    # 先切 train/test
    tr_idx, te_idx = m.grouped_split_with_all_classes(y_all, groups, m.TEST_SIZE, seed)

    # 再在 train 部分切出 val
    y_tr_all = y_all.iloc[tr_idx]
    groups_tr = groups.iloc[tr_idx]
    idx_tr_in, idx_val_in = m.grouped_split_with_all_classes(
        y_tr_all, groups_tr, m.VAL_SIZE, seed + 1
    )
    tr_idx_final = np.array(tr_idx)[idx_tr_in]
    val_idx_final = np.array(tr_idx)[idx_val_in]

    # 类别集合只看训练集
    classes = sorted(y_all.iloc[tr_idx_final].unique())
    print(f"[INFO] 类别数 = {len(classes)}，类别 = {classes}")

    # 2. 共享特征工程
    meta = m._compute_shared_features(df, tr_idx_final, val_idx_final, te_idx, seed)
    X_tr_np  = meta["X_tr_np"]
    X_val_np = meta["X_val_np"]
    X_te_np  = meta["X_te_np"]
    y_tr     = meta["y_tr"]
    y_val    = meta["y_val"]
    y_te     = meta["y_te"]

    # 3. Softmax(LR)：用 train 决定特征，用 val 选 C
    lr_clf, best_c, best_acc_val = m._train_lr_with_val(
        X_tr_np, y_tr, X_val_np, y_val, seed, m.SOFTMAX_C_GRID
    )

    P_lr_tr = m._align_proba(lr_clf.predict_proba(X_tr_np), lr_clf.classes_, classes)
    P_lr_te = m._align_proba(lr_clf.predict_proba(X_te_np), lr_clf.classes_, classes)
    pred_tr_lr = np.array([classes[i] for i in np.argmax(P_lr_tr, axis=1)])
    pred_te_lr = np.array([classes[i] for i in np.argmax(P_lr_te, axis=1)])
    acc_lr_tr = accuracy_score(y_tr, pred_tr_lr)
    acc_lr_te = accuracy_score(y_te, pred_te_lr)

    # 4. MLP
    mlp_model, P_mlp_tr, P_mlp_te, acc_mlp_tr, acc_mlp_te = train_mlp_on_shared_features(
        meta, classes, seed
    )

    # 5. 加权集成
    P_tr_ens = m.w_lr * P_lr_tr + m.w_mlp * P_mlp_tr
    P_te_ens = m.w_lr * P_lr_te + m.w_mlp * P_mlp_te

    pred_tr_ens = np.array([classes[i] for i in np.argmax(P_tr_ens, axis=1)])
    pred_te_ens = np.array([classes[i] for i in np.argmax(P_te_ens, axis=1)])
    acc_tr_ens = accuracy_score(y_tr, pred_tr_ens)
    acc_te_ens = accuracy_score(y_te, pred_te_ens)

    # 6. 打印训练结果
    print("\n=== 单次训练结果 ===")
    print(f"[Softmax ] Train acc = {acc_lr_tr:.4f} | Test acc = {acc_lr_te:.4f}")
    print(f"[MLP     ] Train acc = {acc_mlp_tr:.4f} | Test acc = {acc_mlp_te:.4f}")
    print(f"[Ensemble] Train acc = {acc_tr_ens:.4f} | Test acc = {acc_te_ens:.4f}")
    print(f"[Softmax ] best C on val = {best_c} | best val acc = {best_acc_val:.4f}")

    # 7. 保留原来的 joblib / torch 保存（方便自己调试，不给 pred.py 用）
    joblib.dump(meta["pre"], PREPROCESSOR_PATH)
    joblib.dump(lr_clf, LR_MODEL_PATH)
    joblib.dump(list(classes), CLASSES_PATH)
    torch.save(mlp_model.state_dict(), MLP_STATE_PATH)

    meta_info = {
        "seed": int(seed),
        "best_C": float(best_c) if best_c is not None else None,
        "best_val_acc_softmax": float(best_acc_val),
        "acc_lr_tr": float(acc_lr_tr),
        "acc_lr_te": float(acc_lr_te),
        "acc_mlp_tr": float(acc_mlp_tr),
        "acc_mlp_te": float(acc_mlp_te),
        "acc_ens_tr": float(acc_tr_ens),
        "acc_ens_te": float(acc_te_ens),
        "w_lr": float(m.w_lr),
        "w_mlp": float(m.w_mlp),
        "csv_path": m.CSV_PATH,
        "target_col": m.TARGET_COL,
        "test_size": float(m.TEST_SIZE),
        "val_size": float(m.VAL_SIZE),
    }
    joblib.dump(meta_info, META_PATH)

    print(f"\n[INFO] 预处理器已保存到: {PREPROCESSOR_PATH}")
    print(f"[INFO] Softmax(LR) 模型已保存到: {LR_MODEL_PATH}")
    print(f"[INFO] MLP state_dict 已保存到: {MLP_STATE_PATH}")
    print(f"[INFO] 类别列表已保存到: {CLASSES_PATH}")
    print(f"[INFO] 训练元信息已保存到: {META_PATH}")

    # 8. 导出给 pred.py 用的纯 numpy 参数
    export_params_for_pred(meta, lr_clf, mlp_model, classes,
                           out_path=PARAMS_OUT_PATH,
                           w_lr=m.w_lr, w_mlp=m.w_mlp)

    return {
        "classes": classes,
        "acc_lr": (acc_lr_tr, acc_lr_te),
        "acc_mlp": (acc_mlp_tr, acc_mlp_te),
        "acc_ens": (acc_tr_ens, acc_te_ens),
    }


if __name__ == "__main__":
    train_and_export()
