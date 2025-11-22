# pred.py
# -*- coding: utf-8 -*-
"""
预测脚本

- 只依赖：标准库、numpy、pandas
- 在当前目录下加载一个参数文件 model_params_v3.pkl
"""

import os
import re
import math
import pickle
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# ----------------- 全局配置 / 常量（与训练脚本保持一致） -----------------

PARAMS_FILE = "model_params_v3.pkl"   # 训练脚本导出的参数文件名
TARGET_COL = "label"

# === 关键问卷列名（必须与训练脚本一致） ===
LIKERT_ACADEMIC = "How likely are you to use this model for academic tasks?"
LIKERT_SUBOPT_FREQ = "Based on your experience, how often has this model given you a response that felt suboptimal?"
LIKERT_EXPECT_REF = "How often do you expect this model to provide responses with references or supporting evidence?"
LIKERT_VERIFY_FREQ = "How often do you verify this model's responses?"

MULTI_BEST = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
MULTI_SUBOPT = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"

# 多选题关键字（用于 multi-hot 展开）
TASK_KEYWORDS = {
    "math": "Math computations",
    "coding": "Writing or debugging code",
    "data": "Data processing or analysis",
    "draft": "Drafting professional text",
    "writing": "Writing or editing essays",
    "explain": "Explaining complex concepts simply",
    "convert": "Converting content between formats",
}

# TF-IDF tokenizer 使用的正则（对齐 sklearn 默认 token_pattern）
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")

# ----------------- 一些与训练端一致的工具函数 -----------------


def detect_id_like_columns(df: pd.DataFrame) -> List[str]:
    """
    根据列名中是否含 'id' / 'uuid' / 'guid' 识别 ID-like 列。
    与训练脚本保持一致。
    """
    id_cols: List[str] = []
    pat = re.compile(r"(?:^|[_\-])(?:id|uuid|guid)(?:$|[_\-])", re.I)
    for c in df.columns:
        if pat.search(str(c)):
            id_cols.append(c)
    return id_cols


def expand_multi_select(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """
    多选题拆分成 multi-hot：
    - prefix_math, prefix_coding, ...
    - 删除原长文本列 col
    """
    if col not in df.columns:
        return df
    s = df[col].fillna("")
    for key, pat in TASK_KEYWORDS.items():
        new_col = f"{prefix}_{key}"
        df[new_col] = s.str.contains(pat, regex=False).astype("float32")
    df = df.drop(columns=[col])
    return df


def tokenize(text: str) -> List[str]:
    """模仿 sklearn 默认 tokenizer：全部小写 + 正则分词。"""
    if not isinstance(text, str):
        if text is None or (isinstance(text, float) and math.isnan(text)):
            text = ""
        else:
            text = str(text)
    text = text.lower()
    return TOKEN_PATTERN.findall(text)


def generate_ngrams(tokens: List[str], min_n: int, max_n: int) -> List[str]:
    """生成 n-gram（含 1-gram, 2-gram 等），与 TfidfVectorizer(ngram_range) 对齐。"""
    ngrams: List[str] = []
    L = len(tokens)
    for n in range(min_n, max_n + 1):
        if L < n:
            continue
        for i in range(L - n + 1):
            ngrams.append(" ".join(tokens[i:i + n]))
    return ngrams


def softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    """稳定版 softmax."""
    z = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(z)
    sum_exp = np.sum(exp_z, axis=axis, keepdims=True)
    sum_exp[sum_exp == 0.0] = 1.0
    return exp_z / sum_exp


# ----------------- 参数加载（只用一次，全局缓存） -----------------

_PARAMS_CACHE: Dict[str, Any] | None = None


def _get_default_params_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, PARAMS_FILE)


def load_model_params(path: str | None = None) -> Dict[str, Any]:
    """
    从 pickle 文件中加载训练好的全部参数。
    只包含 numpy 数组 / python 基本类型，不包含 sklearn / torch 对象。
    """
    global _PARAMS_CACHE
    if _PARAMS_CACHE is not None:
        return _PARAMS_CACHE

    if path is None:
        path = _get_default_params_path()

    with open(path, "rb") as f:
        params = pickle.load(f)

    _PARAMS_CACHE = params
    return params


# ----------------- 文本：TF-IDF + SVD 特征 -----------------


def tfidf_transform(texts: List[str], params: Dict[str, Any]) -> np.ndarray:
    """
    用训练脚本导出的 vocabulary_ / idf_ 在推理端手写 TF-IDF 变换。
    - 只支持 analyzer="word"、use_idf=True、norm="l2"、sublinear_tf=False
    """
    vocab: Dict[str, int] = params["tfidf_vocabulary"]
    idf: np.ndarray = np.asarray(params["tfidf_idf"], dtype=np.float32)
    min_n = int(params["tfidf_ngram_min"])
    max_n = int(params["tfidf_ngram_max"])

    V = idf.shape[0]
    N = len(texts)
    X = np.zeros((N, V), dtype=np.float32)

    for i, doc in enumerate(texts):
        tokens = tokenize(doc)
        if not tokens:
            continue

        counts: Dict[int, int] = {}
        for ngram in generate_ngrams(tokens, min_n, max_n):
            j = vocab.get(ngram)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1

        if not counts:
            continue

        row = np.zeros(V, dtype=np.float32)
        for j, c in counts.items():
            row[j] = float(c)

        # tf * idf
        row *= idf

        # l2 归一化
        norm = float(np.sqrt(np.sum(row ** 2)))
        if norm > 0.0:
            row /= norm

        X[i] = row

    return X


def text_features(text_df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    """
    - 对 text_cols 做行级拼接（与训练时 join_text_columns 相同）
    - 调用 tfidf_transform
    - 再乘以训练时导出的 TruncatedSVD.components_.T 得到 100 维 LSA 特征
    """
    if text_df.shape[1] == 0:
        # 没有文本列
        return np.zeros((len(text_df), 0), dtype=np.float32)

    # 行拼接
    docs = (
        text_df.astype(str)
        .fillna("")
        .agg(" ".join, axis=1)
        .tolist()
    )

    X_tfidf = tfidf_transform(docs, params)
    components = np.asarray(params["svd_components"], dtype=np.float32)  # shape: (n_comp, vocab_size)
    # (N, V) @ (V, n_comp)^T -> (N, n_comp)
    X_lsa = X_tfidf @ components.T
    return X_lsa.astype(np.float32)


# ----------------- 结构化特征工程（对齐训练端） -----------------


def build_features_from_df(df: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    """
    对输入 DataFrame 做与训练端一致的特征工程，返回一个
    (n_samples, n_features) 的 numpy.float32 矩阵。

    步骤：
    1. 丢掉 ID-like 列 & student_id
    2. 多选题展开 multi-hot
    3. 按训练时记录的 num_cols / cat_cols / text_cols 划分
    4. 数值 median 填充；类别常量填充 + one-hot；文本 TF-IDF+SVD
    5. 整体 MaxAbsScaler
    """
    dfm = df.copy()

    # 1) 丢掉 ID-like 列（含 student_id），对齐 _compute_shared_features
    id_cols = detect_id_like_columns(dfm.drop(columns=[TARGET_COL], errors="ignore"))
    if "student_id" in dfm.columns:
        id_cols = list(set(id_cols) | {"student_id"})
    if id_cols:
        dfm = dfm.drop(columns=id_cols, errors="ignore")

    # 2) 多选题展开（best_*, subopt_*）
    dfm = expand_multi_select(dfm, MULTI_BEST,  prefix="best")
    dfm = expand_multi_select(dfm, MULTI_SUBOPT, prefix="subopt")

    # 之后的列选择与训练时一致：用保存下来的列名，而不是重新自动检测
    X_df = dfm.drop(columns=[TARGET_COL], errors="ignore")

    num_cols: List[str] = params["num_cols"]
    cat_cols: List[str] = params["cat_cols"]
    text_cols: List[str] = params["text_cols"]

    N = len(X_df)

    # ---- 数值列：median 填充 ----
    if num_cols:
        num_medians = np.asarray(params["num_medians"], dtype=np.float32)
        X_num = np.zeros((N, len(num_cols)), dtype=np.float32)
        for j, col in enumerate(num_cols):
            # 尽量转成数值；失败的视为 NaN
            col_ser = pd.to_numeric(X_df[col], errors="coerce")
            col_arr = col_ser.to_numpy(dtype=np.float32, copy=True)
            mask = np.isnan(col_arr)
            if np.any(mask):
                col_arr[mask] = num_medians[j]
            X_num[:, j] = col_arr
    else:
        X_num = np.zeros((N, 0), dtype=np.float32)

    # ---- 类别列：常量填充 + one-hot ----
    if cat_cols:
        cat_categories = params["cat_categories"]   # list of arrays per column
        cat_fill_values = params["cat_fill_values"] # list of str, 与 cat_cols 对齐

        # 计算总的 one-hot 维度
        n_cat_features = int(sum(len(np.asarray(c)) for c in cat_categories))
        X_cat = np.zeros((N, n_cat_features), dtype=np.float32)

        offset = 0
        for col_idx, col in enumerate(cat_cols):
            cats = list(np.asarray(cat_categories[col_idx], dtype=object))
            fill_val = str(cat_fill_values[col_idx])
            if col not in X_df.columns:
                # 极端情况：测试集中没有这个列，整列视为 fill_val
                values = pd.Series([fill_val] * N)
            else:
                values = X_df[col].astype(object).where(
                    ~X_df[col].isna(),
                    other=fill_val,
                )

            vals_arr = values.astype(str).to_numpy()
            mapping = {cat: j for j, cat in enumerate(cats)}

            for i in range(N):
                v = vals_arr[i]
                j_local = mapping.get(v)
                if j_local is not None:
                    X_cat[i, offset + j_local] = 1.0

            offset += len(cats)
    else:
        X_cat = np.zeros((N, 0), dtype=np.float32)

    # ---- 文本列：join -> TF-IDF -> SVD ----
    if text_cols:
        text_df = X_df[text_cols]
        X_text = text_features(text_df, params)
    else:
        X_text = np.zeros((N, 0), dtype=np.float32)

    # ---- 拼接 + MaxAbsScaler ----
    X_full = np.concatenate([X_num, X_cat, X_text], axis=1).astype(np.float32)

    max_abs = np.asarray(params["maxabs_scale"], dtype=np.float32)  # 训练时的 max_abs_
    if max_abs.ndim != 1 or max_abs.shape[0] != X_full.shape[1]:
        raise ValueError(
            f"MaxAbsScaler 维度不匹配: max_abs.shape={max_abs.shape}, X_full.shape={X_full.shape}"
        )
    scale = max_abs.copy()
    scale[scale == 0.0] = 1.0
    X_scaled = X_full / scale

    return X_scaled


# ----------------- 模型前向：LR + MLP 集成 -----------------


def predict_proba_with_params(X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    给定特征矩阵 X 和导出的参数，计算每个样本的类别概率（Ensemble 后的结果）。
    """
    # ---- Softmax Logistic Regression ----
    lr_coef = np.asarray(params["lr_coef"], dtype=np.float32)      # (n_classes, n_features)
    lr_intercept = np.asarray(params["lr_intercept"], dtype=np.float32)  # (n_classes,)

    logits_lr = X @ lr_coef.T + lr_intercept  # (N, n_classes)
    prob_lr = softmax(logits_lr, axis=1)

    # ---- MLP (两层 ReLU，全连接) ----
    # 可能你只想用 LR，那就把这些权重存为 shape 0，下面会自动跳过
    W1 = np.asarray(params.get("mlp_W1", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    b1 = np.asarray(params.get("mlp_b1", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    W2 = np.asarray(params.get("mlp_W2", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    b2 = np.asarray(params.get("mlp_b2", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    W3 = np.asarray(params.get("mlp_W3", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    b3 = np.asarray(params.get("mlp_b3", np.zeros((0,), dtype=np.float32)), dtype=np.float32)

    if W1.size == 0 or W2.size == 0 or W3.size == 0:
        # 没有 MLP 权重，就退化为纯 LR
        prob_mlp = prob_lr
        w_lr_val = 1.0
        w_mlp_val = 0.0
    else:
        # X: (N, d), W1: (h1, d)
        h1 = np.maximum(0.0, X @ W1.T + b1)      # ReLU
        h2 = np.maximum(0.0, h1 @ W2.T + b2)     # ReLU
        logits_mlp = h2 @ W3.T + b3              # (N, n_classes)
        prob_mlp = softmax(logits_mlp, axis=1)

        w_lr_val = float(params.get("w_lr", 0.4))
        w_mlp_val = float(params.get("w_mlp", 0.6))

    prob_ens = w_lr_val * prob_lr + w_mlp_val * prob_mlp
    return prob_ens


# ----------------- 对外接口：predict_all -----------------


def predict_all(csv_filename: str):
    """
    接口：
    - 输入: CSV 文件路径 (str)
    - 输出: 对应行的预测标签列表（按行顺序对齐）
    """
    params = load_model_params()
    df = pd.read_csv(csv_filename)

    # 特征工程
    X = build_features_from_df(df, params)

    # 概率 & 预测
    prob = predict_proba_with_params(X, params)
    classes: List[str] = list(params["classes"])
    pred_idx = np.argmax(prob, axis=1)
    preds = [classes[i] for i in pred_idx]

    return preds


# 本地调试用
if __name__ == "__main__":
    # 简单 smoke test：如果有一个 test_data_clean.csv，就打印前几条预测
    test_csv = "training_data_clean.csv"
    if os.path.exists(test_csv):
        ys = predict_all(test_csv)
        print("Predictions for", test_csv)
        print(ys[:10])
    else:
        print("No test_data_clean.csv found. Only predict_all() is required for grading.")
