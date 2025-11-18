#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chi_square_categorical.py

针对若干个类别特征，两两之间做卡方独立性检验，
并计算 Cramér's V 作为关联强度。

使用说明：
1. 修改 DATA_PATH 为你的数据文件路径
2. 修改 CATEGORICAL_COLS 为你想要检测的 5 个类别列名
3. 运行：python chi_square_categorical.py
"""

import os
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# 需要 SciPy 做卡方检验
try:
    from scipy.stats import chi2_contingency
except ImportError:
    raise ImportError(
        "本脚本需要 SciPy 库，请先安装：pip install scipy"
    )

# ------------ 配置区（根据需要修改） ------------

DATA_PATH = "training_data_clean.csv"  # 你的数据文件路径

# 请在这里填入你要做卡方检验的 5 个类别列名
CATEGORICAL_COLS = [
    # 举例（请改成你自己的列名）：
    "Which types of tasks do you feel this model handles best? (Select all that apply.)",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How likely are you to use this model for academic tasks?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?",
    "label"
]

OUTPUT_DIR = "chi_square_report"
MIN_EXPECTED_FREQ = 1.0  # 最小期望频数阈值（低于这个会给出提醒，但仍然计算）


# ------------ 工具函数 ------------

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def chi_square_for_pair(df: pd.DataFrame,
                        col_x: str,
                        col_y: str,
                        min_expected_freq: float = MIN_EXPECTED_FREQ):
    """
    对两个类别变量做卡方独立性检验 + Cramér's V 计算。

    返回一个 dict，包括：
    - col_x, col_y
    - n: 有效样本数
    - n_levels_x, n_levels_y: 各自类别数
    - chi2, dof, p_value
    - cramers_v
    - min_expected: 列联表中的最小期望频数
    - warning: 若期望频数太小，给文本提示
    """
    # 仅保留这两列，并去掉缺失
    sub = df[[col_x, col_y]].dropna()
    n = len(sub)
    if n == 0:
        return {
            "col_x": col_x,
            "col_y": col_y,
            "n": 0,
            "n_levels_x": 0,
            "n_levels_y": 0,
            "chi2": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "cramers_v": np.nan,
            "min_expected": np.nan,
            "warning": "no valid rows (all NaN)"
        }

    # 构建列联表
    contingency = pd.crosstab(sub[col_x], sub[col_y])

    r, k = contingency.shape
    if r < 2 or k < 2:
        return {
            "col_x": col_x,
            "col_y": col_y,
            "n": n,
            "n_levels_x": r,
            "n_levels_y": k,
            "chi2": np.nan,
            "dof": np.nan,
            "p_value": np.nan,
            "cramers_v": np.nan,
            "min_expected": np.nan,
            "warning": "contingency table is <2x2, test not meaningful"
        }

    # 卡方检验
    chi2, p, dof, expected = chi2_contingency(contingency, correction=False)
    min_expected = expected.min()

    # Cramér's V
    # V = sqrt(chi2 / (n * (min(r-1, k-1))))
    denom = n * (min(r - 1, k - 1))
    if denom > 0:
        cramers_v = math.sqrt(chi2 / denom)
    else:
        cramers_v = np.nan

    warning_msg = ""
    if min_expected < min_expected_freq:
        warning_msg = (
            f"some expected counts < {min_expected_freq:.2f} "
            "(chi-square approximation may be unreliable)"
        )

    return {
        "col_x": col_x,
        "col_y": col_y,
        "n": n,
        "n_levels_x": r,
        "n_levels_y": k,
        "chi2": chi2,
        "dof": dof,
        "p_value": p,
        "cramers_v": cramers_v,
        "min_expected": min_expected,
        "warning": warning_msg
    }


def run_pairwise_chi_square(df: pd.DataFrame, cat_cols, output_dir: str):
    """
    对 cat_cols 里面的所有列两两之间做卡方检验。
    """
    ensure_dir(output_dir)

    # 确认这些列都在 df 里
    missing_cols = [c for c in cat_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"下列列名不在数据中，请检查: {missing_cols}")

    results = []
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            col_x = cat_cols[i]
            col_y = cat_cols[j]
            res = chi_square_for_pair(df, col_x, col_y)
            results.append(res)

    res_df = pd.DataFrame(results)
    # 按 p 值升序排序，方便看显著相关的组合
    res_df = res_df.sort_values("p_value", ascending=True)

    out_path = os.path.join(output_dir, "chi_square_pairwise_categorical.csv")
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n====== 卡方检验结果（前 20 对按 p 值排序） ======")
    print(res_df.head(20))

    print(f"\n完整结果已保存至: {out_path}")
    return res_df


# ------------ 主流程 ------------

def main():
    ensure_dir(OUTPUT_DIR)

    print(f"读取数据: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if len(CATEGORICAL_COLS) == 0:
        warnings.warn(
            "CATEGORICAL_COLS 目前为空，请在脚本顶部填入 5 个类别列名。"
        )
        print("当前 df.columns 为：")
        print(df.columns.tolist())
        return

    print("\n准备对以下类别列做两两卡方检验：")
    for c in CATEGORICAL_COLS:
        print(" -", c)

    res_df = run_pairwise_chi_square(df, CATEGORICAL_COLS, OUTPUT_DIR)

    # 简单给一个“关联强度”解释参考
    print("\nCramér's V 粗略参考（仅供直观理解）：")
    print(" 0.0 ~ 0.1  : 几乎无关联 / 很弱")
    print(" 0.1 ~ 0.3  : 弱关联")
    print(" 0.3 ~ 0.5  : 中等关联")
    print(" > 0.5      : 较强关联（视具体问题而定）")


if __name__ == "__main__":
    main()
