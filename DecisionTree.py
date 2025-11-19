# ==============================
# Final Decision Tree Pipeline
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("training_data_clean.csv")

# -----------------------------
# 2. Preprocessing
# -----------------------------

# 2.1 Likert-scale columns -> numeric 0–1
likert_cols = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?"
]

def map_likert(value):
    if pd.isna(value):
        return 0
    try:
        return int(str(value).split(" — ")[0]) / 5
    except:
        return 0

for col in likert_cols:
    df[col] = df[col].apply(map_likert)

# 2.2 Multi-select -> one-hot encoding
multi_select_cols = [
    "Which types of tasks do you feel this model handles best? (Select all that apply.)",
    "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
]

for col in multi_select_cols:
    if col in df.columns:
        dummies = df[col].str.get_dummies(sep=",")
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)

# 2.3 Text columns -> bag-of-words (top 50 words per column)
text_cols = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
]

def build_vocab(text_series, max_words=50):
    words = " ".join(text_series.fillna("")).lower().split()
    freq = pd.Series(words).value_counts()
    return freq.index[:max_words].tolist()

vocab = []
for col in text_cols:
    vocab += build_vocab(df[col])
vocab = list(set(vocab))

def text_to_bow(text_series, vocab):
    X = np.zeros((len(text_series), len(vocab)))
    for i, text in enumerate(text_series.fillna("")):
        tokens = text.lower().split()
        for j, word in enumerate(vocab):
            X[i, j] = tokens.count(word)
    return X

X_text = np.zeros((len(df), len(vocab)))
for col in text_cols:
    X_text += text_to_bow(df[col], vocab)

df = df.drop(columns=text_cols)

# 2.4 Encode target labels
label_map = {label: i for i, label in enumerate(df['label'].dropna().unique())}
df['label'] = df['label'].map(label_map)

# 2.5 Keep student_id for grouped splitting
student_col = 'student_id'

# 2.6 Fill missing values
df = df.fillna(0)

# -----------------------------
# 3. Combine numeric + text features
# -----------------------------
X_numeric = df.drop(columns=['label', student_col], errors='ignore').values
X_numeric = (X_numeric - X_numeric.mean(axis=0)) / (X_numeric.std(axis=0) + 1e-8)
X = np.hstack([X_numeric, X_text])
y = df['label'].values

# -----------------------------
# 4. Grouped Train/Val/Test Split by student
# -----------------------------
student_ids = df[student_col].unique()
np.random.shuffle(student_ids)

n_students = len(student_ids)
train_end = int(0.5 * n_students)
val_end = int(0.8 * n_students)

train_students = student_ids[:train_end]
val_students = student_ids[train_end:val_end]
test_students = student_ids[val_end:]

train_mask = df[student_col].isin(train_students)
val_mask = df[student_col].isin(val_students)
test_mask = df[student_col].isin(test_students)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# -----------------------------
# 5. Hyperparameter tuning: max_depth
# -----------------------------
best_depth = None
best_val_acc = 0
train_acc_list = []
val_acc_list = []

for depth in range(1, 21):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = tree.score(X_train, y_train)
    val_acc = tree.score(X_val, y_val)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_depth = depth

print(f"Best max_depth: {best_depth}, Validation Accuracy: {best_val_acc:.4f}")

# Plot accuracy vs max_depth
plt.figure(figsize=(8,5))
plt.plot(range(1,21), train_acc_list, marker='o', label="Train Accuracy")
plt.plot(range(1,21), val_acc_list, marker='o', label="Validation Accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree Accuracy vs max_depth")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# 6. Train final tree
# -----------------------------
final_tree = DecisionTreeClassifier(criterion='entropy', max_depth=best_depth, random_state=42)
final_tree.fit(X_train, y_train)

y_pred_test = final_tree.predict(X_test)

# -----------------------------
# 7. Evaluation
# -----------------------------
# Metrics per class
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average=None)
print("Precision per class:", precision)
print("Recall per class:", recall)
print("F1 per class:", f1)

# Macro averages
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred_test, average='macro')
print("\nMacro Precision:", prec_macro)
print("Macro Recall:", rec_macro)
print("Macro F1:", f1_macro)

# Full classification report
class_names = {v:k for k,v in label_map.items()}
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=class_names.values()))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Decision Tree Confusion Matrix")
plt.show()
