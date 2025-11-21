import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, \
    precision_recall_fscore_support, confusion_matrix, classification_report

# Load training data
df = pd.read_csv("training_data_clean.csv")

# Processing of Likert scale type question into numeric
likert_scale = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?"
]

def map_likert(value):
    if pd.isna(value):
        return 0
    return int(value.split(" — ")[0])

for col in likert_scale:
    df[col] = df[col].apply(map_likert)

# Multi-select one-hot
multi_select_cols = [
    "Which types of tasks do you feel this model handles best? (Select all that apply.)",
    "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
]

for col in multi_select_cols:
    dummies = df[col].str.get_dummies(sep=",")
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[col], inplace=True)

# Encode labels
label_map = {label: i for i, label in enumerate(df['label'].dropna().unique())}
df['label'] = df['label'].map(label_map)

# Text columns into bag-of-words
text_cols = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
]

def build_vocab(text_series):
    vocab = {}
    idx = 0
    for text in text_series.fillna(""):
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

# Build combined vocab
vocab = {}
for col in text_cols:
    col_vocab = build_vocab(df[col])
    for w in col_vocab:
        if w not in vocab:
            vocab[w] = len(vocab)

def count_transform(text_series, vocab):
    N = len(text_series)
    V = len(vocab)
    matrix = np.zeros((N, V))
    for i, text in enumerate(text_series.fillna("")):
        for word in text.lower().split():
            if word in vocab:
                matrix[i, vocab[word]] += 1
    return matrix

# Transform text
text_features = [count_transform(df[col], vocab) for col in text_cols]
X_text = np.hstack(text_features)

# Numeric features
X_num = df.drop(columns=text_cols + ['label', 'student_id'], errors='ignore').values
X_num = X_num.astype(float)
X_num = np.round(X_num * 10)

# Combine features
X = np.hstack([X_num, X_text])
y = df['label'].values

# Get unique student IDs
student_ids = df["student_id"].unique()
np.random.shuffle(student_ids)

n_students = len(student_ids)

# Split the data into 60% training, 20% validation and 20% test sets
train_end = int(0.6 * n_students)
val_end = int(0.8 * n_students)

train_students = student_ids[:train_end]
val_students = student_ids[train_end:val_end]
test_students = student_ids[val_end:]

# Boolean masks
train_mask = df["student_id"].isin(train_students)
val_mask   = df["student_id"].isin(val_students)
test_mask  = df["student_id"].isin(test_students)

# Apply masks
X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

# KNN model
class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_idx = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_idx]
            counts = np.bincount(nearest_labels)
            y_pred.append(np.argmax(counts))
        return np.array(y_pred)

# Hyperparameter tuning for best k (using train + validation sets)
train_results = []
val_results = []
best_k = None
best_val_acc = 0

# Find the best k_value based on the validation accuracy.
# Training accuracies evaluated to assess overfitting on each k value
for k in range(1, 20):
    knn = KNN(k=k)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    train_acc = np.mean(y_train_pred == y_train)

    y_val_pred = knn.predict(X_val)
    val_acc = np.mean(y_val_pred == y_val)

    train_results.append(train_acc)
    val_results.append(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_k = k

print(f"Best k value: {best_k}, Validation Accuracy: {best_val_acc:.4f}")

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(range(1, 20), train_results, marker='o', label="Training Accuracy")
plt.plot(range(1, 20), val_results, marker='o', label="Validation Accuracy")
plt.title("KNN Accuracy vs K Value")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()

# Train KNN value using best k value
knn = KNN(k=best_k)
knn.fit(X_train, y_train)

# Evaluate on all 3 splits
train_acc = np.mean(knn.predict(X_train) == y_train)
val_acc = np.mean(knn.predict(X_val) == y_val)
test_acc = np.mean(knn.predict(X_test) == y_test)

print("Train accuracy:", train_acc)
print("Validation accuracy:", val_acc)
print("Test accuracy:", test_acc)

# Show predicted class names (first 10 rows)
y_pred_test = knn.predict(X_test)
class_names = {v: k for k, v in label_map.items()}
predicted_names = [class_names[i] for i in y_pred_test]

print("Predicted classes (first 10):", predicted_names[:10])


# precision, recall, F1 per class
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred_test, average=None  # per class
)

print("Precision (per class):", precision)
print("Recall (per class):", recall)
print("F1 (per class):", f1)

# macro averages
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, y_pred_test, average='macro'
)

print("\nMacro Precision:", prec_macro)
print("Macro Recall:", rec_macro)
print("Macro F1:", f1_macro)


# Full breakdown
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=class_names.values()))


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (KNN)")
plt.show()

# Print confusion matrix
print("\nConfusion Matrix:")
print(cm)



