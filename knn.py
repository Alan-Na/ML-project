import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Multi-select columns into one-hot encoding
multi_select_cols = [
    "Which types of tasks do you feel this model handles best? (Select all that apply.)",
    "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
]

for col in multi_select_cols:
    dummies = df[col].str.get_dummies(sep=",")
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[col], inplace=True)

# Encode target label
label_map = {label: i for i, label in enumerate(df['label'].dropna().unique())}
df['label'] = df['label'].map(label_map)

# Processing of text survey type questions
text_cols = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
]

# Each unique word gets a unique index
def build_vocab(text_series):
    vocab = {}
    idx = 0
    for text in text_series.fillna(""):
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

# Combine vocab from all text columns
vocab = {}
for col in text_cols:
    col_vocab = build_vocab(df[col])
    for w in col_vocab:
        if w not in vocab:
            vocab[w] = len(vocab)

# Creat a new matrix in which [i, j] is the element that counts how many
# times the word j appear in text i
def count_transform(text_series, vocab):
    N = len(text_series)
    V = len(vocab)
    matrix = np.zeros((N, V))
    for i, text in enumerate(text_series.fillna("")):
        for word in text.lower().split():
            if word in vocab:
                matrix[i, vocab[word]] += 1
    return matrix

# Transform text columns
text_features = [count_transform(df[col], vocab) for col in text_cols]
X_text = np.hstack(text_features)

# Combining all features together
X_num = df.drop(columns=text_cols + ['label', 'student_id'], errors='ignore').values
X_num = X_num.astype(float)
X_num = np.round(X_num * 10)  # scale numeric features

X = np.hstack([X_num, X_text])
y = df['label'].values

# Make a Train and Test data
n_samples = X.shape[0]
indices = np.arange(n_samples)
np.random.shuffle(indices)

train_idx = indices[:int(0.7*n_samples)]
test_idx = indices[int(0.7*n_samples):]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# KNN model that uses Euclidean distances
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

# Training and evalua
knn = KNN(k=9)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print("Test accuracy:", accuracy)

# Print first 10 predicted class names
class_names = {v:k for k,v in label_map.items()}
predicted_names = [class_names[i] for i in y_pred]
print("Predicted classes (first 10):", predicted_names[:10])



#The code that tries to find best k value for the model

# results = []  # store tuples (k, accuracy)
#
# for k in range(1, 21):  # try k = 1 to 20
#     knn = KNN(k=k)
#     knn.fit(X_train, y_train)
#     y_pred = knn.predict(X_test)
#     acc = np.mean(y_pred == y_test)
#     results.append((k, acc))

# # Convert to pandas DataFrame
# df_results = pd.DataFrame(results, columns=["k", "accuracy"])
#
# # Plot using pandas built-in plot
# df_results.plot(x="k", y="accuracy", marker="o", legend=False, figsize=(8, 5))
# plt.title("KNN Accuracy vs K Value")
# plt.xlabel("K Value")
# plt.ylabel("Accuracy")
# plt.grid(True)
# plt.show()
