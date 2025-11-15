import pandas as pd
import numpy as np

# Loading a file
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

# Converting text to bag of words
text_cols = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
]

# Build vocabulary
def build_vocab(text_series, max_words=100):
    words = " ".join(text_series.fillna("")).lower().split()
    freq = pd.Series(words).value_counts()
    return freq.index[:max_words].tolist()

vocab = []
for col in text_cols:
    vocab += build_vocab(df[col], max_words=100)
vocab = list(set(vocab))  # unique words

# Convert text to bag-of-words
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

# Combining all features
X_num = df.drop(columns=['student_id', 'label']).values
# Normalize numeric features
X_num = (X_num - X_num.mean(axis=0)) / (X_num.std(axis=0) + 1e-8)
# Normalize text features
X_text = (X_text - X_text.mean(axis=0)) / (X_text.std(axis=0) + 1e-8)

# Combine
X = np.hstack([X_num, X_text])

# Preparation of labels
label_map = {label: i for i, label in enumerate(df['label'].dropna().unique())}
y = df['label'].map(label_map).values
n_samples, n_features = X.shape
n_classes = len(label_map)

# Make a Train and Test data
np.random.seed(42)
indices = np.arange(n_samples)
np.random.shuffle(indices)

split = int(0.7 * n_samples)
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# One-hot encoding and class weights (inverse frequency)
Y_train = np.zeros((len(y_train), n_classes))
Y_train[np.arange(len(y_train)), y_train] = 1

class_counts = np.sum(Y_train, axis=0)
class_weights = len(y_train) / (n_classes * class_counts)

# Regular softmax classifier with gradient descent
class SoftmaxRegular:
    def __init__(self, learning_rate=0.00025, n_epochs=800, momentum=0.9, lambda_reg=0.005):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.momentum = momentum
        self.lambda_reg = lambda_reg

        self.W = None
        self.b = None

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def fit(self, X_train, Y_train, class_weights):
        n_samples, n_features = X_train.shape
        n_classes = Y_train.shape[1]

        np.random.seed(42)
        self.W = 0.01 * np.random.randn(n_features, n_classes)
        self.b = np.zeros(n_classes)

        vW = np.zeros_like(self.W)
        vb = np.zeros_like(self.b)

        for epoch in range(self.n_epochs):
            logits = X_train @ self.W + self.b
            probs = self.softmax(logits)

            # Weighted cross-entropy loss + L2 regularization
            loss = -np.mean(np.sum(Y_train * np.log(probs + 1e-12) * class_weights, axis=1))
            loss += self.lambda_reg * np.sum(self.W ** 2)

            # Gradients
            grad_logits = (probs - Y_train) * class_weights / len(Y_train)
            dW = X_train.T @ grad_logits + 2 * self.lambda_reg * self.W
            db = np.sum(grad_logits, axis=0)

            # Momentum updates
            vW = self.momentum * vW - self.learning_rate * dW
            vb = self.momentum * vb - self.learning_rate * db
            self.W += vW
            self.b += vb

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        logits = X @ self.W + self.b
        probs = self.softmax(logits)
        y_pred = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)
        return y_pred, confidence, probs



# Training, prediction and evaluation
model = SoftmaxRegular(
    learning_rate=0.00025,
    n_epochs=800,
    momentum=0.9,
    lambda_reg=0.005
)

model.fit(X_train, Y_train, class_weights)

y_train_pred, _, _ = model.predict(X_train)
y_test_pred, confidence, probs = model.predict(X_test)

class_names = {v: k for k, v in label_map.items()}

train_acc = np.mean(y_train_pred == y_train)
test_acc = np.mean(y_test_pred == y_test)

print("\nPredicted class names (first 10):", [class_names[i] for i in y_test_pred[:10]])
print("Prediction confidence (first 10):", confidence[:10])
print(f"Training accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
