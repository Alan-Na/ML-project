import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------- 1. Load CSV -----------------------------
df = pd.read_csv("training_data_clean.csv")

# ----------------------------- 2. Preprocessing -----------------------------
# Likert-scale columns
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

# Multi-select columns -> one-hot
multi_select_cols = [
    "Which types of tasks do you feel this model handles best? (Select all that apply.)",
    "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"
]

for col in multi_select_cols:
    if col in df.columns:
        dummies = df[col].str.get_dummies(sep=",")
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)

# Text -> bag-of-words
text_cols = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?"
]

def build_vocab(text_series, max_words=200):
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

X_text /= (X_text.max(axis=0) + 1e-8)
df = df.drop(columns=text_cols)

# Encode labels
labels = df['label'].fillna("Unknown").values
classes = np.unique(labels)
num_classes = len(classes)
label_map = {c: i for i, c in enumerate(classes)}
y = np.array([label_map[l] for l in labels])

# Drop non-feature columns
if 'student_id' in df.columns:
    df = df.drop(columns=['student_id'])
df = df.fillna(0)

# Combine numeric + text
X_numeric = df.drop(columns=['label']).values
X_numeric = (X_numeric - X_numeric.mean(axis=0)) / (X_numeric.std(axis=0) + 1e-8)
X = np.hstack([X_numeric, X_text])

# ----------------------------- 3. Train/Test split -----------------------------
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split = int(0.7 * X.shape[0])
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ----------------------------- 4. PyTorch Model -----------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, output_dim=num_classes, dropout=0.3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ----------------------------- 5. Hyperparameter search -----------------------------
hidden_options = [(64, 32), (128, 64), (32, 16)]
lr_options = [0.01, 0.005, 0.001]
dropout_options = [0.2, 0.3, 0.4]

best_acc = 0
best_config = {}
best_model_state = None

for hidden1, hidden2 in hidden_options:
    for lr in lr_options:
        for dropout in dropout_options:
            model = SimpleNN(X_train.shape[1], hidden1, hidden2, num_classes, dropout)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

            # Early stopping parameters
            patience = 20
            wait = 0
            epochs = 200
            best_val_acc = 0
            best_state = None

            for epoch in range(1, epochs+1):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()

                # Evaluate
                model.eval()
                with torch.no_grad():
                    preds = torch.argmax(model(X_test_t), dim=1)
                    acc = (preds == y_test_t).float().mean().item()

                if acc > best_val_acc:
                    best_val_acc = acc
                    best_state = model.state_dict()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            if best_val_acc > best_acc:
                best_acc = best_val_acc
                best_config = {'hidden1': hidden1, 'hidden2': hidden2, 'lr': lr, 'dropout': dropout}
                best_model_state = best_state

print("Best Test Accuracy:", best_acc)
print("Best Hyperparameters:", best_config)

# ----------------------------- 6. Save best model and metadata -----------------------------
torch.save(best_model_state, "best_model.pth")
np.save("vocab.npy", np.array(vocab, dtype=object))
np.save("label_map.npy", np.array([label_map[c] for c in classes]))
print("Best model, vocab, and label map saved.")
