# # Customer Support Intelligence Platform — Phase 5: BiLSTM Deep Learning Model
# 
# **Brief requirements covered (Day 11):**
# - Build a PyTorch BiLSTM model using **pre-trained GloVe embeddings (100d)**
#   on Ticket Description text
# - Train for Ticket Type multi-class classification
# - Apply BatchNorm and Dropout
# - Apply early stopping
# - Compare against classical ML baseline
# 
# **Using GloVe here specifically (not Word2Vec):** Day 5-6 used self-trained
# Word2Vec as the brief's allowed alternative for that step. This step
# specifically calls for GloVe, so we use genuine pretrained embeddings this
# time - downloaded via gensim's downloader API (glove-wiki-gigaword-100),
# which fetches the same Stanford GloVe vectors without needing a manual
# 822MB zip download.
# 
# **Expectation, set honestly from prior results:** every classical model
# (6 total, across 2 notebooks) converged to near-baseline performance,
# confirming a weak-signal dataset. A BiLSTM might extract marginally
# different signal through sequential processing, but a dramatic jump would
# be surprising - we report whatever we actually find.

import sys
sys.path.insert(0, "../src")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gensim.downloader as gensim_api
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from collections import Counter

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_df = pd.read_csv("../data_processed/train.csv")
val_df = pd.read_csv("../data_processed/val.csv")
test_df = pd.read_csv("../data_processed/test.csv")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ## 1. Download pretrained GloVe 100d embeddings
# 
# **Why this specific source:** `glove-wiki-gigaword-100` via gensim's
# downloader is the same Stanford GloVe vectors (trained on Wikipedia +
# Gigaword), pre-converted to a format gensim can load directly - avoids
# manually downloading and parsing Stanford's raw 822MB multi-dimension zip
# file. First run downloads (~128MB, cached afterward).

print("Downloading GloVe 100d embeddings (cached after first run)...")
glove_vectors = gensim_api.load("glove-wiki-gigaword-100")
print(f"GloVe vocabulary size: {len(glove_vectors)}")
print(f"Embedding dimension: {glove_vectors.vector_size}")

# ## 2. Build vocabulary and embedding matrix
# 
# Vocabulary built from our OWN training data's cleaned text - only words that
# actually appear in our tickets get a slot. For each vocabulary word, we look
# up its GloVe vector if available; words GloVe doesn't know (rare/misspelled
# words) get a random initialization instead of failing.

MAX_VOCAB_SIZE = 8000
MAX_SEQ_LENGTH = 100  # truncate/pad descriptions to this many tokens
EMBEDDING_DIM = 100

all_train_tokens = []
for text in train_df["Description_Clean"].fillna(""):
    all_train_tokens.extend(text.split())

word_counts = Counter(all_train_tokens)
most_common = word_counts.most_common(MAX_VOCAB_SIZE - 2)  # reserve 2 slots for PAD/UNK

word_to_idx = {"<PAD>": 0, "<UNK>": 1}
for word, _ in most_common:
    word_to_idx[word] = len(word_to_idx)

vocab_size = len(word_to_idx)
print(f"Vocabulary size (from training data): {vocab_size}")

embedding_matrix = np.random.normal(0, 0.1, (vocab_size, EMBEDDING_DIM)).astype(np.float32)
embedding_matrix[0] = np.zeros(EMBEDDING_DIM)  # PAD token = all zeros

found_in_glove = 0
for word, idx in word_to_idx.items():
    if word in glove_vectors:
        embedding_matrix[idx] = glove_vectors[word]
        found_in_glove += 1

print(f"Words found in GloVe: {found_in_glove}/{vocab_size} ({100*found_in_glove/vocab_size:.1f}%)")

# ## 3. PyTorch Dataset and text-to-sequence conversion

def text_to_sequence(text, word_to_idx, max_len):
    tokens = text.split() if isinstance(text, str) else []
    seq = [word_to_idx.get(tok, word_to_idx["<UNK>"]) for tok in tokens[:max_len]]
    seq = seq + [word_to_idx["<PAD>"]] * (max_len - len(seq))
    return seq

type_label_map = {label: i for i, label in enumerate(sorted(train_df["Ticket Type"].unique()))}
NUM_CLASSES = len(type_label_map)

class TicketDataset(Dataset):
    def __init__(self, df, word_to_idx, max_len, label_map):
        self.sequences = [text_to_sequence(t, word_to_idx, max_len) for t in df["Description_Clean"]]
        self.labels = [label_map[l] for l in df["Ticket Type"]]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

train_dataset = TicketDataset(train_df, word_to_idx, MAX_SEQ_LENGTH, type_label_map)
val_dataset = TicketDataset(val_df, word_to_idx, MAX_SEQ_LENGTH, type_label_map)
test_dataset = TicketDataset(test_df, word_to_idx, MAX_SEQ_LENGTH, type_label_map)

BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ## 4. BiLSTM model architecture
# 
# **Matches brief spec exactly:** GloVe embedding layer (initialized with our
# matrix, fine-tuned during training) -> Bidirectional LSTM -> BatchNorm ->
# Dropout -> Linear classification head.

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=64, num_classes=5, dropout=0.5):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix), freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Concatenate final forward and backward hidden states
        final_hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        normalized = self.batch_norm(final_hidden)
        dropped = self.dropout(normalized)
        return self.fc(dropped)

model = BiLSTMClassifier(embedding_matrix, hidden_dim=64, num_classes=NUM_CLASSES, dropout=0.5).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(model)

# ## 5. Training loop with early stopping
# 
# Early stopping: halts training if validation loss doesn't improve for
# 5 consecutive epochs, keeping the best checkpoint - same pattern used
# successfully in the SmartVision AI project's later, more careful attempts.

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

MAX_EPOCHS = 30
PATIENCE = 5

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * sequences.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

import copy
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_loss = float("inf")
best_model_state = copy.deepcopy(model.state_dict())
epochs_without_improvement = 0

for epoch in range(MAX_EPOCHS):
    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        marker = " <- best so far"
    else:
        epochs_without_improvement += 1
        marker = f" ({epochs_without_improvement}/{PATIENCE} epochs without improvement)"

    print(f"Epoch {epoch+1:2d}/{MAX_EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{marker}")

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch+1}.")
        break

model.load_state_dict(best_model_state)
print(f"\nLoaded best model (val_loss={best_val_loss:.4f})")

# ## 6. Evaluation on test set

type_label_map_inv = {v: k for k, v in type_label_map.items()}

model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        outputs = model(sequences)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
test_f1_macro = f1_score(all_labels, all_preds, average="macro")
try:
    test_roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
except ValueError:
    test_roc_auc = None

print(f"BiLSTM Test Accuracy: {test_acc:.4f}")
print(f"BiLSTM Test F1-macro: {test_f1_macro:.4f}")
print(f"BiLSTM Test ROC-AUC:  {test_roc_auc:.4f}" if test_roc_auc else "ROC-AUC: N/A")
print()
label_names = [type_label_map_inv[i] for i in range(NUM_CLASSES)]
print(classification_report(all_labels, all_preds, target_names=label_names))

# ## 7. Comparison against all prior models

print("FULL COMPARISON: Ticket Type Classification Across All Models")
print("=" * 65)
print(f"{'Model':<40s} {'Accuracy':>10s}")
print("-" * 51)
print(f"{'Logistic Regression (baseline)':<40s} {0.2071:>10.4f}")
print(f"{'Naive Bayes (baseline)':<40s} {0.2134:>10.4f}")
print(f"{'Random Forest (tabular+TFIDF)':<40s} {0.1850:>10.4f}")
print(f"{'XGBoost (Optuna-tuned)':<40s} {0.2102:>10.4f}")
print(f"{'BiLSTM (GloVe embeddings)':<40s} {test_acc:>10.4f}")
print()
print("Random guessing baseline for 5 classes: 0.2000")
print()
if test_acc > 0.23:
    print("BiLSTM shows a meaningful improvement over classical models -")
    print("sequential processing may be extracting signal the others missed.")
else:
    print("BiLSTM confirms the same weak-signal pattern as every classical")
    print("model - consistent, reproducible evidence across 7 different")
    print("model architectures now (6 classical + this deep learning model).")

# ## 8. Save model and log to MLflow

mlflow.set_tracking_uri("../mlruns")
mlflow.set_experiment("customer_support_deep_learning")

with mlflow.start_run(run_name="bilstm_glove_ticket_type"):
    mlflow.log_param("model_type", "BiLSTM")
    mlflow.log_param("embedding", "GloVe-100d (glove-wiki-gigaword-100)")
    mlflow.log_param("hidden_dim", 64)
    mlflow.log_param("dropout", 0.5)
    mlflow.log_param("max_seq_length", MAX_SEQ_LENGTH)
    mlflow.log_param("vocab_size", vocab_size)
    mlflow.log_param("epochs_trained", len(history["train_loss"]))
    mlflow.log_param("early_stopping_patience", PATIENCE)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1_macro", test_f1_macro)
    if test_roc_auc:
        mlflow.log_metric("test_roc_auc", test_roc_auc)
    mlflow.pytorch.log_model(model, "model")

torch.save(model.state_dict(), "../models/bilstm_glove_model.pth")
import json
with open("../models/bilstm_vocab.json", "w") as f:
    json.dump(word_to_idx, f)

print("Saved model to models/bilstm_glove_model.pth")
print("Saved vocabulary to models/bilstm_vocab.json")
print("Logged run to MLflow experiment 'customer_support_deep_learning'")

