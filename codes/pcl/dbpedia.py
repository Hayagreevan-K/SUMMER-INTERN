

import os
import sys
import math
import random
import argparse
from pathlib import Path
import time

# try to import required packages; install if missing
def try_import(name, pkg=None):
    try:
        return __import__(name)
    except Exception as e:
        pkg = pkg or name
        print(f"Package {name} not found. Installing {pkg}...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return __import__(name)

datasets = try_import("datasets")
transformers = try_import("transformers")
torch = try_import("torch")
nn = torch.nn
optim = torch.optim
from torch.utils.data import DataLoader, Dataset

# deterministic seeds
import numpy as np
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Config / hyperparameters (kept conservative but editable)
BATCH_SIZE = 32
EPOCHS_PER_CLASS = 5   # PCL papers often use many epochs; adjust if needed
LR = 1e-3
WEIGHT_DECAY = 1e-4
HEAD_HIDDEN = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = '/kaggle/working' if os.path.exists("/kaggle") else "."

def prepare_tokenizer_and_model(model_name="bert-base-uncased"):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    encoder = transformers.AutoModel.from_pretrained(model_name)
    encoder.to(DEVICE)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return tokenizer, encoder

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def collate_batch(batch):
    keys = [k for k in batch[0].keys() if k!='label']
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    return out, labels

def extract_features(encoder, batch_inputs):
    # batch_inputs: dict of tokenized inputs on DEVICE
    with torch.no_grad():
        outputs = encoder(**batch_inputs)
        # outputs.last_hidden_state (bs, seq, hidden); use pooled output if available
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        else:
            # fallback to mean pooling
            feats = outputs.last_hidden_state.mean(dim=1)
    return feats

class Head(nn.Module):
    def __init__(self, in_dim, hidden=HEAD_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)  # (bs,)

def train_one_class(head, optimizer, pos_loader, encoder, l2_lambda=WEIGHT_DECAY, epochs=EPOCHS_PER_CLASS):
    head.train()
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        for batch_inputs, labels in pos_loader:
            # labels are all the target class (1)
            # move to device and extract features
            batch_inputs = {k: v.to(DEVICE) for k, v in batch_inputs.items()}
            feats = extract_features(encoder, batch_inputs)
            feats.requires_grad = False  # features frozen; we won't calculate input grads
            logits = head(feats)
            targets = torch.ones_like(logits, device=DEVICE)
            loss = criterion(logits, targets)
            # L2 regularization on head parameters as proxy for H-reg
            l2 = 0.0
            for p in head.parameters():
                l2 = l2 + p.pow(2).sum()
            loss = loss + l2_lambda * l2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * logits.size(0)
            n += logits.size(0)
        if n>0:
            print(f"  Epoch {epoch+1}/{epochs} - avg loss: {epoch_loss/n:.4f}")
    return head

def evaluate_heads(heads, encoder, dataloader):
    encoder.eval()
    for h in heads:
        h.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_inputs, labels in dataloader:
            batch_inputs = {k: v.to(DEVICE) for k, v in batch_inputs.items()}
            feats = extract_features(encoder, batch_inputs)
            # collect logits for each head
            logits_list = []
            for h in heads:
                logits_list.append(h(feats).unsqueeze(1))  # (bs,1)
            logits = torch.cat(logits_list, dim=1)  # (bs, num_classes)
            preds = logits.argmax(dim=1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)
    acc = correct/total if total>0 else 0.0
    return acc

def main():
    print("Loading DBPedia dataset...")
    ds = datasets.load_dataset("dbpedia_14")
    train = ds["train"]
    test = ds["test"]
    # labels are 0..13 for 14 classes
    num_classes = len(set(train["label"]))
    print(f"Num classes: {num_classes}")

    tokenizer, encoder = prepare_tokenizer_and_model()
    # tokenize texts
    def tokenize_batch(examples):
        return tokenizer(examples["content"], truncation=True, padding="max_length", max_length=128)
    train_tok = train.map(tokenize_batch, batched=True, remove_columns=["content"])
    test_tok = test.map(tokenize_batch, batched=True, remove_columns=["content"])

    # build datasets
    train_encodings = {k: train_tok[k] for k in ["input_ids","attention_mask"]}
    test_encodings = {k: test_tok[k] for k in ["input_ids","attention_mask"]}
    train_labels = train_tok["label"]
    test_labels = test_tok["label"]
    train_dataset = TextDataset(train_encodings, train_labels)
    test_dataset = TextDataset(test_encodings, test_labels)

    # DataLoaders
    train_loader_full = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Prepare per-class indices
    class_indices = {c: [] for c in range(num_classes)}
    for i, lbl in enumerate(train_labels):
        class_indices[lbl].append(i)

    # Initialize heads (one per class)
    # get feature dim by extracting one batch
    sample_batch = next(iter(train_loader_full))[0]
    sample_batch = {k: v.to(DEVICE) for k, v in sample_batch.items()}
    feat_sample = extract_features(encoder, sample_batch).cpu()
    feat_dim = feat_sample.size(1)
    print(f"Feature dim: {feat_dim}")

    heads = [Head(feat_dim).to(DEVICE) for _ in range(num_classes)]
    optimizers = [optim.Adam(h.parameters(), lr=LR) for h in heads]

    # Training: for each class, train its head on positive samples only
    print("Starting per-class training...")
    for c in range(num_classes):
        print(f" Training head for class {c}  (num positives = {len(class_indices[c])})")
        pos_indices = class_indices[c]
        if len(pos_indices)==0:
            print("  No positives for this class in train set; skipping")
            continue
        # build Subset and DataLoader for positive samples
        from torch.utils.data import Subset
        pos_ds = Subset(train_dataset, pos_indices)
        pos_loader = DataLoader(pos_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        heads[c] = train_one_class(heads[c], optimizers[c], pos_loader, encoder, l2_lambda=WEIGHT_DECAY, epochs=EPOCHS_PER_CLASS)
        # save head after training
        torch.save(heads[c].state_dict(), os.path.join(OUTPUT_DIR, f"head_class_{c}.pth"))

    # Evaluate
    print("Evaluating combined model (argmax over heads)...")
    acc = evaluate_heads(heads, encoder, test_loader)
    print(f"Test accuracy (argmax over heads): {acc*100:.2f}%")

    # Save tokenizer and encoder (encoder not fine-tuned so saving is optional)
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "dbpedia_tokenizer"))
    # Save heads collectively
    for idx, h in enumerate(heads):
        torch.save(h.state_dict(), os.path.join(OUTPUT_DIR, f"dbpedia_head_{idx}.pth"))

if __name__ == "__main__":

    # Ensure output directory exists and is Kaggle-friendly
    args.output_dir = args.output_dir if hasattr(args, 'output_dir') else '/kaggle/working'
    os.makedirs(args.output_dir, exist_ok=True)
    main()
