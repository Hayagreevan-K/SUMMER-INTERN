# Install required packages

# Import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torchvision import models, transforms, datasets
import numpy as np
import pandas as pd
import clip

import random
# Kaggle-friendly deterministic defaults and seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
from torch.nn.functional import cross_entropy
import json
import time

# Ensure GPU is selected in Kaggle notebook settings under Accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} at {time.strftime('%H:%M:%S %Z on %Y-%m-%d')}")

# Regularization Classes
class EWC(object):
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.fisher = {}
        self.params = {}

    def update(self, model, dataloader, device):
        model.eval()
        self.params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            out = model(x)
            loss = cross_entropy(out, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data.pow(2)
        return

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return self.lambda_ / 2 * loss

class MAS(object):
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.omega = {}
        self.params = {}

    def update(self, model, dataloader, device):
        model.eval()
        self.params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.omega = {n: torch.zeros_like(p) for n, p in self.params.items()}
        for x, _ in dataloader:
            x = x.to(device)
            model.zero_grad()
            out = model.feature_extractor(x).pow(2).mean(1)
            for o in out:
                o.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        self.omega[n] += p.grad.data.abs()
            break  # Single batch update to speed up (adjust as per paper if needed)
        return

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.omega:
                loss += (self.omega[n] * (p - self.params[n]).pow(2)).sum()
        return self.lambda_ / 2 * loss

class SI(object):
    def __init__(self, lambda_):
        self.lambda_ = lambda_
        self.W = {}
        self.p_old = {}
        self.epsilon = 0.0000001

    def init(self, model):
        self.p_old = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.W = {n: torch.zeros_like(p) for n, p in self.p_old.items()}
        self.theta = {n: torch.zeros_like(p) for n, p in self.p_old.items()}

    def update(self, model, grad):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.theta[n] = p.detach() - self.p_old[n]
                self.W[n] += grad[n] * self.theta[n]
                self.p_old[n] = p.detach().clone()

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.W:
                omega = self.W[n] / (self.theta[n].pow(2) + self.epsilon)
                loss += (omega * (p - self.p_old[n]).pow(2)).sum()
        return self.lambda_ * loss

class GEM(object):
    def __init__(self, memory_size_per_class, lambda_):
        self.memory_size_per_class = memory_size_per_class
        self.lambda_ = lambda_
        self.memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_memory(self, model, dataloader, current_classes):
        features = {c: [] for c in current_classes}
        images = {c: [] for c in current_classes}
        labels = {c: [] for c in current_classes}
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                feat = model.feature_extractor(x)
            for f, img, label in zip(feat, x, y):
                c = label.item()
                features[c].append(f.cpu().numpy())
                images[c].append(img)
                labels[c].append(c)
        exemplars_x = []
        exemplars_y = []
        for c in features:
            if len(features[c]) == 0:
                continue
            features[c] = np.array(features[c])
            images[c] = torch.stack(images[c])
            class_mean = features[c].mean(0)
            selected_idx = []
            current_mean = np.zeros_like(class_mean)
            for k in range(min(self.memory_size_per_class, len(features[c]))):
                dist = np.linalg.norm(current_mean + class_mean / (k + 1) - features[c], axis=1)
                idx = np.argmin(dist)
                selected_idx.append(idx)
                current_mean = (current_mean * k + features[c][idx]) / (k + 1)
                features[c] = np.delete(features[c], idx, axis=0)
            selected_images = images[c][selected_idx]
            exemplars_x.extend(selected_images)
            exemplars_y.extend([c] * len(selected_idx))
        if exemplars_x:
            self.memory.append((torch.stack(exemplars_x), torch.tensor(exemplars_y)))

    def compute_grads(self, model):
        grads = []
        for mem_x, mem_y in self.memory:
            mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
            model.zero_grad()
            out = model(mem_x)
            loss = cross_entropy(out, mem_y)
            loss.backward()
            task_grad = [p.grad.data.clone() for p in model.parameters() if p.requires_grad]
            grads.append(task_grad)
        return grads

    def project_grad(self, grad, prev_grads):
        for task_grads in prev_grads:
            for g, pg in zip(grad, task_grads):
                gv = g.view(-1)
                pgv = pg.view(-1)
                dot = torch.dot(gv, pgv)
                if dot < 0:
                    g -= (dot / (pgv.norm()**2)) * pg
        return grad

    def penalty(self, model, current_grad):
        if len(self.memory) == 0:
            return 0
        prev_grads = self.compute_grads(model)
        self.project_grad(current_grad, prev_grads)
        return 0

def main(rank: int, flags):
    # Setup
    torch.manual_seed(1993)  # Seed for ImageNet-100 subset as per paper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} at {time.strftime('%H:%M:%S %Z on %Y-%m-%d')}")

    # Work directly with input paths
    data_path = '/kaggle/input/imagenet100'
    train_subdirs = ['train.X1', 'train.X2', 'train.X3', 'train.X4']

    # Load class names from Labels.json
    try:
        with open(f'{data_path}/Labels.json', 'r') as f:
            labels = json.load(f)
            class_names = list(labels.values())  # Use descriptive names from Labels.json
    except FileNotFoundError:
        print(f"Error: Labels.json not found at {data_path}/Labels.json. Check path.")
        return
    num_classes = len(class_names)
    print(f"Found {num_classes} classes from Labels.json: {class_names[:5]}...")

    # ImageNet-100 transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets and compute cumulative targets
    train_datasets = []
    cumulative_targets = []
    for subdir in train_subdirs:
        train_path = f'{data_path}/{subdir}'
        if os.path.exists(train_path):
            try:
                dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
                train_datasets.append(dataset)
                cumulative_targets.extend(dataset.targets)
            except Exception as e:
                print(f"Error loading {train_path}: {e}")
        else:
            print(f"Warning: {train_path} not found, skipping.")
    if not train_datasets:
        print("Error: No training datasets loaded. Check paths.")
        return
    train_ds = ConcatDataset(train_datasets)

    # Use full dataset for oracle and subset for experiments if needed
    train_ds_subset = train_ds  # Full dataset for oracle; adjust subset_size if needed
    print(f"Using full dataset with {len(train_ds_subset)} training samples")

    test_ds = datasets.ImageFolder(root=f'{data_path}/val.X', transform=test_transform)
    if len(test_ds) == 0:
        print("Error: Validation dataset is empty. Check val.X path.")
        return
    print(f"Total validation samples: {len(test_ds)}")

    # For class-incremental learning (CIL) with 10 tasks of 10 classes each
    classes = list(range(num_classes))
    task_classes = [classes[i * 10:(i + 1) * 10] for i in range(10)]

    # Model
    class CLModel(nn.Module):
        def __init__(self, num_classes, use_lingo):
            super().__init__()
            self.feature_extractor = models.resnet18(weights=None)
            d = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
            self.classifier = nn.Linear(d, num_classes, bias=False)
            if use_lingo:
                clip_model, _ = clip.load("ViT-B/32", device=device)
                texts = clip.tokenize(class_names).to(device)
                with torch.no_grad():
                    text_features = clip_model.encode_text(texts).float()
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                self.classifier.weight.data = text_features
                self.classifier.weight.requires_grad = False

        def forward(self, x):
            features = self.feature_extractor(x)
            return self.classifier(features)

    # Oracle training
    print("Starting oracle training")
    oracle_model = CLModel(num_classes, use_lingo=False)
    oracle_model.to(device)
    optimizer = optim.SGD(oracle_model.parameters(), lr=0.1, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    for epoch in range(90):  # 90 epochs as per ImageNet-100
        oracle_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = oracle_model(x)
            loss = cross_entropy(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    oracle_model.eval()
    correct, total = 0, 0
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = oracle_model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    oracle_acc = correct / total * 100
    print(f"Oracle accuracy: {oracle_acc:.2f}%")

    # Run experiments
    results = []
    epochs = 90  # 90 epochs per task as per ImageNet-100
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for method in ['EWC', 'MAS', 'SI', 'LUCIR']:  # LUCIR as per paper baseline
            for use_lingo in [False, True]:
                model = CLModel(num_classes, use_lingo)
                model.to(device)
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
                scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

                if method == 'EWC':
                    reg = EWC(100)
                elif method == 'MAS':
                    reg = MAS(0.1)
                elif method == 'SI':
                    reg = SI(10)
                    reg.init(model)
                elif method == 'LUCIR':
                    reg = None  # LUCIR uses distillation, simplified here

                A = []  # A[i] = list of acc for task i after each subsequent task

                cumulative_classes = []
                for task_id in range(len(task_classes)):
                    cumulative_classes += task_classes[task_id]
                    current_classes = task_classes[task_id]
                    train_indices = [i for i in range(len(train_ds)) if cumulative_targets[i] in current_classes]
                    if not train_indices:
                        print(f"Warning: No samples for task {task_id + 1} (classes {current_classes}). Skipping.")
                        continue
                    train_subset = Subset(train_ds, train_indices)
                    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

                    for epoch in range(epochs):  # 90 epochs per task
                        model.train()
                        for x, y in train_loader:
                            x, y = x.to(device), y.to(device)
                            out = model(x)
                            loss = cross_entropy(out, y)
                            if method in ['EWC', 'MAS'] and task_id > 0:
                                loss += reg.penalty(model)
                            elif method == 'SI':
                                loss += reg.penalty(model)

                            optimizer.zero_grad()
                            loss.backward()
                            if method == 'SI':
                                saved_grad = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                            optimizer.step()
                            if method == 'SI':
                                reg.update(model, saved_grad)
                        scheduler.step()

                    if method in ['EWC', 'MAS']:
                        reg.update(model, train_loader, device)

                    model.eval()
                    for t in range(len(A), task_id + 1):
                        A.append([])
                    for t in range(task_id + 1):
                        test_indices = [i for i in range(len(test_ds)) if test_ds.targets[i] in task_classes[t]]
                        test_subset = Subset(test_ds, test_indices)
                        test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
                        correct, total = 0, 0
                        with torch.no_grad():
                            for x, y in test_loader:
                                x, y = x.to(device), y.to(device)
                                out = model(x)
                                pred = out.argmax(dim=1)
                                correct += (pred == y).sum().item()
                                total += y.size(0)
                        acc = correct / total if total > 0 else 0
                        A[t].append(acc)

                N = len(A)  # Number of completed tasks
                if N == 0:
                    last = 0
                    F = 0
                    avg = 0
                else:
                    last = sum(A[i][-1] for i in range(N)) / N * 100
                    F = sum(max(A[i]) - A[i][-1] for i in range(N-1)) / (N-1) * 100 if N > 1 else 0
                    avg = (1 / N) * sum(
                        (1 / (j + 1)) * sum(A[i][j - i] for i in range(j + 1))
                        for j in range(N)
                    ) * 100
                results.append({
                    'method': method,
                    'lingo': use_lingo,
                    'seed': seed,
                    'avg_acc': avg,
                    'last_acc': last,
                    'forgetting': F
                })
                print(f"Method {method}, Lingo {use_lingo}, Seed {seed}: Avg {avg:.2f}%, Last {last:.2f}%, F {F:.2f}%")

    df = pd.DataFrame(results)
    df.to_csv('/kaggle/working/results.csv', index=False)
    print("Results saved to /kaggle/working/results.csv")

if __name__ == '__main__':

    # Ensure output directory exists and is Kaggle-friendly
    args.output_dir = args.output_dir if hasattr(args, 'output_dir') else '/kaggle/working'
    os.makedirs(args.output_dir, exist_ok=True)
    flags = {
        'seed': 1993
    }
    main(0, flags)