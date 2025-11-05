# Install required packages

# Import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
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

# Ensure GPU is selected in Kaggle notebook settings under Accelerator

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
                    self.fisher[n] += p.grad.data.pow(2) / len(dataloader)
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
                o.backward(retain_graph=True)
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        self.omega[n] += p.grad.data.abs() / len(dataloader)
                model.zero_grad()
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

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.theta[n] = p.detach() - self.p_old[n]
                self.W[n] += p.grad.detach() * self.theta[n]
                self.p_old[n] = p.detach().clone()

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.W:
                omega = self.W[n] / (self.theta[n].pow(2) + self.epsilon)
                loss += (omega * (p - self.p_old[n]).pow(2)).sum()
        return self.lambda_ * loss

class GEM(object):
    def __init__(self, memory_size, lambda_):
        self.memory_size = memory_size
        self.lambda_ = lambda_
        self.memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_memory(self, dataloader):
        samples_x, samples_y = [], []
        for x, y in dataloader:
            samples_x.append(x)
            samples_y.append(y)
            if len(samples_x) * len(x) >= self.memory_size:
                break
        mem_x = torch.cat(samples_x[:self.memory_size // len(samples_x)], dim=0)
        mem_y = torch.cat(samples_y[:self.memory_size // len(samples_y)])
        self.memory.append((mem_x, mem_y))

    def compute_grads(self, model):
        grads = []
        for mem_x, mem_y in self.memory:
            for start in range(0, len(mem_x), 64):  # Reduced chunk size to 64 to further avoid OOM
                batch_x = mem_x[start:start+64].to(self.device)
                batch_y = mem_y[start:start+64].to(self.device)
                model.zero_grad()
                out = model(batch_x)
                loss = cross_entropy(out, batch_y)
                loss.backward()
                grads.append([p.grad.data.clone() for p in model.parameters() if p.requires_grad])
        return grads

    def project_grad(self, grad, prev_grads):
        projected_grad = []
        # Use the first set of previous gradients as a reference (simplification)
        if prev_grads and prev_grads[0]:  # Ensure prev_grads is not empty
            ref_grads = prev_grads[0]  # Take the first batch's gradients
            for g, pg in zip(grad, ref_grads):
                if torch.dot(g.view(-1), pg.view(-1)) < 0:
                    dot_product = torch.dot(g.view(-1), pg.view(-1))
                    norm_sq = torch.norm(pg.view(-1)) ** 2
                    g = g - (dot_product / norm_sq) * pg
                projected_grad.append(g)
        else:
            projected_grad = grad  # No projection if no previous gradients
        return projected_grad

    def penalty(self, model, current_grad):
        if len(self.memory) == 0:
            return 0
        prev_grads = self.compute_grads(model)
        projected_grad = self.project_grad(current_grad, prev_grads)
        for p, g in zip([p for p in model.parameters() if p.requires_grad], projected_grad):
            p.grad.data = g
        return 0

def main(rank: int, flags):
    # Setup
    torch.manual_seed(flags['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class names for OfficeHome (65 classes)
    class_names = [
        'Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Binder', 'Bottle', 'Bucket', 'Calculator', 'Calendar',
        'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign',
        'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle',
        'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven',
        'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler',
        'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Telephone', 'Toys',
        'Trash_Can', 'Webcam'
    ]
    num_classes = len(class_names)

    # Domains
    domains = ['Art', 'Clipart', 'Product', 'Real World']
    root = '/kaggle/input/officehome/OfficeHomeDataset_10072016'

    # Transform (ImageNet style for ResNet18)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and split datasets (80/20 train/test per class per domain)
    train_subsets = {}
    test_subsets = {}
    for domain in domains:
        ds = datasets.ImageFolder(os.path.join(root, domain), transform=transform)
        train_indices = []
        test_indices = []
        for l in range(num_classes):
            indices = [i for i in range(len(ds)) if ds.targets[i] == l]
            if len(indices) > 0:
                train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=flags['seed'])
                train_indices.extend(train_idx)
                test_indices.extend(test_idx)
        train_subsets[domain] = Subset(ds, train_indices)
        test_subsets[domain] = Subset(ds, test_indices)

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

    # Run experiments
    results = []

    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for method in ['EWC', 'MAS', 'SI', 'GEM']:
            for use_lingo in [False, True]:
                model = CLModel(num_classes, use_lingo)
                model.to(device)

                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
                scheduler = MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)

                if method == 'EWC':
                    reg = EWC(100)
                elif method == 'MAS':
                    reg = MAS(0.1)
                elif method == 'SI':
                    reg = SI(0.3)
                    reg.init(model)
                elif method == 'GEM':
                    reg = GEM(1300, 5)

                task_accs = []

                for task_id in range(len(domains)):
                    train_ds = train_subsets[domains[task_id]]
                    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
                    for epoch in tqdm(range(80), desc=f"Task {task_id+1}, Method {method}, Lingo {use_lingo}, Seed {seed}"):
                        model.train()
                        for x, y in train_loader:
                            x, y = x.to(device), y.to(device)
                            out = model(x)
                            loss = cross_entropy(out, y)
                            if method in ['EWC', 'MAS']:
                                if task_id > 0:
                                    loss += reg.penalty(model)
                            elif method == 'SI':
                                loss += reg.penalty(model)
                            elif method == 'GEM':
                                optimizer.zero_grad()
                                loss.backward()
                                current_grad = [p.grad.data.clone() for p in model.parameters() if p.requires_grad]
                                reg.penalty(model, current_grad)
                                for p, g in zip([p for p in model.parameters() if p.requires_grad], current_grad):
                                    p.grad.data = g
                                continue

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        scheduler.step()
                        if method == 'SI':
                            reg.update(model)

                    if method in ['EWC', 'MAS']:
                        reg_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
                        reg.update(model, reg_loader, device)
                    elif method == 'GEM':
                        reg.update_memory(train_loader)

                    model.eval()
                    task_acc = []
                    for t in range(task_id + 1):
                        test_loader = DataLoader(test_subsets[domains[t]], batch_size=128, shuffle=False)
                        correct, total = 0, 0
                        with torch.no_grad():
                            for x, y in test_loader:
                                x, y = x.to(device), y.to(device)
                                out = model(x)
                                pred = out.argmax(dim=1)
                                correct += (pred == y).sum().item()
                                total += y.size(0)
                        task_acc.append(correct / total if total > 0 else 0)
                    task_accs.append(task_acc)

                last_acc = np.mean(task_accs[-1])
                forgetting = np.mean([max(task_accs[j][i] for j in range(i, len(task_accs))) - task_accs[-1][i] for i in range(len(task_accs)-1)])
                results.append({
                    'method': method,
                    'lingo': use_lingo,
                    'seed': seed,
                    'last_acc': last_acc * 100,
                    'forgetting': forgetting * 100
                })

    df = pd.DataFrame(results)
    df.to_csv('/kaggle/working/results.csv', index=False)

if __name__ == '__main__':

    # Ensure output directory exists and is Kaggle-friendly
    args.output_dir = args.output_dir if hasattr(args, 'output_dir') else '/kaggle/working'
    os.makedirs(args.output_dir, exist_ok=True)
    flags = {'seed': 0}
    main(0, flags)