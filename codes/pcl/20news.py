import os
import sys
import torch

import random
# Kaggle-friendly deterministic defaults and seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, Subset
import numpy as np
import argparse
import psutil
import csv
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import gc
from tqdm import tqdm

class News20Dataset(Dataset):
    def __init__(self, subset='train', transform=None):
        data = fetch_20newsgroups(subset=subset, categories=None, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
        self.texts = data.data
        self.labels = data.target
        self.transform = transform
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.features = self.vectorizer.fit_transform(self.texts).toarray()
        self.features = torch.tensor(self.features, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        return feature, label

class PCLModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(PCLModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(10000, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.feature_dim = 128
        for param in self.feature_extractor[:-2].parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout_rate)
        self.class_heads = nn.ModuleDict()

    def add_head(self, class_idx, device):
        new_head = nn.Linear(self.feature_dim, 1).to(device)
        if len(self.class_heads) > 0:
            existing_head = next(iter(self.class_heads.values()))
            new_head.load_state_dict(existing_head.state_dict())
        self.class_heads[str(class_idx)] = new_head

    def forward(self, x, class_idx=None):
        features = self.feature_extractor(x)
        features = self.dropout(features)
        if class_idx is not None:
            return self.class_heads[str(class_idx)](features)
        return {k: v(features) for k, v in self.class_heads.items()}

class EWC(object):
    def __init__(self, model, dataset, device, batch_size=128, num_batches=10):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        try:
            self._precision_matrices = self._diag_fisher()
            for n, p in self.params.items():
                self._means[n] = p.clone().detach()
            print(f"Memory usage after EWC init: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Error in EWC initialization: {e}")
            raise

    def _diag_fisher(self):
        precision_matrices = {n: torch.zeros_like(p).to(self.device) for n, p in self.params.items()}
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        try:
            for i, batch in enumerate(tqdm(dataloader, desc="Computing Fisher Matrix", leave=False)):
                if i >= self.num_batches:
                    break
                self.model.zero_grad()
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(inputs, class_idx=targets[0].item())
                loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), torch.ones_like(outputs.squeeze()))
                loss.backward()
                for n, p in self.model.named_parameters():
                    if p.grad is not None and n in self.params:
                        precision_matrices[n] += p.grad.data ** 2 / len(dataloader)
            return precision_matrices
        except Exception as e:
            print(f"Error in Fisher matrix computation: {e}")
            raise

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._means:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

def train_pcl_ewc(rank, args):
    torch.manual_seed(args['seed'] + rank)
    np.random.seed(args['seed'] + rank)
    device = torch.device('cpu')
    if rank == 0:
        print(f"Rank {rank} using device: {device}")
        print(f"Initial memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    trainset = News20Dataset(subset='train', transform=None)
    testset = News20Dataset(subset='test', transform=None)
    if args['debug']:
        train_indices = np.random.choice(len(trainset), len(trainset) // 10, replace=False)
        test_indices = np.random.choice(len(testset), len(testset) // 10, replace=False)
        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)
    if rank == 0:
        print(f"Memory usage after dataset loading: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        gc.collect()

    num_classes = 20
    classes_per_task = 2
    tasks = [(i, min(i + classes_per_task - 1, num_classes - 1)) for i in range(0, num_classes, classes_per_task)]
    tasks = [tuple(range(start, end + 1)) if start != end else (start,) for start, end in tasks]
    if rank == 0:
        print(f"20NEWS: {num_classes} classes, {len(tasks)} tasks, classes per task: {tasks}")

    model = PCLModel(dropout_rate=args['dropoutrate']).to(device)
    if rank == 0:
        print(f"Memory usage after model init: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        gc.collect()

    csv_file = "20news_pcl_ewc_results.csv"
    if rank == 0:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Epoch', 'Task_Accuracy', 'All_Classes_Accuracy'] + [f'Previous_Task_{i}_Accuracy' for i in range(1, len(tasks)+1)])

    ewc_list = []
    for task_idx, task_classes in enumerate(tasks, 1):
        try:
            if rank == 0:
                print(f"Starting training for Task {task_idx}/{len(tasks)} in 20NEWS")
            for class_idx in task_classes:
                model.add_head(class_idx, device)
            if rank == 0:
                print(f"Added heads for classes {task_classes}")

            train_indices = [i for i, item in enumerate(trainset) if item[1] in task_classes]
            test_indices = [i for i, item in enumerate(testset) if item[1] in task_classes]
            if not train_indices or not test_indices:
                raise ValueError(f"No samples found for task classes {task_classes} in Task {task_idx}")
            task_trainset = Subset(trainset, train_indices)
            task_testset = Subset(testset, test_indices)
            if rank == 0:
                print(f"Task {task_idx}/{len(tasks)} in 20NEWS: {len(task_trainset)} train samples, {len(task_testset)} test samples")
                if len(task_trainset) == 0 or len(task_testset) == 0:
                    raise ValueError(f"Empty dataset for Task {task_idx}")

            train_loader = torch.utils.data.DataLoader(task_trainset, batch_size=args['batch_size'], shuffle=True, num_workers=2, prefetch_factor=2)
            test_loader = torch.utils.data.DataLoader(task_testset, batch_size=args['batch_size'], shuffle=False, num_workers=2, prefetch_factor=2)
            if rank == 0:
                print(f"Memory usage before training Task {task_idx}: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
                gc.collect()

            optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args['lr'], momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])

            for epoch in range(args['epochs']):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                try:
                    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Task {task_idx}, Epoch {epoch + 1}", leave=False), 1):
                        inputs, targets = batch[0].to(device), batch[1].to(device)
                        optimizer.zero_grad()
                        loss = 0
                        for class_idx in task_classes:
                            outputs = model(inputs, class_idx=class_idx)
                            class_targets = (targets == class_idx).float()
                            nll = F.binary_cross_entropy_with_logits(outputs.squeeze(), class_targets)
                            h_reg = sum(torch.norm(p, p=2) ** 2 for p in model.class_heads[str(class_idx)].parameters())
                            loss += (nll + args['lambda_h'] * h_reg) / len(task_classes)
                        if task_idx > 1:
                            for ewc in ewc_list:
                                loss += args['lambda_ewc'] * ewc.penalty(model)
                        if torch.isnan(loss) or torch.isinf(loss):
                            raise ValueError(f"Loss is NaN or Inf at Task {task_idx}, Epoch {epoch + 1}, Batch {batch_idx}")
                        if loss.item() > 100:
                            print(f"Warning: High loss {loss.item():.4f} at Task {task_idx}, Epoch {epoch + 1}, Batch {batch_idx}")
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        optimizer.step()

                        running_loss += loss.item()
                        outputs = torch.stack([model(inputs, class_idx=c).squeeze() for c in task_classes], dim=1)
                        outputs = torch.sigmoid(outputs)
                        _, predicted = torch.max(outputs, 1)
                        predicted = torch.tensor([task_classes[p] for p in predicted], device=device)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                        if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                            if rank == 0:
                                print(f"Task {task_idx}, Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, "
                                      f"Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}, Memory: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
                                gc.collect()

                    epoch_loss = running_loss / len(train_loader)
                    epoch_acc = 100. * correct / total
                    if rank == 0:
                        print(f"Task {task_idx}, Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
                    scheduler.step()

                except Exception as e:
                    if rank == 0:
                        print(f"Error in Task {task_idx}, Epoch {epoch + 1}, Batch {batch_idx}: {e}")
                    raise

            try:
                ewc = EWC(model, task_trainset, device, batch_size=args['batch_size'], num_batches=10)
                ewc_list.append(ewc)
            except Exception as e:
                if rank == 0:
                    print(f"Error in EWC for Task {task_idx}: {e}")
                raise

            model.eval()
            correct = 0
            total = 0
            try:
                with torch.no_grad():
                    for batch in tqdm(test_loader, desc=f"Testing Task {task_idx}", leave=False):
                        inputs, targets = batch[0].to(device), batch[1].to(device)
                        outputs = torch.stack([model(inputs, class_idx=c).squeeze() for c in task_classes], dim=1)
                        outputs = torch.sigmoid(outputs)
                        _, predicted = torch.max(outputs, 1)
                        predicted = torch.tensor([task_classes[p] for p in predicted], device=device)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                test_acc = 100. * correct / total
                if rank == 0:
                    print(f"20NEWS PCL Task {task_idx}/{len(tasks)}, Test Accuracy: {test_acc:.2f}%")
            except Exception as e:
                if rank == 0:
                    print(f"Error in testing Task {task_idx}: {e}")
                raise

            all_test_acc = 0.0
            if rank == 0:
                print(f"Evaluating all classes up to Task {task_idx}")
                all_test_indices = [i for i, item in enumerate(testset) if item[1] in [c for t in tasks[:task_idx] for c in t]]
                all_task_testset = Subset(testset, all_test_indices)
                all_test_loader = torch.utils.data.DataLoader(all_task_testset, batch_size=args['batch_size'], shuffle=False, num_workers=2, prefetch_factor=2)
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in tqdm(all_test_loader, desc=f"Testing All Classes Task {task_idx}", leave=False):
                        inputs, targets = batch[0].to(device), batch[1].to(device)
                        all_classes = [c for t in tasks[:task_idx] for c in t]
                        outputs = torch.stack([model(inputs, class_idx=c).squeeze() for c in all_classes], dim=1)
                        outputs = torch.sigmoid(outputs)
                        _, predicted = torch.max(outputs, 1)
                        predicted = torch.tensor([all_classes[p] for p in predicted], device=device)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                all_test_acc = 100. * correct / total
                print(f"20NEWS PCL Task {task_idx}/{len(tasks)}, All Classes Accuracy: {all_test_acc:.2f}%")

            prev_task_accuracies = {}
            for prev_task_idx, prev_task_classes in enumerate(tasks[:task_idx], 1):
                try:
                    prev_test_indices = [i for i, item in enumerate(testset) if item[1] in prev_task_classes]
                    prev_task_testset = Subset(testset, prev_test_indices)
                    prev_test_loader = torch.utils.data.DataLoader(prev_task_testset, batch_size=args['batch_size'], shuffle=False, num_workers=2, prefetch_factor=2)
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for batch in tqdm(prev_test_loader, desc=f"Testing Previous Task {prev_task_idx}", leave=False):
                            inputs, targets = batch[0].to(device), batch[1].to(device)
                            outputs = torch.stack([model(inputs, class_idx=c).squeeze() for c in prev_task_classes], dim=1)
                            outputs = torch.sigmoid(outputs)
                            _, predicted = torch.max(outputs, 1)
                            predicted = torch.tensor([prev_task_classes[p] for p in predicted], device=device)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()

                    prev_test_acc = 100. * correct / total
                    prev_task_accuracies[prev_task_idx] = prev_test_acc
                    if rank == 0:
                        print(f"20NEWS PCL Task {task_idx}/{len(tasks)}, Previous Task {prev_task_idx} Test Accuracy: {prev_test_acc:.2f}%")
                except Exception as e:
                    if rank == 0:
                        print(f"Error in testing Previous Task {prev_task_idx} for Task {task_idx}: {e}")
                    raise

            if rank == 0:
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [task_idx, args['epochs'], test_acc, all_test_acc] + [prev_task_accuracies.get(i, 0.0) for i in range(1, len(tasks)+1)]
                    writer.writerow(row)

            if rank == 0:
                torch.save(model.state_dict(), f'checkpoint_task_{task_idx}.pth')
                print(f"Saved checkpoint for Task {task_idx}")
                print(f"Memory usage after Task {task_idx}: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
                gc.collect()

        except Exception as e:
            if rank == 0:
                print(f"Error in Task {task_idx}: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='PCL with EWC for 20 Newsgroups', add_help=False)
    parser.add_argument('--max_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_cpu', type=int, default=2, help='number of CPU workers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index (ignored)')
    parser.add_argument('--dropoutrate', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lambda_ewc', type=float, default=10.0, help='EWC regularization strength')
    parser.add_argument('--lambda_h', type=float, default=0.01, help='H-reg regularization strength')
    parser.add_argument('--debug', type=bool, default=False, help='Use subset of data for debugging')

    valid_args = ['--max_epochs', '--batch_size', '--n_cpu', '--gpu', '--dropoutrate', '--lr', '--seed', '--lambda_ewc', '--lambda_h', '--debug']
    filtered_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in valid_args:
            filtered_argv.append(sys.argv[i])
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('-'):
                filtered_argv.append(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        else:
            i += 1

    args = vars(parser.parse_args(filtered_argv[1:]))
    args['epochs'] = args.pop('max_epochs')
    print("Running in single-process mode.")
    train_pcl_ewc(0, args)

if __name__ == '__main__':

    # Ensure output directory exists and is Kaggle-friendly
    args.output_dir = args.output_dir if hasattr(args, 'output_dir') else '/kaggle/working'
    os.makedirs(args.output_dir, exist_ok=True)
    main()