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
from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
import argparse
import psutil
import csv
from torchvision import datasets, transforms
from tqdm import tqdm
import gc

class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 640)
        self.dropout = nn.Dropout(dropout_rate)
        self.class_heads = nn.ModuleDict()

    def add_head(self, class_idx, device):
        self.class_heads[str(class_idx)] = nn.Sequential(
            nn.Linear(640, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, x, class_idx=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        features = self.fc1(x)
        features = self.dropout(features)
        if class_idx is not None:
            return self.class_heads[str(class_idx)](features)
        return {k: v(features) for k, v in self.class_heads.items()}

class EWC(object):
    def __init__(self, model, dataset, device, batch_size=32, num_batches=10):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        for n, p in self.params.items():
            self._means[n] = p.clone().detach()
        print(f"Memory usage after EWC init: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    def _diag_fisher(self):
        precision_matrices = {n: torch.zeros_like(p).to(self.device) for n, p in self.params.items()}
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for i, (inputs, targets) in enumerate(tqdm(dataloader, desc="Computing Fisher Matrix", leave=False)):
            if i >= self.num_batches:
                break
            self.model.zero_grad()
            inputs = inputs.to(self.device).requires_grad_(True)
            outputs = self.model(inputs, class_idx=targets[0].item())
            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), torch.ones_like(outputs.squeeze()))
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None and n in self.params:
                    precision_matrices[n] += p.grad.data ** 2 / len(dataloader) * 0.1
        for n in precision_matrices:
            precision_matrices[n] /= precision_matrices[n].max() + 1e-8
        return precision_matrices

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
    device = torch.device('cpu')  # Kaggle CPU
    if rank == 0:
        print(f"Rank {rank} using device: {device}")
        print(f"Initial memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if args['debug']:
        train_indices = np.random.choice(len(trainset), len(trainset) // 10, replace=False)
        test_indices = np.random.choice(len(testset), len(testset) // 10, replace=False)
        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)

    num_classes = 10
    classes_per_task = 2
    tasks = [(i, min(i + classes_per_task - 1, num_classes - 1)) for i in range(0, num_classes, classes_per_task)]
    tasks = [tuple(range(start, end + 1)) for start, end in tasks]
    if rank == 0:
        print(f"MNIST: {num_classes} classes, {len(tasks)} tasks, classes per task: {tasks}")

    model = SimpleCNN(dropout_rate=args['dropoutrate']).to(device)
    if rank == 0:
        print(f"Memory usage after model init: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
        gc.collect()

    csv_file = '/kaggle/working/mnist_pcl_ewc_results.csv'
    if rank == 0:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Task', 'Epoch', 'Task_Accuracy', 'All_Classes_Accuracy'] + [f'Previous_Task_{i}_Accuracy' for i in range(1, len(tasks)+1)])

    ewc_list = []
    for task_idx, task_classes in enumerate(tasks, 1):
        try:
            if rank == 0:
                print(f"Starting training for Task {task_idx}/{len(tasks)}")
            for class_idx in task_classes:
                model.add_head(class_idx, device)
            if rank == 0:
                print(f"Added heads for classes {task_classes}")

            optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args['lr'], momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])

            train_indices = [i for i, (_, label) in enumerate(trainset) if label in task_classes]
            test_indices = [i for i, (_, label) in enumerate(testset) if label in task_classes]
            task_trainset = Subset(trainset, train_indices)
            task_testset = Subset(testset, test_indices)

            if rank == 0:
                print(f"Task {task_idx}/{len(tasks)}: {len(task_trainset)} train samples, {len(task_testset)} test samples")
                print(f"Task {task_idx} labels: {set([trainset[i][1] for i in train_indices])}")
                if len(task_trainset) == 0 or len(task_testset) == 0:
                    raise ValueError(f"Empty dataset for Task {task_idx}")

            train_loader = DataLoader(task_trainset, batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=False)
            test_loader = DataLoader(task_testset, batch_size=args['batch_size'], shuffle=False, num_workers=4, pin_memory=False)

            for epoch in range(args['epochs']):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Task {task_idx}, Epoch {epoch + 1}", leave=False), 1):
                    inputs, labels = inputs.to(device).requires_grad_(True), labels.to(device)
                    optimizer.zero_grad()
                    loss = 0
                    for class_idx in task_classes:
                        outputs = model(inputs, class_idx=class_idx)
                        class_targets = (labels == class_idx).float()
                        nll = F.binary_cross_entropy_with_logits(outputs.squeeze(), class_targets)
                        features = model.conv1(inputs)
                        features = model.pool(F.relu(features))
                        features = model.conv2(features)
                        features = model.pool(F.relu(features))
                        features = features.view(-1, 64 * 7 * 7)
                        features = model.fc1(features)
                        features = model.dropout(features)
                        head_output = model.class_heads[str(class_idx)](features)
                        grad_input = torch.autograd.grad(head_output.sum(), inputs, create_graph=True)[0]
                        h_reg = torch.norm(grad_input, p=2, dim=(1, 2, 3)).mean()
                        loss += nll + args['lambda_h'] * h_reg
                    if task_idx > 1:
                        for ewc in ewc_list:
                            loss += args['lambda_ewc'] * ewc.penalty(model)
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise ValueError(f"Loss is NaN or Inf at Task {task_idx}, Epoch {epoch + 1}, Batch {batch_idx}")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

                    running_loss += loss.item()
                    outputs = torch.stack([model(inputs, class_idx=c).squeeze() for c in task_classes], dim=1)
                    outputs = torch.sigmoid(outputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted = torch.tensor([task_classes[p.item()] for p in predicted], device=labels.device)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    if batch_idx % 10 == 0 or batch_idx == len(train_loader):
                        if rank == 0:
                            print(f"Task {task_idx}, Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, "
                                  f"Loss: {loss.item():.4f}, Memory: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
                            gc.collect()

                epoch_loss = running_loss / len(train_loader)
                epoch_acc = 100. * correct / total
                if rank == 0:
                    print(f"Task {task_idx}, Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
                scheduler.step()

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
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc=f"Testing Task {task_idx}", leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = torch.stack([model(inputs, class_idx=c).squeeze() for c in task_classes], dim=1)
                    outputs = torch.sigmoid(outputs)
                    _, predicted = torch.max(outputs, 1)
                    predicted = torch.tensor([task_classes[p.item()] for p in predicted], device=labels.device)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            test_acc = 100. * correct / total
            if rank == 0:
                print(f"MNIST PCL Task {task_idx}/{len(tasks)}, Test Accuracy: {test_acc:.2f}%")

            all_test_acc = 0.0
            if rank == 0:
                print(f"Evaluating all classes up to Task {task_idx}")
                all_test_indices = [i for i, (_, label) in enumerate(testset) if label in [c for t in tasks[:task_idx] for c in t]]
                all_task_testset = Subset(testset, all_test_indices)
                all_test_loader = DataLoader(all_task_testset, batch_size=args['batch_size'], shuffle=False, num_workers=4, pin_memory=False)
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in tqdm(all_test_loader, desc=f"Testing All Classes Task {task_idx}", leave=False):
                        inputs, labels = inputs.to(device), labels.to(device)
                        all_classes = [c for t in tasks[:task_idx] for c in t]
                        outputs = torch.stack([model(inputs, class_idx=c).squeeze() for c in all_classes], dim=1)
                        outputs = torch.sigmoid(outputs)
                        _, predicted = torch.max(outputs, 1)
                        predicted = torch.tensor([all_classes[p.item()] for p in predicted], device=labels.device)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                all_test_acc = 100. * correct / total
                print(f"MNIST PCL Task {task_idx}/{len(tasks)}, All Classes Accuracy: {all_test_acc:.2f}%")

            prev_task_accuracies = {}
            for prev_task_idx, prev_task_classes in enumerate(tasks[:task_idx], 1):
                prev_test_indices = [i for i, (_, label) in enumerate(testset) if label in prev_task_classes]
                prev_task_testset = Subset(testset, prev_test_indices)
                prev_test_loader = DataLoader(prev_task_testset, batch_size=args['batch_size'], shuffle=False, num_workers=4, pin_memory=False)
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in tqdm(prev_test_loader, desc=f"Testing Previous Task {prev_task_idx}", leave=False):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = torch.stack([model(inputs, class_idx=c).squeeze() for c in prev_task_classes], dim=1)
                        outputs = torch.sigmoid(outputs)
                        _, predicted = torch.max(outputs, 1)
                        predicted = torch.tensor([prev_task_classes[p.item()] for p in predicted], device=labels.device)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                prev_test_acc = 100. * correct / total
                prev_task_accuracies[prev_task_idx] = prev_test_acc
                if rank == 0:
                    print(f"MNIST PCL Task {task_idx}/{len(tasks)}, Previous Task {prev_task_idx} Test Accuracy: {prev_test_acc:.2f}%")

            if rank == 0:
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [task_idx, args['epochs'], test_acc, all_test_acc] + [prev_task_accuracies.get(i, 0.0) for i in range(1, len(tasks)+1)]
                    writer.writerow(row)

            if rank == 0:
                torch.save(model.state_dict(), f'/kaggle/working/checkpoint_task_{task_idx}.pth')
                print(f"Saved checkpoint for Task {task_idx}")
                print(f"Memory usage after Task {task_idx}: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
                gc.collect()

        except Exception as e:
            if rank == 0:
                print(f"Error in Task {task_idx}: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='PCL with EWC for MNIST', add_help=False)
    parser.add_argument('--dataset', default='mnist', choices=['mnist'])
    parser.add_argument('--max_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of CPU workers')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lambda_ewc', type=float, default=1.0, help='EWC regularization strength')
    parser.add_argument('--lambda_h', type=float, default=0.001, help='H-reg regularization strength')
    parser.add_argument('--dropoutrate', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--debug', type=bool, default=False, help='Use subset of data for debugging')

    valid_args = ['--dataset', '--max_epochs', '--batch_size', '--n_cpu', '--lr', '--seed', '--lambda_ewc', '--lambda_h', '--dropoutrate', '--debug']
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