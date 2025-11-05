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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import psutil
import argparse
import os
import pandas as pd
from copy import deepcopy
import logging
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EMNISTDataset(Dataset):
    def __init__(self, root, train=True, task_classes=None):
        logging.info(f"Loading EMNIST {'train' if train else 'test'} dataset at {root}")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        try:
            self.dataset = datasets.EMNIST(root=root, split='byclass', train=train, download=True, transform=transform)
        except Exception as e:
            logging.error(f"Failed to load EMNIST dataset: {str(e)}")
            raise RuntimeError(f"Could not load EMNIST dataset at {root}. Error: {str(e)}")
        
        self.task_classes = task_classes
        if task_classes is not None:
            indices = [i for i, (_, label) in enumerate(self.dataset) if label in task_classes]
            self.dataset = Subset(self.dataset, indices)

        labels = [self.dataset[i][1] for i in range(len(self.dataset))]
        unique_labels = set(labels)
        logging.info(f"Unique labels: {unique_labels}")
        if task_classes and not all(label in task_classes for label in unique_labels):
            raise ValueError(f"Unexpected labels in dataset: {unique_labels}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {'image': image, 'labels': torch.tensor(label, dtype=torch.long)}

class CustomVGG(nn.Module):
    def __init__(self, num_classes=62):
        super(CustomVGG, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7 -> 3x3
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()
        count = 0
        for batch in self.dataloader:
            self.model.zero_grad()
            images = batch['image'].to(self.device)
            labels = batch['labels'].to(self.device)
            logits = self.model(images)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n] += p.grad.data ** 2
            count += images.size(0)
        for n in precision_matrices:
            precision_matrices[n] /= count
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._means:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

def validate(model, val_loader, device, task_classes=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            logits = model(images)
            if task_classes is not None:
                task_logits = logits[:, task_classes]
                task_labels = torch.tensor([task_classes.index(l.item()) if l.item() in task_classes else -1 for l in labels]).to(device)
                valid_mask = task_labels != -1
                if not valid_mask.any():
                    continue
                task_logits = task_logits[valid_mask]
                task_labels = task_labels[valid_mask]
                _, predicted = torch.max(task_logits, 1)
                total += task_labels.size(0)
                correct += (predicted == task_labels).sum().item()
            else:
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    return 100 * correct / total if total > 0 else 0.0

def train_ewc(rank, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Rank {rank} using device: {device}")
    logging.info(f"Initial memory usage: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    # Initialize CSV results
    csv_file = args['output_csv']
    results = []
    if os.path.exists(csv_file):
        os.remove(csv_file)
        logging.info(f"Cleared existing CSV file: {csv_file}")

    # Define tasks: 10 tasks, 6-7 classes each
    task_class_map = {
        0: list(range(0, 6)),   # Classes 0-5
        1: list(range(6, 12)),  # Classes 6-11
        2: list(range(12, 18)), # Classes 12-17
        3: list(range(18, 24)), # Classes 18-23
        4: list(range(24, 30)), # Classes 24-29
        5: list(range(30, 36)), # Classes 30-35
        6: list(range(36, 42)), # Classes 36-41
        7: list(range(42, 48)), # Classes 42-47
        8: list(range(48, 54)), # Classes 48-53
        9: list(range(54, 62)), # Classes 54-61
    }
    num_tasks = 10

    # Load datasets
    logging.info("Loading EMNIST datasets")
    train_dataset_full = EMNISTDataset(args['data_root'], train=True)
    test_dataset_full = EMNISTDataset(args['data_root'], train=False)

    # Initialize model, optimizer, and criterion
    model = CustomVGG(num_classes=62).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    criterion = nn.CrossEntropyLoss()

    best_model_state = None
    best_all_acc = 0.0
    ewc = None

    for task_id in range(num_tasks):
        task_classes = task_class_map[task_id]
        logging.info(f"Starting training for Task {task_id+1}/{num_tasks}, classes {task_classes}")

        # Create task-specific datasets
        train_dataset = EMNISTDataset(args['data_root'], train=True, task_classes=task_classes)
        test_dataset = EMNISTDataset(args['data_root'], train=False, task_classes=task_classes)
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'], pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True)

        # Check class distribution
        train_labels = [item['labels'].item() for item in train_dataset]
        class_counts = {i: train_labels.count(i) for i in task_classes}
        logging.info(f"Task {task_id+1} class distribution: {class_counts}")
        if any(count == 0 for count in class_counts.values()):
            raise ValueError(f"Task {task_id+1}: Some classes have zero examples: {class_counts}")

        # Training loop
        model.train()
        for epoch in range(args['epochs']):
            total_loss = 0
            total_batches = 0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Task {task_id+1}, Epoch {epoch+1}")):
                images = batch['image'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                logits = model(images)
                task_labels = torch.tensor([task_classes.index(l.item()) for l in labels]).to(device)
                task_logits = logits[:, task_classes]
                loss = criterion(task_logits, task_labels)
                if ewc is not None:
                    loss += args['ewc_lambda'] * ewc.penalty(model)
                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError(f"Loss is NaN or Inf at Task {task_id+1}, Epoch {epoch+1}, Batch {batch_idx+1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1

                if (batch_idx + 1) % 10 == 0:
                    memory_mb = psutil.Process().memory_info().rss / 1024**2
                    logging.info(f"Task {task_id+1}, Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                                 f"Loss: {loss.item():.4f}, Memory: {memory_mb:.2f} MB")
                    results.append({
                        'Task': task_id + 1,
                        'Epoch': epoch + 1,
                        'Batch': batch_idx + 1,
                        'Train_Loss': loss.item(),
                        'Train_Accuracy': None,
                        'Test_Accuracy': None,
                        'Cumulative_Accuracy': None,
                        'Memory_MB': memory_mb
                    })

            # Compute epoch-level metrics
            train_acc = validate(model, train_loader, device, task_classes)
            avg_loss = total_loss / total_batches
            memory_mb = psutil.Process().memory_info().rss / 1024**2
            logging.info(f"Task {task_id+1}, Epoch {epoch+1}, Loss: {avg_loss:.4f}, "
                         f"Accuracy: {train_acc:.2f}%")
            results.append({
                'Task': task_id + 1,
                'Epoch': epoch + 1,
                'Batch': None,
                'Train_Loss': avg_loss,
                'Train_Accuracy': train_acc,
                'Test_Accuracy': None,
                'Cumulative_Accuracy': None,
                'Memory_MB': memory_mb
            })

        # Update EWC
        logging.info(f"Initializing EWC for Task {task_id+1}")
        task_loader = DataLoader(train_dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True)
        ewc = EWC(model, task_loader, device)
        memory_mb = psutil.Process().memory_info().rss / 1024**2
        logging.info(f"Memory usage after EWC init: {memory_mb:.2f} MB")

        # Evaluate on test set for current task
        test_acc = validate(model, test_loader, device, task_classes)
        logging.info(f"EMNIST Task {task_id+1}/{num_tasks}, Test Accuracy: {test_acc:.2f}%")

        # Evaluate on all classes up to current task
        all_classes = [cls for t in range(task_id + 1) for cls in task_class_map[t]]
        all_test_dataset = EMNISTDataset(args['data_root'], train=False, task_classes=all_classes)
        all_test_loader = DataLoader(all_test_dataset, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True)
        all_acc = validate(model, all_test_loader, device)
        logging.info(f"EMNIST Task {task_id+1}/{num_tasks}, All Classes Accuracy: {all_acc:.2f}%")

        # Save task-level results
        memory_mb = psutil.Process().memory_info().rss / 1024**2
        results.append({
            'Task': task_id + 1,
            'Epoch': None,
            'Batch': None,
            'Train_Loss': None,
            'Train_Accuracy': None,
            'Test_Accuracy': test_acc,
            'Cumulative_Accuracy': all_acc,
            'Memory_MB': memory_mb
        })

        # Write results to CSV
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
        logging.info(f"Saved results to {csv_file}")

        # Save best model
        if all_acc > best_all_acc:
            best_all_acc = all_acc
            best_model_state = deepcopy(model.state_dict())
            torch.save(best_model_state, f'best_ewc_task_{task_id+1}.pth')
            logging.info(f"Saved checkpoint for Task {task_id+1}")

        logging.info(f"Memory usage after Task {task_id+1}: {memory_mb:.2f} MB")

    if best_model_state is not None:
        torch.save(best_model_state, 'best_ewc_model.pth')
        logging.info(f"Saved best model with all classes accuracy: {best_all_acc:.2f}%")

    # Final save to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_file, index=False)
    logging.info(f"Final results saved to {csv_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per task')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--ewc_lambda', type=float, default=0.4, help='EWC regularization strength')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--data_root', type=str, default='/kaggle/working/emnist', help='Path to EMNIST dataset')
    parser.add_argument('--output_csv', type=str, default='/kaggle/working/results.csv', help='Path to output CSV file')
    
    args, _ = parser.parse_known_args()
    args = vars(args)
    
    if 'ipykernel_launcher' in sys.argv[0] or 'jupyter' in sys.argv[0]:
        logging.info("Detected Jupyter environment, using default arguments")
        args = {
            'batch_size': 128,
            'epochs': 10,
            'lr': 0.001,
            'ewc_lambda': 0.4,
            'num_workers': 4,
            'data_root': '/kaggle/working/emnist',
            'output_csv': '/kaggle/working/results.csv'
        }
    
    logging.info(f"Arguments: {args}")
    logging.info("Running in single-process mode with EMNIST dataset.")
    train_ewc(0, args)

if __name__ == '__main__':

    # Ensure output directory exists and is Kaggle-friendly
    args.output_dir = args.output_dir if hasattr(args, 'output_dir') else '/kaggle/working'
    os.makedirs(args.output_dir, exist_ok=True)
    main()