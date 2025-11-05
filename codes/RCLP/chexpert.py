#!/usr/bin/env python3
# chexpert.py - RCLP-style training script for CheXpert dataset
# (trimmed header for brevity)
import os, sys, argparse, time, json, copy, random
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms, models
from PIL import Image

def seed_everything(seed=0):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = False

class MultiLabelImageDataset(Dataset):
    def __init__(self, df, root_dir, img_col='image_path', label_col='labels', transform=None, classes=None):
        self.df = df.reset_index(drop=True)
        self.root = Path(root_dir)
        self.img_col = img_col; self.label_col = label_col; self.transform = transform
        if classes is None:
            all_labels = set()
            for v in self.df[label_col].fillna(''):
                if isinstance(v, str):
                    parts = [x.strip() for x in v.split(';') if x.strip()!='']
                    for p in parts: all_labels.add(p)
            self.classes = sorted(list(all_labels))
        else:
            self.classes = classes
        self.class2idx = {c:i for i,c in enumerate(self.classes)}

    def __len__(self): return len(self.df)
    def _parse_labels(self, labstr):
        vec = np.zeros(len(self.classes), dtype=np.float32)
        if isinstance(labstr, (list,tuple)): parts = labstr
        else: parts = [x.strip() for x in str(labstr).split(';') if x.strip()!='']
        for p in parts:
            if p in self.class2idx: vec[self.class2idx[p]] = 1.0
        return vec
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row[self.img_col]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None: img = self.transform(img)
        labels = self._parse_labels(row[self.label_col])
        return img, torch.from_numpy(labels)

class DenseNet121Feature(nn.Module):
    def __init__(self, pretrained=True, out_features=19):
        super().__init__()
        dnet = models.densenet121(pretrained=pretrained)
        self.features = dnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(1024, out_features)
        self._intermediate = None
        def _hook(module, inp, out): self._intermediate = out.detach()
        try:
            self.features.denseblock3.register_forward_hook(_hook)
        except Exception:
            self.features.register_forward_hook(_hook)
    def forward(self, x, return_features=False):
        feats = self.features(x)
        pooled = self.avgpool(feats).view(x.size(0), -1)
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled, self._intermediate
        return logits

class ReplayMemory:
    def __init__(self, max_size):
        self.max_size = int(max_size); self.data = []; random.seed(0)
    def add_samples(self, samples):
        self.data.extend(samples)
        if len(self.data) > self.max_size: self.data = random.sample(self.data, self.max_size)
    def sample_batch(self, batch_size):
        if len(self.data)==0: return []
        return random.sample(self.data, min(batch_size, len(self.data)))
    def __len__(self): return len(self.data)

def get_transforms(image_size=256):
    return transforms.Compose([transforms.Resize((image_size,image_size)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def mask_loss_on_replay(logits, targets, old_label_indices):
    mask = torch.zeros_like(targets)
    mask[:, old_label_indices] = 1.0
    bce = nn.BCEWithLogitsLoss(reduction='none')
    loss_per_element = bce(logits, targets)
    masked = loss_per_element * mask
    denom = mask.sum()
    if denom == 0: return masked.sum()*0.0
    return masked.sum()/denom

from sklearn.metrics import f1_score, roc_auc_score

def evaluate_model_metrics(model, dataloader, device, num_classes):
    model.eval(); sigmoid = nn.Sigmoid()
    all_probs = []; all_targets = []
    with torch.no_grad():
        for x,y in dataloader:
            x = x.to(device)
            logits, pooled, inter = model(x, return_features=True)
            probs = sigmoid(logits).cpu().numpy(); all_probs.append(probs); all_targets.append(y.numpy())
    if len(all_probs)==0: return {'avg_f1':0.0,'avg_auc':0.0}
    all_probs = np.vstack(all_probs); all_targets = np.vstack(all_targets)
    f1s=[]; aucs=[]
    for c in range(num_classes):
        gt = all_targets[:,c]; pred = (all_probs[:,c]>0.5).astype(int)
        f1s.append(f1_score(gt,pred,zero_division=0))
        try: aucs.append(roc_auc_score(gt, all_probs[:,c]))
        except: aucs.append(0.5)
    return {'avg_f1':np.mean(f1s),'avg_auc':np.mean(aucs)}

def compute_class_thresholds_local(model, dataloader, device, num_classes):
    model.eval(); sigmoid = nn.Sigmoid()
    all_outputs=[]; all_targets=[]
    with torch.no_grad():
        for x,y in dataloader:
            x = x.to(device)
            logits, pooled, inter = model(x, return_features=True)
            probs = sigmoid(logits).cpu().numpy(); all_outputs.append(probs); all_targets.append(y.numpy())
    if len(all_outputs)==0: return np.array([0.5]*num_classes)
    all_outputs = np.vstack(all_outputs); all_targets = np.vstack(all_targets)
    thresholds = np.zeros(num_classes, dtype=np.float32)
    from sklearn.metrics import f1_score
    for c in range(num_classes):
        gt = all_targets[:,c]
        if gt.sum()==0: thresholds[c]=0.5; continue
        best_thr=0.5; best_f1=0.0
        for thr in np.linspace(0.1,0.9,17):
            preds=(all_outputs[:,c]>thr).astype(int); f1=f1_score(gt,preds,zero_division=0)
            if f1>best_f1: best_f1=f1; best_thr=thr
        thresholds[c]=best_thr
    return thresholds

def train_rclp(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    if not os.path.exists(args.meta_csv): print('Meta CSV not found', args.meta_csv); sys.exit(1)
    meta = pd.read_csv(args.meta_csv)
    for c in ['image_path','labels']:
        if c not in meta.columns: print('Meta CSV missing', c); sys.exit(1)
    dataset = MultiLabelImageDataset(meta, args.data_dir, transform=get_transforms(), classes=args.classes.split(';') if args.classes else None)
    num_classes = len(dataset.classes); print('Num samples', len(dataset), 'Num classes', num_classes)
    n = len(dataset); n_train=int(0.8*n); n_val=int(0.1*n); n_test=n-n_train-n_val
    train_set, val_set, test_set = random_split(dataset, [n_train,n_val,n_test], generator=torch.Generator().manual_seed(args.seed))
    train_loader_full = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = DenseNet121Feature(pretrained=True, out_features=num_classes).to(device)
    old_model = copy.deepcopy(model); old_model.eval(); 
    for p in old_model.parameters(): p.requires_grad=False
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mem_size = max(1, int(len(train_set)*0.03))
    memory = ReplayMemory(mem_size)
    if not os.path.exists(args.tasks_json): print('Tasks JSON not found', args.tasks_json); sys.exit(1)
    tasks_spec = json.load(open(args.tasks_json,'r')); task_order=list(tasks_spec.keys())
    sigmoid = nn.Sigmoid()
    results=[]
    for t_idx, t_key in enumerate(task_order):
        task_classes = tasks_spec[t_key]; task_label_indices = [dataset.class2idx[c] for c in task_classes if c in dataset.class2idx]
        print(f"Task {t_idx+1}/{len(task_order)} classes={task_classes}")
        # collect indices in train_set whose positive labels include any of task_label_indices
        train_indices = [i for i in range(len(train_set)) if train_set[i][1].numpy()[task_label_indices].sum()>0]
        if len(train_indices)==0: print('No samples for this task; skipping'); continue
        task_subset = Subset(train_set, train_indices); task_loader = DataLoader(task_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        # train epochs
        for epoch in range(args.epochs_per_task):
            model.train(); epoch_loss=0.0; n_samples=0
            for xb, yb in task_loader:
                xb=xb.to(device); yb=yb.to(device)
                logits, pooled, inter = model(xb, return_features=True)
                loss_curr = nn.BCEWithLogitsLoss()(logits, yb)
                loss_replay = torch.tensor(0.0, device=device); loss_distill = torch.tensor(0.0, device=device)
                replay_batch = memory.sample_batch(args.batch_size)
                if len(replay_batch)>0:
                    imgs=[]; targets=[]
                    for img_path, labels_vec, orig_idx in replay_batch:
                        try:
                            img = Image.open(os.path.join(args.data_dir, img_path)).convert('RGB')
                            img = get_transforms()(img); imgs.append(img); targets.append(torch.tensor(labels_vec,dtype=torch.float32))
                        except Exception:
                            continue
                    if len(imgs)>0:
                        imgs = torch.stack(imgs).to(device); targets = torch.stack(targets).to(device)
                        logits_r, pooled_r, inter_r = model(imgs, return_features=True)
                        old_label_indices = []
                        for prev_k in task_order[:t_idx]:
                            for cname in tasks_spec[prev_k]:
                                if cname in dataset.class2idx: old_label_indices.append(dataset.class2idx[cname])
                        if len(old_label_indices)>0:
                            loss_replay = mask_loss_on_replay(logits_r, targets, old_label_indices)
                        with torch.no_grad():
                            logits_old, pooled_old, inter_old = old_model(imgs, return_features=True)
                        loss_distill = nn.MSELoss()(pooled_r, pooled_old.detach()) * args.gamma
                loss = loss_curr + loss_replay + loss_distill
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += loss.item()*xb.size(0); n_samples += xb.size(0)
            if n_samples>0: print(f" Epoch {epoch+1}/{args.epochs_per_task} - loss {epoch_loss/n_samples:.4f}")
        # compute thresholds on val set
        thresholds = compute_class_thresholds_local(model, val_loader, device, num_classes)
        # backward propagate labels to memory
        if len(memory)>0:
            updated=[]
            for img_path, labels_vec, orig_idx in memory.data:
                try:
                    img = Image.open(os.path.join(args.data_dir, img_path)).convert('RGB')
                    img_t = get_transforms()(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits_m, pooled_m, inter_m = model(img_t, return_features=True)
                        probs = sigmoid(logits_m).cpu().numpy()[0]
                    new_labels = (probs > thresholds).astype(float)
                    updated.append((img_path, new_labels, orig_idx))
                except Exception:
                    updated.append((img_path, labels_vec, orig_idx))
            memory.data = updated
        # forward step: add subset of current task samples with propagated labels to memory
        to_add=[]
        task_indices_list = list(range(len(task_subset))); random.shuffle(task_indices_list)
        num_to_store = max(1, int(mem_size / max(1, len(task_order))))
        for idx in task_indices_list[:num_to_store]:
            img, labels = task_subset[idx]
            with torch.no_grad():
                img_t = img.unsqueeze(0).to(device)
                logits_m, pooled_m, inter_m = model(img_t, return_features=True)
                probs = sigmoid(logits_m).cpu().numpy()[0]
            pseudo = (probs > thresholds).astype(float)
            fname = getattr(img, 'filename', f't{t_idx}_idx{idx}')
            to_add.append((fname, pseudo.tolist(), -1))
        memory.add_samples(to_add)
        old_model = copy.deepcopy(model); old_model.eval(); 
        for p in old_model.parameters(): p.requires_grad=False
        # evaluate
        metrics = evaluate_model_metrics(model, test_loader, device, num_classes)
        print(f" After Task {t_idx+1}: avg_f1 {metrics['avg_f1']:.4f}, avg_auc {metrics['avg_auc']:.4f}")
    # end tasks loop

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--meta-csv', type=str, required=True)
    parser.add_argument('--tasks-json', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./outputs')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs-per-task', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--run-name', type=str, default='rclp_chexpert')
    parser.add_argument('--classes', type=str, default='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_rclp(args)
