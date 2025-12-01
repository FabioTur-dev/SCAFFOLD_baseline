#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random, numpy as np, scipy.io as sio, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ==============================================================
# CONFIG
# ==============================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.5]
NUM_ROUNDS = 50
LOCAL_EPOCHS = 2
BATCH = 128

BASE_LR = 0.01
LR_LAYER1 = BASE_LR * 0.01
LR_LAYER2 = BASE_LR * 0.10
LR_FINAL   = BASE_LR

BETA = 0.01
DAMPING = 0.05
GRAD_CLIP = 5.0
IMG_SIZE = 160
SEED = 42


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def log(x): print(x, flush=True)



# ==============================================================
# LOAD SVHN (SAFE)
# ==============================================================

def load_svhn_cpu(path):
    d = sio.loadmat(path)

    X = d["X"]      # (32,32,3,N)
    y = d["y"].reshape(-1)

    # -> (N,3,32,32)
    X = np.transpose(X, (3, 2, 0, 1))
    X = torch.tensor(X, dtype=torch.float32) / 255.0
    y = torch.tensor(y, dtype=torch.long)
    y[y == 10] = 0

    return X, y



# ==============================================================
# DATASET FIXATO (definitivo)
# ==============================================================

class RawDataset(Dataset):
    def __init__(self, X, y, idx):
        self.X = X
        self.y = y
        self.idx = idx

        self.T = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(IMG_SIZE, padding=8),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        img = self.X[k]

        # safety: ensure CHW
        if img.dim() == 4:
            img = img.squeeze()
        if img.shape[0] != 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        img = self.T(img)
        return img, self.y[k]



# ==============================================================
# DIRICHLET SPLIT
# ==============================================================

def dirichlet_split(labels, n_clients, alpha):
    labels = labels.numpy()
    cls = np.unique(labels)
    split = [[] for _ in range(n_clients)]

    for c in cls:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha]*n_clients)
        cuts = (np.cumsum(p)*len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for i in range(n_clients):
            split[i].extend(chunks[i])

    for s in split: random.shuffle(s)
    return split



# ==============================================================
# MODEL (LR MULTISCALE / BN trainabile)
# ==============================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.m = models.resnet18(pretrained=True)

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        self.param_groups = {
            "layer1": [], "layer2": [],
            "layer3": [], "layer4": [], "fc": []
        }

        for name, p in self.m.named_parameters():
            if "layer1" in name:
                self.param_groups["layer1"].append(p)
            elif "layer2" in name:
                self.param_groups["layer2"].append(p)
            elif "layer3" in name:
                self.param_groups["layer3"].append(p)
            elif "layer4" in name:
                self.param_groups["layer4"].append(p)
            else:
                self.param_groups["fc"].append(p)

    def get_optimizer(self):
        return optim.SGD([
            {"params": self.param_groups["layer1"], "lr": LR_LAYER1},
            {"params": self.param_groups["layer2"], "lr": LR_LAYER2},
            {"params": self.param_groups["layer3"], "lr": LR_FINAL},
            {"params": self.param_groups["layer4"], "lr": LR_FINAL},
            {"params": self.param_groups["fc"],     "lr": LR_FINAL},
        ], momentum=0.9, weight_decay=5e-4)

    def forward(self, x): return self.m(x)



# ==============================================================
# LOCAL TRAINING
# ==============================================================

def run_client(model, idxs, X, y, c_global, c_local, device, rnd):
    ds = RawDataset(X, y, idxs)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2)

    opt = model.get_optimizer()
    ce = nn.CrossEntropyLoss()

    params = [p for p in model.parameters()]
    old = [p.detach().clone().cpu() for p in params]

    freeze_l1 = (rnd <= 2)

    # Map param index → name
    param_names = [n for (n, _) in model.m.named_parameters()]

    for ep in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            out = model(xb)
            loss = ce(out, yb)
            loss.backward()

            for i, p in enumerate(params):
                name = param_names[i]

                # warmup
                if freeze_l1 and "layer1" in name:
                    p.grad.zero_()
                    continue

                p.grad += DAMPING * (c_global[i].to(device) - c_local[i].to(device))

            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()

    new = [p.detach().clone().cpu() for p in params]

    delta_c = []
    for i in range(len(params)):
        diff = new[i] - old[i]
        dc = BETA * diff
        delta_c.append(dc)
        c_local[i] += dc

    return new, delta_c, c_local



# ==============================================================
# EVAL
# ==============================================================

def evaluate(model, Xte, Yte, device):
    Ttest = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    class TestDS(Dataset):
        def __len__(self): return len(Xte)
        def __getitem__(self, i):
            img = Xte[i]
            if img.dim() == 4: img = img.squeeze()
            if img.shape[0] != 3 and img.shape[-1] == 3:
                img = img.permute(2,0,1)
            img = Ttest(img)
            return img, Yte[i]

    loader = DataLoader(TestDS(), batch_size=256, shuffle=False)

    model.eval()
    c = 0
    t = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            c += (pred == yb).sum().item()
            t += yb.size(0)
    return c/t



# ==============================================================
# FEDERATED LOOP
# ==============================================================

def federated_run():
    device = "cuda"

    log("Carico SVHN (CPU, 32x32)...")
    Xtr, Ytr = load_svhn_cpu("./data/train_32x32.mat")
    Xte, Yte = load_svhn_cpu("./data/test_32x32.mat")

    for alpha in DIR_ALPHAS:
        log(f"\n==== SVHN | α={alpha} ====\n")

        splits = dirichlet_split(Ytr, NUM_CLIENTS, alpha)

        global_model = ResNet18Pre(10).to(device)
        params = [p for p in global_model.parameters()]

        c_global = [torch.zeros_like(p).cpu() for p in params]
        c_locals = [[torch.zeros_like(p).cpu() for p in params] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS+1):

            new_all = []
            dc_all = []

            for cid in range(NUM_CLIENTS):

                # clone weights
                local = ResNet18Pre(10).to(device)
                j = 0
                for p in local.parameters():
                    p.data.copy_(params[j].data)
                    j += 1

                newp, dc, newcl = run_client(
                    local, splits[cid], Xtr, Ytr,
                    c_global, c_locals[cid],
                    device, rnd
                )

                new_all.append(newp)
                dc_all.append(dc)
                c_locals[cid] = newcl

            avg = []
            for i in range(len(params)):
                avg.append(torch.stack([cl[i] for cl in new_all]).mean(0))

            with torch.no_grad():
                for i, p in enumerate(params):
                    p.copy_(avg[i].to(device))

            for i in range(len(c_global)):
                c_global[i] = torch.stack([dc[i] for dc in dc_all]).mean(0)

            acc = evaluate(global_model, Xte, Yte, device)
            log(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")


# ==============================================================
# MAIN
# ==============================================================

def main():
    set_seed(SEED)
    federated_run()


if __name__ == "__main__":
    main()














