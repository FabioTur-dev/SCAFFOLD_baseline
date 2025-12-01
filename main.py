#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, random, numpy as np, scipy.io as sio, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ==============================================================
# CONFIG
# ==============================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.05, 0.5]
NUM_ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH = 128

LR_INIT = 0.01
LR_DECAY_ROUND = 15
LR_DECAY = 0.003

BETA = 0.01
DAMPING = 0.1
GRAD_CLIP = 5.0
SEED = 42


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def log(x):
    print(x, flush=True)


# ==============================================================
# SVHN LOADER — 100% bulletproof
# ==============================================================

def load_svhn(path):
    mat = sio.loadmat(path)
    X = mat["X"]               # (32,32,3,N)
    y = mat["y"].squeeze()
    y[y == 10] = 0

    # (32,32,3,N) -> (N,32,32,3)
    X = np.transpose(X, (3, 0, 1, 2))

    return X, y.astype(np.int64)


# ==============================================================
# DATASET WRAPPER
# ==============================================================

class SVHNDataset(Dataset):
    def __init__(self, X, y, idxs, augment=False):
        self.X = X
        self.y = y
        self.idxs = idxs

        if augment:
            self.T = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(160),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        else:
            self.T = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(160),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        idx = self.idxs[i]
        img = self.X[idx]          # (32,32,3)
        label = int(self.y[idx])
        return self.T(img), label


# ==============================================================
# DIRICHLET SPLIT (senza client vuoti)
# ==============================================================

def dirichlet_split(labels, n_clients, alpha):
    labels = np.array(labels)
    n = len(labels)

    per = [[] for _ in range(n_clients)]
    classes = np.unique(labels)

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for cid in range(n_clients):
            per[cid].extend(chunks[cid])

    # RIMUOVE eventuali client vuoti (causa errore num_samples=0)
    for cid in range(n_clients):
        if len(per[cid]) == 0:
            # assegna un indice random di emergenza
            per[cid].append(np.random.randint(0, n))

    for cid in range(n_clients):
        random.shuffle(per[cid])

    return per


# ==============================================================
# RESNET18 COMPATIBILE
# ==============================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()
        try:
            self.m = models.resnet18(weights="IMAGENET1K_V1")
        except:
            try:
                self.m = models.resnet18(pretrained=True)
            except:
                self.m = models.resnet18()

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        for p in self.m.parameters():
            p.requires_grad = False

        for name, p in self.m.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                p.requires_grad = True

    def forward(self, x):
        return self.m(x)


# ==============================================================
# LOCAL TRAINING — SCAFFOLD
# ==============================================================

def run_client(model, idxs, X, y, c_global, c_local, lr, device):

    ds = SVHNDataset(X, y, idxs, augment=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    old_params = [p.detach().clone().cpu() for p in trainable]

    opt = optim.SGD(trainable, lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    E = max(1, len(loader))

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()

        # SCAFFOLD correction
        for i, p in enumerate(trainable):
            p.grad += DAMPING * (c_global[i].to(device) - c_local[i].to(device))

        torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        opt.step()

    # compute deltas
    new_params = [p.detach().clone().cpu() for p in trainable]
    delta_c = []

    for i in range(len(trainable)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / E)
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, delta_c, c_local


# ==============================================================
# EVAL
# ==============================================================

def evaluate(model, X, y):
    device = next(model.parameters()).device

    T = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    correct, total = 0, 0
    for i in range(len(y)):
        img = T(X[i]).unsqueeze(0).to(device)
        label = int(y[i])

        with torch.no_grad():
            pred = model(img).argmax(1).item()

        correct += (pred == label)
        total += 1

    return correct / total


# ==============================================================
# FEDERATED
# ==============================================================

def federated_run():

    device = "cuda:0"

    # ---- load SVHN ----
    X_train, y_train = load_svhn("./data/train_32x32.mat")
    X_test,  y_test  = load_svhn("./data/test_32x32.mat")

    for alpha in DIR_ALPHAS:

        log(f"\n==== SVHN | α={alpha} ====\n")

        splits = dirichlet_split(y_train, NUM_CLIENTS, alpha)

        global_model = ResNet18Pre(10).to(device)
        trainable = [p for p in global_model.parameters() if p.requires_grad]

        c_global = [torch.zeros_like(p).cpu() for p in trainable]
        c_locals = [[torch.zeros_like(p).cpu() for p in trainable] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS+1):

            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_DECAY
            global_start = [p.detach().clone().cpu() for p in trainable]

            new_params_all = []
            delta_c_all = []

            # sequential SCAFFOLD
            for cid in range(NUM_CLIENTS):

                local_model = ResNet18Pre(10).to(device)

                # load global start
                with torch.no_grad():
                    j = 0
                    for p in local_model.parameters():
                        if p.requires_grad:
                            p.copy_(global_start[j].to(device))
                            j += 1

                new_params, dc, new_cl = run_client(
                    local_model,
                    splits[cid],
                    X_train,
                    y_train,
                    c_global,
                    c_locals[cid],
                    lr,
                    device
                )

                c_locals[cid] = new_cl
                new_params_all.append(new_params)
                delta_c_all.append(dc)

            # aggregate parameters
            avg_params = []
            for i in range(len(trainable)):
                stack = torch.stack([cp[i] for cp in new_params_all], dim=0)
                avg_params.append(stack.mean(0))

            # apply to global model
            with torch.no_grad():
                j = 0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[j].to(device))
                        j += 1

            # update c_global
            for i in range(len(c_global)):
                c_global[i] = torch.stack([dc[i] for dc in delta_c_all]).mean(0)

            # evaluate
            acc = evaluate(global_model, X_test, y_test)
            log(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")


# ==============================================================
# MAIN
# ==============================================================

def main():
    set_seed(SEED)
    federated_run()


if __name__ == "__main__":
    main()













