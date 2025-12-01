#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ======================================================
# CONFIG
# ======================================================
NUM_CLIENTS = 10
DIR_ALPHAS = [0.5]
NUM_ROUNDS = 30
LOCAL_EPOCHS = 1
BATCH = 128

# BEST hyperparameters (dalla tua versione)
LR_INIT = 0.01
LR_DECAY_ROUND = 15
LR_DECAY = 0.0015

# SCAFFOLD
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

# ======================================================
# LOAD SVHN
# ======================================================
def load_svhn_raw(path):
    mat = sio.loadmat(path)
    X = mat["X"]          # (32,32,3,N)
    y = mat["y"].squeeze()
    y[y == 10] = 0
    y = y.astype(np.int64)
    X = np.transpose(X, (3, 2, 0, 1))
    X_t = torch.from_numpy(X).to(torch.uint8)
    y_t = torch.from_numpy(y).long()
    return X_t, y_t

# ======================================================
# DIRICHLET SPLIT
# ======================================================
def dirichlet_split(labels, n_clients, alpha):
    labels_np = labels.numpy()
    per = [[] for _ in range(n_clients)]
    classes = np.unique(labels_np)

    for c in classes:
        idx = np.where(labels_np == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha]*n_clients)
        cuts = (np.cumsum(p)*len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for cid in range(n_clients):
            per[cid].extend(chunks[cid])

    for cid in range(n_clients):
        if len(per[cid]) == 0:
            per[cid].append(np.random.randint(0, len(labels_np)))
        random.shuffle(per[cid])

    return per

# ======================================================
# MINI-RESNET SVHN-FRIENDLY (ResNet-20 style)
# ======================================================

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.shortcut = (
            nn.Identity() if in_c == out_c and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class MiniResNetSVHN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(16, 16, stride=1)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)

        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_c, out_c, stride):
        return nn.Sequential(
            BasicBlock(in_c, out_c, stride),
            BasicBlock(out_c, out_c, 1)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ======================================================
# LOCAL TRAINING
# ======================================================
def run_client(model, idxs, X_cpu_u8, y_cpu, c_global, c_local, lr, device):

    model.train()
    train_params = [p for p in model.parameters() if p.requires_grad]
    old_params = [p.detach().clone() for p in train_params]

    opt = optim.SGD(train_params, lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(LOCAL_EPOCHS):
        for start in range(0, len(idxs), BATCH):
            b = idxs[start:start+BATCH]
            xb = X_cpu_u8[b].to(device, dtype=torch.float32) / 255.0
            yb = y_cpu[b].to(device)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()

            for i, p in enumerate(train_params):
                p.grad += DAMPING * (c_global[i] - c_local[i])

            torch.nn.utils.clip_grad_norm_(train_params, GRAD_CLIP)
            opt.step()

    new_params = [p.detach().clone() for p in train_params]
    delta_c = []
    E = max(1, len(idxs)//BATCH)

    for i in range(len(train_params)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / E)
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, delta_c, c_local

# ======================================================
# EVAL
# ======================================================
def evaluate(model, X_cpu_u8, y_cpu, device):
    model.eval()
    correct = 0
    tot = len(y_cpu)

    with torch.no_grad():
        for start in range(0, tot, 256):
            xb = X_cpu_u8[start:start+256].to(device, dtype=torch.float32) / 255.0
            yb = y_cpu[start:start+256].to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()

    return correct / tot

# ======================================================
# FEDERATED LOOP
# ======================================================
def federated_run():
    device = "cuda:0"

    log("Carico SVHN (CPU, 32x32)...")
    X_train, y_train = load_svhn_raw("./data/train_32x32.mat")
    X_test, y_test = load_svhn_raw("./data/test_32x32.mat")

    for alpha in DIR_ALPHAS:
        log(f"\n==== SVHN | α={alpha} ====\n")

        splits = dirichlet_split(y_train, NUM_CLIENTS, alpha)
        global_model = MiniResNetSVHN(10).to(device)

        train_params = [p for p in global_model.parameters() if p.requires_grad]
        c_global = [torch.zeros_like(p) for p in train_params]
        c_locals = [[torch.zeros_like(p) for p in train_params] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS+1):
            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_DECAY

            gs = [p.detach().clone() for p in train_params]

            new_params_all = []
            delta_c_all = []

            for cid in range(NUM_CLIENTS):
                local_model = MiniResNetSVHN(10).to(device)

                with torch.no_grad():
                    j = 0
                    for p in local_model.parameters():
                        if p.requires_grad:
                            p.copy_(gs[j])
                            j += 1

                newp, dc, newcl = run_client(
                    local_model, splits[cid],
                    X_train, y_train,
                    c_global, c_locals[cid],
                    lr, device
                )

                new_params_all.append(newp)
                delta_c_all.append(dc)
                c_locals[cid] = newcl

            avg_params = []
            for i in range(len(train_params)):
                avg_params.append(
                    torch.stack([cp[i] for cp in new_params_all]).mean(0)
                )

            with torch.no_grad():
                j = 0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[j])
                        j += 1

            for i in range(len(c_global)):
                c_global[i] = torch.stack([dc[i] for dc in delta_c_all]).mean(0)

            acc = evaluate(global_model, X_test, y_test, device)
            log(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")

# ======================================================
# MAIN
# ======================================================
def main():
    set_seed(SEED)
    federated_run()

if __name__ == "__main__":
    main()














