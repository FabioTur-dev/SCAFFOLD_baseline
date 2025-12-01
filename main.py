#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# ======================================================
# CONFIG
# ======================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.5]
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

IMG_SIZE = 224    # NEW: meglio per ResNet pretrained


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def log(x):
    print(x, flush=True)


# ======================================================
# GPU AUGMENT
# ======================================================

def gpu_augment(x):
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[3])
    return x


# ======================================================
# CIFAR-100 LOADING
# ======================================================

def load_cifar100():
    tr = datasets.CIFAR100("./data", train=True, download=True)
    te = datasets.CIFAR100("./data", train=False, download=True)

    Xtr = torch.tensor(tr.data).permute(0, 3, 1, 2).contiguous()  # [N,3,32,32]
    Xte = torch.tensor(te.data).permute(0, 3, 1, 2).contiguous()

    ytr = torch.tensor(tr.targets, dtype=torch.long)
    yte = torch.tensor(te.targets, dtype=torch.long)

    return Xtr, ytr, Xte, yte


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
        p = np.random.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for cid in range(n_clients):
            per[cid].extend(chunks[cid])

    for cid in range(n_clients):
        if len(per[cid]) == 0:
            per[cid].append(np.random.randint(0, len(labels_np)))
        random.shuffle(per[cid])

    return per


# ======================================================
# RESNET18 pretrained – unlock layer2, layer3, layer4, fc
# ======================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()

        try:
            self.m = models.resnet18(weights="IMAGENET1K_V1")
        except:
            self.m = models.resnet18(pretrained=True)

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        # freeze all
        for p in self.m.parameters():
            p.requires_grad = False

        # unlock L2, L3, L4, fc
        for name, p in self.m.named_parameters():
            if (
                name.startswith("layer2")
                or name.startswith("layer3")
                or name.startswith("layer4")
                or name.startswith("fc")
            ):
                p.requires_grad = True

    def forward(self, x):
        return self.m(x)


# ======================================================
# LOCAL TRAINING (SCAFFOLD)
# ======================================================

def run_client(model, idxs, X_cpu_u8, y_cpu, c_global, c_local, lr, device):
    model.train()
    train_params = [p for p in model.parameters() if p.requires_grad]

    old_params = [p.detach().clone() for p in train_params]

    opt = optim.SGD(train_params, lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1,3,1,1)
    std  = torch.tensor([0.5, 0.5, 0.5], device=device).view(1,3,1,1)

    for _ in range(LOCAL_EPOCHS):
        for start in range(0, len(idxs), BATCH):
            batch_idx = idxs[start:start+BATCH]

            xb_u8 = X_cpu_u8[batch_idx]
            yb = y_cpu[batch_idx].to(device)

            xb = xb_u8.to(device, dtype=torch.float32) / 255.0
            xb = F.interpolate(xb, size=IMG_SIZE, mode="bilinear", align_corners=False)
            xb = (xb - mean) / std
            xb = gpu_augment(xb)

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
    E = max(1, len(idxs) // BATCH)

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

    mean = torch.tensor([0.5,0.5,0.5], device=device).view(1,3,1,1)
    std  = torch.tensor([0.5,0.5,0.5], device=device).view(1,3,1,1)

    with torch.no_grad():
        for start in range(0, tot, 256):
            xb_u8 = X_cpu_u8[start:start+256]
            yb = y_cpu[start:start+256].to(device)

            xb = xb_u8.to(device, dtype=torch.float32) / 255.0
            xb = F.interpolate(xb, size=IMG_SIZE, mode="bilinear", align_corners=False)
            xb = (xb - mean) / std

            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()

    return correct / tot


# ======================================================
# FEDERATED LOOP
# ======================================================

def federated_run():
    device = "cuda:0"

    log("Carico CIFAR-100 (CPU, 32x32)...")
    Xtr, ytr, Xte, yte = load_cifar100()

    for alpha in DIR_ALPHAS:
        log(f"\n==== CIFAR-100 | α={alpha} ====\n")

        splits = dirichlet_split(ytr, NUM_CLIENTS, alpha)

        global_model = ResNet18Pre(100).to(device)
        train_params = [p for p in global_model.parameters() if p.requires_grad]

        c_global = [torch.zeros_like(p) for p in train_params]
        c_locals = [[torch.zeros_like(p) for p in train_params] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS+1):
            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_DECAY
            gs = [p.detach().clone() for p in train_params]

            new_params_all = []
            delta_c_all = []

            for cid in range(NUM_CLIENTS):
                local_model = ResNet18Pre(100).to(device)

                with torch.no_grad():
                    j = 0
                    for p in local_model.parameters():
                        if p.requires_grad:
                            p.copy_(gs[j])
                            j += 1

                new_params, dc, new_cl = run_client(
                    local_model, splits[cid], Xtr, ytr,
                    c_global, c_locals[cid], lr, device
                )

                c_locals[cid] = new_cl
                new_params_all.append(new_params)
                delta_c_all.append(dc)

            avg_params = []
            for i in range(len(train_params)):
                avg_params.append(torch.stack([cp[i] for cp in new_params_all]).mean(0))

            with torch.no_grad():
                j = 0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[j])
                        j += 1

            for i in range(len(train_params)):
                c_global[i] = torch.stack([dc[i] for dc in delta_c_all]).mean(0)

            acc = evaluate(global_model, Xte, yte, device)
            log(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")


def main():
    set_seed(SEED)
    federated_run()


if __name__ == "__main__":
    main()














