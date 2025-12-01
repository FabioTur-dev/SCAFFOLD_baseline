#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models


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

# SCAFFOLD – moderate (fix per CIFAR-100)
BETA = 0.003
DAMPING = 0.03
GRAD_CLIP = 5.0

SEED = 42

# CIFAR-100 → 64x64 è l'optimum
IMG_SIZE = 64


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
# RESNET18 — sblocca layer2–3–4 + fc
# ======================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()

        # compatibilità
        try:
            self.m = models.resnet18(weights="IMAGENET1K_V1")
        except:
            try:
                self.m = models.resnet18(pretrained=True)
            except:
                self.m = models.resnet18()

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        # Freeze all layers
        for p in self.m.parameters():
            p.requires_grad = False

        # Unlock specific blocks
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
# FIX FONDAMENTALE → copia pesi by-name
# ======================================================

def load_weights_by_name(target, source):
    sd_t = target.state_dict()
    sd_s = source.state_dict()

    for k in sd_t:
        if k in sd_s and sd_t[k].shape == sd_s[k].shape:
            sd_t[k].copy_(sd_s[k])


# ======================================================
# LOCAL TRAINING (SCAFFOLD)
# ======================================================

def run_client(model, idxs, X, y, c_global, c_local, lr, device):
    model.train()
    train_params = [p for p in model.parameters() if p.requires_grad]

    old_params = [p.detach().clone() for p in train_params]

    opt = optim.SGD(train_params, lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    for _ in range(LOCAL_EPOCHS):
        for start in range(0, len(idxs), BATCH):
            batch_idx = idxs[start:start+BATCH]

            xb = X[batch_idx].to(device, dtype=torch.float32)
            yb = y[batch_idx].to(device)

            xb = F.interpolate(xb, size=IMG_SIZE, mode="bilinear", align_corners=False)
            xb = (xb - mean) / std
            xb = gpu_augment(xb)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()

            # SCAFFOLD correction
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

def evaluate(model, X, y, device):
    model.eval()
    correct = 0
    tot = len(y)

    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1,3,1,1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1,3,1,1)

    with torch.no_grad():
        for start in range(0, tot, 256):
            xb = X[start:start+256].to(device, dtype=torch.float32)
            yb = y[start:start+256].to(device)

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
    tr = datasets.CIFAR100("./data", train=True, download=True)
    te = datasets.CIFAR100("./data", train=False, download=True)

    X_train = torch.tensor(tr.data).permute(0,3,1,2).float() / 255.0
    y_train = torch.tensor(tr.targets).long()

    X_test = torch.tensor(te.data).permute(0,3,1,2).float() / 255.0
    y_test = torch.tensor(te.targets).long()

    for alpha in DIR_ALPHAS:
        log(f"\n==== CIFAR-100 | α={alpha} ====\n")

        splits = dirichlet_split(y_train, NUM_CLIENTS, alpha)

        global_model = ResNet18Pre(100).to(device)
        train_params = [p for p in global_model.parameters() if p.requires_grad]

        c_global = [torch.zeros_like(p) for p in train_params]
        c_locals = [[torch.zeros_like(p) for p in train_params] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS + 1):
            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_DECAY

            new_params_all = []
            delta_c_all = []

            for cid in range(NUM_CLIENTS):
                local = ResNet18Pre(100).to(device)

                # FIX COPIA SICURA BY-NAME
                with torch.no_grad():
                    load_weights_by_name(local, global_model)

                newp, dc, newcl = run_client(
                    local, splits[cid], X_train, y_train,
                    c_global, c_locals[cid], lr, device
                )

                c_locals[cid] = newcl
                new_params_all.append(newp)
                delta_c_all.append(dc)

            # AGGREGATE NEW PARAMS
            avg_params = []
            for i in range(len(train_params)):
                avg_params.append(torch.stack([cp[i] for cp in new_params_all]).mean(0))

            with torch.no_grad():
                j = 0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[j])
                        j += 1

            # AGGIORNA C_GLOBAL
            for i in range(len(train_params)):
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














