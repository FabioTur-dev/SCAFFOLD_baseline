#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import to_pil_image


# ==============================================================
# CONFIG – Sequential SCAFFOLD (REAL & STABLE)
# ==============================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.5]
NUM_ROUNDS = 50
LOCAL_EPOCHS = 2
BATCH = 128

LR_INIT = 0.004
LR_DECAY_ROUND = 20
LR_DECAY = 0.0015

BETA = 0.01
DAMPING = 0.1
GRAD_CLIP = 5.0
SEED = 42


# ==============================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def log(msg):
    print(msg, flush=True)


# ==============================================================
# DATASET WRAPPER
# ==============================================================

class RawDataset(Dataset):
    def __init__(self, data, labels, indices, augment=False):
        self.data = data
        self.labels = labels
        self.indices = indices

        if augment:
            self.T = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(112),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),
                                     (0.229,0.224,0.225)),
                transforms.RandomErasing(p=0.25)
            ])
        else:
            self.T = transforms.Compose([
                transforms.Resize(112),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),
                                     (0.229,0.224,0.225)),
            ])

    def __getitem__(self, i):
        idx = self.indices[i]
        img = to_pil_image(self.data[idx])
        img = self.T(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.indices)


# ==============================================================
# DIRICHLET SPLIT
# ==============================================================

def dirichlet_split(labels, n_clients, alpha):
    labels = np.array(labels)
    per = [[] for _ in range(n_clients)]
    classes = np.unique(labels)

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for i in range(n_clients):
            per[i].extend(chunks[i])
    for cl in per:
        random.shuffle(cl)

    return per


# ==============================================================
# MODEL – RESNET18 (layer3 + layer4 + fc sbloccati)
# ==============================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights
            self.m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except:
            self.m = models.resnet18(pretrained=True)

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        for name, p in self.m.named_parameters():
            if ("layer3" in name) or ("layer4" in name) or ("fc" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, x):
        return self.m(x)


# ==============================================================
# LOCAL TRAINING (SCAFFOLD CLIENT)
# ==============================================================

def run_client(model, train_idx, data, labels, c_global, c_local, lr, device):
    ds = RawDataset(data, labels, train_idx, augment=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    old_params = [p.detach().clone().cpu() for p in trainable]

    opt = optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    E = len(loader)

    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()

            for i, p in enumerate(trainable):
                p.grad += DAMPING * (c_global[i].to(device) - c_local[i].to(device))

            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step()

    # Compute scaffold deltas
    new_params = [p.detach().clone().cpu() for p in trainable]
    delta_c = []

    for i in range(len(trainable)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / max(E, 1))
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, delta_c, c_local


# ==============================================================
# GLOBAL EVAL
# ==============================================================

def evaluate(model, loader, device):
    model.eval()
    c = 0
    tot = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            c += (pred == y).sum().item()
            tot += y.size(0)
    return c / tot


# ==============================================================
# FEDERATED LOOP (SEQUENTIAL)
# ==============================================================

def federated_run(ds_name):

    device = "cuda:0"

    # Load raw data once
    if ds_name == "CIFAR10":
        tr = datasets.CIFAR10("./data", train=True, download=True)
        data = torch.tensor(tr.data).permute(0, 3, 1, 2)
        labels = torch.tensor(tr.targets)
        nc = 10
    else:
        raise ValueError("Only CIFAR10 implemented in version A")

    # Test loader
    transform_test = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225))
    ])
    te = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
    testloader = DataLoader(te, batch_size=256, shuffle=False)

    labels_np = labels.numpy()

    for alpha in DIR_ALPHAS:

        log(f"\n==== DATASET={ds_name} | α={alpha} ====\n")

        splits = dirichlet_split(labels_np, NUM_CLIENTS, alpha)

        # Global model
        global_model = ResNet18Pre(nc).to(device)
        trainable = [p for p in global_model.parameters() if p.requires_grad]

        # SCAFFOLD control variates
        c_global = [torch.zeros_like(p).cpu() for p in trainable]
        c_locals = [
            [torch.zeros_like(p).cpu() for p in trainable]
            for _ in range(NUM_CLIENTS)
        ]

        # Sequential rounds
        for rnd in range(1, NUM_ROUNDS + 1):

            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_DECAY

            # Freeze the starting global weights for all clients this round
            global_start = [p.detach().clone().cpu() for p in trainable]

            new_params_all = []
            delta_c_all = []

            for cid in range(NUM_CLIENTS):

                # Local model from SAME global start
                local_model = ResNet18Pre(nc).to(device)

                # load global start weights
                with torch.no_grad():
                    idx = 0
                    for p in local_model.parameters():
                        if p.requires_grad:
                            p.copy_(global_start[idx].to(device))
                            idx += 1

                # Train client
                new_params, delta_c, new_c_local = run_client(
                    local_model,
                    splits[cid],
                    data, labels,
                    c_global, c_locals[cid],
                    lr,
                    device
                )

                c_locals[cid] = new_c_local
                new_params_all.append(new_params)
                delta_c_all.append(delta_c)

            # Aggregate params
            avg_params = []
            for i in range(len(trainable)):
                stacked = torch.stack([client_params[i] for client_params in new_params_all], dim=0)
                avg_params.append(stacked.mean(0))

            # Update global model
            with torch.no_grad():
                idx = 0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[idx].to(device))
                        idx += 1

            # Update c_global
            for i in range(len(c_global)):
                stacked = torch.stack([dc[i] for dc in delta_c_all], dim=0)
                c_global[i] = stacked.mean(0)

            # Eval
            acc = evaluate(global_model, testloader, device)
            log(f"[ROUND {rnd}] ACC={acc*100:.2f}%")


# ==============================================================
# MAIN
# ==============================================================

def main():
    set_seed(SEED)
    federated_run("CIFAR10")


if __name__ == "__main__":
    main()












