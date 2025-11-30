#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import to_pil_image

# ==============================================================
# CONFIG — FAST Sequential SCAFFOLD
# ==============================================================

NUM_CLIENTS = 10
NUM_ROUNDS = 50
LOCAL_EPOCHS = 2
BATCH = 128

LR_INIT = 0.01        # più alto per accelerare i primi round
LR_AFTER = 0.003      # decay dopo round 15
LR_DECAY_ROUND = 15

BETA = 0.01
DAMPING = 0.1
GRAD_CLIP = 5.0

SEED = 42

# ==============================================================

def log(msg):
    print(msg, flush=True)

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

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
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.Resize(160),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),
                                     (0.229,0.224,0.225)),
                transforms.RandomErasing(p=0.35, scale=(0.02,0.2)),
            ])
        else:
            self.T = transforms.Compose([
                transforms.Resize(160),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),
                                     (0.229,0.224,0.225)),
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = to_pil_image(self.data[idx])
        return self.T(img), self.labels[idx]

# ==============================================================

def dirichlet_split(labels, n_clients, alpha):
    labels = np.array(labels)
    per = [[] for _ in range(n_clients)]
    classes = np.unique(labels)

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        dist = np.random.dirichlet([alpha]*n_clients)
        cuts = (np.cumsum(dist)*len(idx)).astype(int)
        splits = np.split(idx, cuts[:-1])
        for cid in range(n_clients):
            per[cid].extend(splits[cid])

    for cid in range(n_clients):
        random.shuffle(per[cid])

    return per

# ==============================================================

class ResNet18Partial(nn.Module):
    def __init__(self, nc):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights
            w = ResNet18_Weights.IMAGENET1K_V1
            self.m = models.resnet18(weights=w)
        except:
            self.m = models.resnet18(pretrained=True)

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        # Unfreeze layer3 + layer4 + fc
        for name, p in self.m.named_parameters():
            if ("layer3" in name) or ("layer4" in name) or ("fc" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, x):
        return self.m(x)

# ==============================================================

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    tot = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            tot += len(y)
    return correct/tot

# ==============================================================

def federated_run(ds_name):

    # ---------------------------------------------
    # Load raw dataset once
    # ---------------------------------------------
    if ds_name == "CIFAR10":
        d = datasets.CIFAR10("./data", train=True, download=True)
        X = torch.tensor(d.data).permute(0,3,1,2)
        y = torch.tensor(d.targets)
        nc = 10
    else:
        raise ValueError("Only CIFAR10 implemented in this FAST version.")

    # ---------------------------------------------
    # Precompute test loader
    # ---------------------------------------------
    test_tf = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225)),
    ])
    te = datasets.CIFAR10("./data", train=False, download=True,
                          transform=test_tf)
    testloader = DataLoader(te, batch_size=256, shuffle=False)

    # ---------------------------------------------
    # Dirichlet split
    # ---------------------------------------------
    splits = dirichlet_split(y.numpy(), NUM_CLIENTS, alpha=0.5)

    # ---------------------------------------------
    # Pre-create client DataLoaders (FAST)
    # ---------------------------------------------
    client_loaders = []
    for cid in range(NUM_CLIENTS):
        ds = RawDataset(X, y, splits[cid], augment=True)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                        num_workers=2, pin_memory=True)
        client_loaders.append(dl)

    # ---------------------------------------------
    # Global model
    # ---------------------------------------------
    device = "cuda"
    global_model = ResNet18Partial(nc).to(device)
    trainable = [p for p in global_model.parameters() if p.requires_grad]

    # ---------------------------------------------
    # SCAFFOLD control variates
    # ---------------------------------------------
    c_global = [torch.zeros_like(p).to(device) for p in trainable]
    c_local = [
        [torch.zeros_like(p).to(device) for p in trainable]
        for _ in range(NUM_CLIENTS)
    ]

    # ==============================================================
    # MAIN FEDERATED LOOP — SEQUENTIAL FAST
    # ==============================================================

    for rnd in range(1, NUM_ROUNDS+1):

        lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_AFTER
        log(f"[ROUND {rnd}] lr={lr}")

        agg = [torch.zeros_like(p) for p in trainable]
        delta_c_sum = [torch.zeros_like(p) for p in trainable]

        # -----------------------------------------
        # TRAIN EACH CLIENT SEQUENTIALLY
        # -----------------------------------------
        for cid in range(NUM_CLIENTS):

            # copy global weights into a single reusable model
            with torch.no_grad():
                idx = 0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_( trainable[idx].detach() )
                        idx += 1

            loader = client_loaders[cid]
            opt = optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=5e-4)
            loss_fn = nn.CrossEntropyLoss()

            old_params = [p.detach().clone() for p in trainable]

            for _ in range(LOCAL_EPOCHS):
                for xb,yb in loader:
                    xb,yb = xb.to(device), yb.to(device)

                    opt.zero_grad()
                    out = global_model(xb)
                    loss = loss_fn(out, yb)
                    loss.backward()

                    # SCAFFOLD correction
                    for i,p in enumerate(trainable):
                        p.grad += DAMPING * (c_global[i] - c_local[cid][i])

                    torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                    opt.step()

            # compute updates
            new_params = [p.detach().clone() for p in trainable]

            # control variate update
            E = len(loader)
            for i in range(len(trainable)):
                diff = new_params[i] - old_params[i]
                dc = BETA * (diff / max(E,1))
                c_local[cid][i] += dc
                delta_c_sum[i] += dc
                agg[i] += new_params[i]

        # -----------------------------------------
        # Server aggregation
        # -----------------------------------------
        for i in range(len(trainable)):
            trainable[i].data.copy_( agg[i] / NUM_CLIENTS )
            c_global[i] = delta_c_sum[i] / NUM_CLIENTS

        # -----------------------------------------
        # Evaluate
        # -----------------------------------------
        acc = evaluate(global_model, testloader, device)
        log(f"ACC={acc*100:.2f}%")

# ==============================================================

def main():
    set_seed(SEED)
    federated_run("CIFAR10")

if __name__ == "__main__":
    main()












