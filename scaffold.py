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
DIR_ALPHAS = [0.1, 0.5]
NUM_ROUNDS = 20
LOCAL_EPOCHS = 2
BATCH = 128

LR_INIT = 0.01
LR_DECAY_ROUND = 15
LR_DECAY = 0.003

BETA = 0.01
DAMPING = 0.1
GRAD_CLIP = 5.0
SEED = 42

IMG_SIZE = 160


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

    for cid in range(n_clients):
        random.shuffle(per[cid])

    return per


# ======================================================
# RESNET18 PRETRAINED (download automatico)
# only layer3 + layer4 + fc are trainable
# ======================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()

        # Compatibile con tutte le versioni di torchvision
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


# ======================================================
# LOCAL TRAINING
# ======================================================

def run_client(model, idxs, X_cpu, y_cpu, c_global, c_local, lr, device):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    old_params = [p.detach().clone() for p in params]

    opt = optim.SGD(params, lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    for _ in range(LOCAL_EPOCHS):
        for start in range(0, len(idxs), BATCH):
            batch = idxs[start:start+BATCH]

            xb = X_cpu[batch].to(device, dtype=torch.float32) / 255.0
            yb = y_cpu[batch].to(device)

            xb = F.interpolate(xb, size=IMG_SIZE, mode="bilinear", align_corners=False)
            xb = (xb - mean) / std
            xb = gpu_augment(xb)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()

            # SCAFFOLD correction
            for i, p in enumerate(params):
                p.grad += DAMPING * (c_global[i] - c_local[i])

            torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()

    # compute Δc
    new_params = [p.detach().clone() for p in params]
    delta_c = []
    E = max(1, len(idxs) // BATCH)

    for i in range(len(params)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / E)
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, delta_c, c_local


# ======================================================
# EVALUATION
# ======================================================

def evaluate(model, X_cpu, y_cpu, device):
    model.eval()
    correct = 0
    tot = len(y_cpu)

    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1,3,1,1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1,3,1,1)

    with torch.no_grad():
        for start in range(0, tot, 256):
            xb = X_cpu[start:start+256].to(device, dtype=torch.float32) / 255.0
            yb = y_cpu[start:start+256].to(device)

            xb = F.interpolate(xb, size=IMG_SIZE, mode="bilinear", align_corners=False)
            xb = (xb - mean) / std

            out = model(xb)
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()

    return correct / tot


# ======================================================
# FEDERATED LOOP
# ======================================================

def federated_run():
    device = "cuda:0"

    log("Carico CIFAR-100 (CPU, 32x32)...")

    transform = transforms.ToTensor()
    trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

    X_train = (trainset.data.transpose(0,3,1,2)).copy()
    X_test = (testset.data.transpose(0,3,1,2)).copy()

    X_train = torch.from_numpy(X_train).to(torch.uint8)
    X_test = torch.from_numpy(X_test).to(torch.uint8)

    y_train = torch.tensor(trainset.targets, dtype=torch.long)
    y_test = torch.tensor(testset.targets, dtype=torch.long)

    for alpha in DIR_ALPHAS:
        log(f"\n==== CIFAR-100 | α={alpha} ====\n")

        splits = dirichlet_split(y_train, NUM_CLIENTS, alpha)

        global_model = ResNet18Pre(100).to(device)
        params = [p for p in global_model.parameters() if p.requires_grad]

        c_global = [torch.zeros_like(p) for p in params]
        c_local = [[torch.zeros_like(p) for p in params] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS+1):
            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_DECAY

            global_snapshot = [p.detach().clone() for p in params]

            newP_all = []
            dC_all = []

            for cid in range(NUM_CLIENTS):
                local_model = ResNet18Pre(100).to(device)

                with torch.no_grad():
                    j = 0
                    for p in local_model.parameters():
                        if p.requires_grad:
                            p.copy_(global_snapshot[j])
                            j += 1

                newP, dC, cl = run_client(local_model, splits[cid], X_train, y_train,
                                          c_global, c_local[cid], lr, device)

                c_local[cid] = cl
                newP_all.append(newP)
                dC_all.append(dC)

            # federated averaging
            avgP = []
            for i in range(len(params)):
                avgP.append(torch.stack([cp[i] for cp in newP_all]).mean(0))

            with torch.no_grad():
                j = 0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avgP[j])
                        j += 1

            # update global control variate
            for i in range(len(params)):
                c_global[i] = torch.stack([dc[i] for dc in dC_all]).mean(0)

            acc = evaluate(global_model, X_test, y_test, device)
            log(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")



def main():
    set_seed(SEED)
    federated_run()


if __name__ == "__main__":
    main()













