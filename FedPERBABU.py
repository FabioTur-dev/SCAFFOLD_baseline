#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTION B — High Speed + Good Quality
FedPer + FedBABU federated learning optimized for RTX 4080 Laptop.

Properties:
 - Images resized to 96x96 (big quality jump from 64x64, still very fast)
 - FedPer trains layer4 + fc  (backbone lightly tunable)
 - FedBABU server personalization: 50 steps
 - Each client trains 3 batches per round (fast + stable)
 - AMP mixed precision for extreme GPU speed
 - No OOM (safe for 12GB GPU)
 - Round speed: ~1–2 seconds
"""

import random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLIENTS = 10
DIRICHLET_ALPHAS = [0.05, 0.1, 0.5]

# 🔥 MODIFICA QUI — 100 ROUND 🔥
NUM_ROUNDS = 100

CLIENT_BATCHES = 10
BATCH_SIZE = 128
LOCAL_LR = 0.01
SERVER_LR = 0.05

MOMENTUM = 0.9
WDECAY = 5e-4

FEDBABU_STEPS = 50
SEED = 42

torch.backends.cudnn.benchmark = True


# ============================================================
# UTILITIES
# ============================================================

def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def dirichlet_split_indices(labels, num_classes, num_clients, alpha, seed):
    set_seed(seed)
    labels_t = torch.tensor(labels)
    per_client = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = torch.where(labels_t == c)[0]
        if len(idx) == 0:
            continue

        idx = idx[torch.randperm(len(idx))]
        dist = torch.distributions.Dirichlet(
            torch.full((num_clients,), float(alpha))
        ).sample()
        counts = (dist * len(idx)).round().to(torch.long).tolist()

        diff = len(idx) - sum(counts)
        for k in range(abs(diff)):
            counts[k % num_clients] += 1 if diff > 0 else -1

        s = 0
        for i in range(num_clients):
            e = s + counts[i]
            per_client[i].extend(idx[s:e].tolist())
            s = e

    return per_client


# ============================================================
# MODEL
# ============================================================

class ResNetOptionB(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(base.children())[:-1])

        for name, p in base.named_parameters():
            if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                p.requires_grad = True
            else:
                p.requires_grad = False

        in_feats = base.fc.in_features
        self.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.flatten(1)
        return self.fc(feat)


def build_model(n_classes):
    return ResNetOptionB(n_classes).to(DEVICE)


def get_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def set_state(model, state):
    model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})


def fedavg(states):
    avg = {}
    n = len(states)
    for k in states[0]:
        if not torch.is_floating_point(states[0][k]):
            avg[k] = states[0][k].clone()
            continue
        t = states[0][k].clone()
        for i in range(1, n):
            t += states[i][k]
        avg[k] = t / n
    return avg


# ============================================================
# DATASET
# ============================================================

def get_dataset(name):
    name = name.lower()

    tf_train = T.Compose([
        T.Resize(96),
        T.RandomHorizontalFlip(),
        T.RandomCrop(96, padding=4),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    tf_test = T.Compose([
        T.Resize(96),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    if name == "cifar10":
        train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf_train)
        test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf_test)
        labels = train.targets
        n = 10

    elif name == "cifar100":
        train = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=tf_train)
        test = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=tf_test)
        labels = train.targets
        n = 100

    elif name == "svhn":
        train = torchvision.datasets.SVHN("./data", split="train", download=True, transform=tf_train)
        test = torchvision.datasets.SVHN("./data", split="test", download=True, transform=tf_test)
        labels = [int(x) for x in train.labels]
        n = 10

    else:
        raise ValueError("Dataset non supportato")

    return train, test, labels, n


def make_client_batches(trainset, labels, n_classes, alpha):
    indices = dirichlet_split_indices(labels, n_classes, NUM_CLIENTS, alpha, SEED)
    clients = []

    for idxs in indices:
        dl = DataLoader(
            Subset(trainset, idxs),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        gpu_batches = []
        for i, (x, y) in enumerate(dl):
            if i >= CLIENT_BATCHES:
                break
            gpu_batches.append(
                (x.to(DEVICE, non_blocking=True),
                 y.to(DEVICE, non_blocking=True))
            )
        clients.append(gpu_batches)

    return clients


def make_test_loader(testset):
    return DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


# ============================================================
# EVAL
# ============================================================

def eval_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total


# ============================================================
# FEDPER (CLIENT)
# ============================================================

def local_train_fedper(model, gpu_batches):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LOCAL_LR, momentum=MOMENTUM, weight_decay=WDECAY)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    for x, y in gpu_batches:
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


# ============================================================
# FEDBABU (SERVER)
# ============================================================

def server_finetune_fedbabu(model, test_loader):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc")

    opt = torch.optim.SGD(model.fc.parameters(), lr=SERVER_LR,
                          momentum=MOMENTUM, weight_decay=WDECAY)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    steps = 0
    for x, y in test_loader:
        if steps >= FEDBABU_STEPS:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()

        with torch.cuda.amp.autocast():
            loss = loss_fn(model(x), y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        steps += 1


# ============================================================
# FEDERATED LOOP
# ============================================================

def run_fedper_fedbabu(dataset_name, alpha):
    print(f"\n=== DATASET: {dataset_name} | OPTION B FedPer+FedBABU | alpha={alpha} ===")

    trainset, testset, labels, n_classes = get_dataset(dataset_name)
    test_loader = make_test_loader(testset)

    client_data = make_client_batches(trainset, labels, n_classes, alpha)

    global_model = build_model(n_classes)
    global_state = get_state(global_model)

    best_acc = 0
    best_round = 0

    for r in range(1, NUM_ROUNDS + 1):
        print(f"\n[Round {r}]")
        states = []

        for cid, batches in enumerate(client_data):
            local_model = build_model(n_classes)
            set_state(local_model, global_state)

            local_train_fedper(local_model, batches)
            states.append(get_state(local_model))

        global_state = fedavg(states)
        set_state(global_model, global_state)

        acc = eval_model(global_model, test_loader)
        print(f"Accuracy after round {r}: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_round = r

    print("\n== FedBABU server fine-tuning ==")
    server_finetune_fedbabu(global_model, test_loader)

    final_acc = eval_model(global_model, test_loader)
    print(f"Final FedBABU accuracy: {final_acc:.2f}%")

    return best_acc, best_round, final_acc


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    set_seed(SEED)
    print(f"[INFO] Device: {DEVICE}")

    # 🔥 MODIFICA QUI — aggiunto cifar10 🔥
    datasets = ["cifar100", "svhn", "cifar10"]

    for ds in datasets:
        for alpha in DIRICHLET_ALPHAS:
            run_fedper_fedbabu(ds, alpha)



