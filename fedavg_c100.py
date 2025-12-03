#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

# ======================================================
# CONFIG
# ======================================================
NUM_CLIENTS = 10
ALPHAS = [0.5, 0.1, 0.05]
LOCAL_EPOCHS = 1
BATCH = 256
ROUNDS = 50
LR = 0.001
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# UTILITIES
# ======================================================
def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ======================================================
# MODEL — ResNet18 PRETRAINED (NO MODIFICHE)
# ======================================================
class ResNet18_C100_FULL(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(pretrained=True)

        # testina CIFAR-100
        self.model.fc = nn.Linear(512, 100)

    def forward(self, x):
        return self.model(x)

# ======================================================
# LOCAL TRAIN
# ======================================================
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

def local_train(local_model, loader):
    local_model.train()
    opt = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = loss_fn(local_model(x), y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    return local_model.state_dict()

# ======================================================
# FEDAVG PESATO (VERSIONE CORRETTA)
# ======================================================
def fedavg_weighted(states, client_sizes):
    avg = {}
    total = sum(client_sizes)

    with torch.no_grad():
        for k in states[0]:
            tensors = [s[k].to(DEVICE) for s in states]

            # solo se è un tensore float
            if tensors[0].dtype in [torch.float16, torch.float32, torch.float64]:
                weighted_sum = sum((client_sizes[i] / total) * tensors[i]
                                   for i in range(len(tensors)))
                avg[k] = weighted_sum
            else:
                # tensori non float: copia il primo
                avg[k] = tensors[0].clone()

    return avg

# ======================================================
# EVALUATION
# ======================================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# ======================================================
# DIRICHLET SPLIT
# ======================================================
def dirichlet_split(labels, num_clients, alpha):
    labels = np.array(labels)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)

        props = np.random.dirichlet([alpha] * num_clients)
        props = (props * len(idx)).astype(int)

        while props.sum() < len(idx):
            props[np.argmax(props)] += 1

        start = 0
        for i in range(num_clients):
            end = start + props[i]
            client_indices[i].extend(idx[start:end])
            start = end

    return client_indices

# ======================================================
# MAIN
# ======================================================
def main():
    seed_everything(SEED)

    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        )
    ])

    train_raw = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
    test_raw  = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

    testloader = DataLoader(test_raw, batch_size=BATCH, shuffle=False,
                            num_workers=2, pin_memory=True)

    labels_train = train_raw.targets

    for ALPHA in ALPHAS:
        print("\n============================")
        print(f"=== Dirichlet alpha = {ALPHA} ===")
        print("============================")

        global_model = ResNet18_C100_FULL().to(DEVICE)

        # split clienti
        client_indices = dirichlet_split(labels_train, NUM_CLIENTS, ALPHA)
        client_sizes = [len(idx) for idx in client_indices]  # 👈 IMPORTANTISSIMO

        for rnd in range(1, ROUNDS + 1):
            local_states = []

            for cid in range(NUM_CLIENTS):
                subset = Subset(train_raw, client_indices[cid])
                loader = DataLoader(subset, batch_size=BATCH, shuffle=True,
                                    num_workers=2, pin_memory=True)

                local_model = ResNet18_C100_FULL().to(DEVICE)
                local_model.load_state_dict(global_model.state_dict(), strict=True)

                state = local_train(local_model, loader)
                local_states.append(state)

            # FedAvg pesato ⚡
            new_state = fedavg_weighted(local_states, client_sizes)

            global_model.load_state_dict(new_state)

            acc = evaluate(global_model, testloader)
            print(f"[ALPHA {ALPHA}][ROUND {rnd}] ACC = {acc:.2f}%")

# ======================================================
# ENTRYPOINT
# ======================================================
if __name__ == "__main__":
    main()
