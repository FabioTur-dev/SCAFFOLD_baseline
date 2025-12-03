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
ALPHAS = [0.05]
LOCAL_EPOCHS = 1
BATCH = 256
ROUNDS = 50
LR = 0.001
SEED = 42
FEDPROX_MU = 0.01  # coefficiente μ di FedProx

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
# MODEL 32×32 FRIENDLY RESNET-18
# ======================================================
class ResNet18_SVHN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.model(x)

# ======================================================
# LOCAL TRAIN (FedProx)
# ======================================================
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

def local_train_fedprox(local_model, global_model, loader, mu=FEDPROX_MU):
    local_model.train()
    opt = optim.SGD(local_model.parameters(), lr=LR, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    # snapshot dei pesi globali per il termine di prossimità
    global_params = {
        name: p.detach().clone()
        for name, p in global_model.named_parameters()
    }

    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                preds = local_model(x)
                loss = loss_fn(preds, y)

                # ----- termine proximal FedProx -----
                prox = 0.0
                for name, param in local_model.named_parameters():
                    prox += ((param - global_params[name].to(DEVICE)) ** 2).sum()

                loss = loss + (mu / 2.0) * prox

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    return local_model.state_dict()

# ======================================================
# FEDAVG PESATO (CORRETTO)
# ======================================================
def fedavg_weighted(states, client_sizes):
    avg = {}
    total = sum(client_sizes)

    with torch.no_grad():
        for k in states[0]:
            tensors = [s[k].to(DEVICE) for s in states]

            if tensors[0].dtype in [torch.float16, torch.float32, torch.float64]:
                weighted = sum(
                    (client_sizes[i] / total) * tensors[i]
                    for i in range(len(states))
                )
                avg[k] = weighted
            else:
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
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4377, 0.4438, 0.4728),
            (0.1980, 0.2010, 0.1970)
        )
    ])

    train_raw = datasets.SVHN("./data", split="train", download=True, transform=transform)
    test_raw  = datasets.SVHN("./data", split="test",  download=True, transform=transform)

    testloader = DataLoader(test_raw, batch_size=BATCH, shuffle=False,
                            num_workers=2, pin_memory=True)

    labels_train = train_raw.labels

    for ALPHA in ALPHAS:
        print("\n============================")
        print(f"=== Dirichlet alpha = {ALPHA} (FedProx μ={FEDPROX_MU}) ===")
        print("============================")

        global_model = ResNet18_SVHN().to(DEVICE)

        client_indices = dirichlet_split(labels_train, NUM_CLIENTS, ALPHA)
        client_sizes = [len(idx) for idx in client_indices]

        for rnd in range(1, ROUNDS + 1):
            local_states = []

            for cid in range(NUM_CLIENTS):
                subset = Subset(train_raw, client_indices[cid])
                loader = DataLoader(subset, batch_size=BATCH, shuffle=True,
                                    num_workers=2, pin_memory=True)

                local_model = ResNet18_SVHN().to(DEVICE)
                local_model.load_state_dict(global_model.state_dict(), strict=True)

                # FedProx local training
                state = local_train_fedprox(local_model, global_model, loader)
                local_states.append(state)

            # Aggregazione FedAvg pesata
            new_state = fedavg_weighted(local_states, client_sizes)
            global_model.load_state_dict(new_state)

            acc = evaluate(global_model, testloader)
            print(f"[ALPHA {ALPHA}][ROUND {rnd}] ACC = {acc:.2f}%")

# ======================================================
# ENTRYPOINT
# ======================================================
if __name__ == "__main__":
    main()
