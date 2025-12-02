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
ALPHAS = [0.5, 0.1, 0.05]   # una o più alpha
LOCAL_EPOCHS = 1
BATCH = 256
ROUNDS = 25
LR = 0.001
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def main():

    seed_everything(SEED)

    # ======================================================
    # TRANSFORMS
    # ======================================================
    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # standard ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ======================================================
    # LOAD SVHN
    # ======================================================
    trainset = datasets.SVHN(root="./data", split='train',
                              download=True, transform=transform)
    testset  = datasets.SVHN(root="./data", split='test',
                              download=True, transform=transform)

    testloader = DataLoader(
        testset, batch_size=256, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ======================================================
    # DIRICHLET SPLIT (usando dataset.labels)
    # ======================================================
    def dirichlet_split(dataset, num_clients, alpha):
        labels = np.array(dataset.labels)
        num_classes = 10
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
    # MODEL FACTORY
    # ======================================================
    def build_resnet18():
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)
        return model

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
    # FEDAVG
    # ======================================================
    def fedavg(states):

        states_float = [
            {k: v.float() if v.dtype in (torch.int64, torch.long) else v
             for k, v in st.items()}
            for st in states
        ]

        avg = {}
        with torch.no_grad():
            for k in states_float[0].keys():
                stacked = torch.stack([s[k] for s in states_float], dim=0).to(DEVICE)
                avg[k] = stacked.mean(dim=0)

        return avg

    # ======================================================
    # EVALUATION
    # ======================================================
    def evaluate(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for x, y in testloader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return 100 * correct / total

    # ======================================================
    # LOOP SU ALPHA
    # ======================================================
    for ALPHA in ALPHAS:
        print(f"\n============================")
        print(f"=== Dirichlet alpha = {ALPHA} ===")
        print(f"============================\n")

        # Dirichlet split SVHN
        client_indices = dirichlet_split(trainset, NUM_CLIENTS, ALPHA)

        # modello globale
        global_model = build_resnet18().to(DEVICE)

        # FEDAVG main loop
        for rnd in range(1, ROUNDS + 1):
            local_states = []

            for cid in range(NUM_CLIENTS):
                subset = Subset(trainset, client_indices[cid])
                loader = DataLoader(
                    subset,
                    batch_size=BATCH,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True
                )

                local_model = build_resnet18().to(DEVICE)
                local_model.load_state_dict(global_model.state_dict(), strict=True)

                st = local_train(local_model, loader)
                local_states.append(st)

            new_state = fedavg(local_states)
            global_model.load_state_dict(new_state)

            acc = evaluate(global_model)
            print(f"[ALPHA {ALPHA}][ROUND {rnd}] ACC = {acc:.2f}%")

# ======================================================
# ENTRYPOINT
# ======================================================
if __name__ == "__main__":
    main()
