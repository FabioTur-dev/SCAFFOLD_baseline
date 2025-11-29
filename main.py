#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MULTI-GPU PARALLEL-CLIENT FEDERATED LEARNING (SCAFFOLD-Lite)

Ogni GPU allena uno o più client in parallelo.
Scala perfettamente su RunC.ai con 2-8 GPU.

Include:
  - CIFAR-10, CIFAR-100, SVHN
  - α ∈ {0.5, 0.1, 0.05}
  - 100 federated rounds
  - ResNet18 pretrained + partial fine-tuning
  - SCAFFOLD-Lite
  - Real parallelism: ogni GPU = 1 client alla volta
"""

import os
import argparse
import random
import numpy as np
from queue import Empty
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# ========================================================================
# CONFIG
# ========================================================================
NUM_CLIENTS = 10
DIRICHLET_ALPHAS = [0.5, 0.1, 0.05]
NUM_ROUNDS = 100
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LR_INIT = 0.01
LR_DECAY_ROUND = 50
SEED = 42
BETA = 0.01                 # SCAFFOLD-Lite stable


# ========================================================================
# UTILITY
# ========================================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dirichlet_split(labels, n_clients, alpha):
    labels = np.array(labels)
    classes = np.unique(labels)
    per_client = [[] for _ in range(n_clients)]

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha] * n_clients)
        splits = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, splits[:-1])
        for i in range(n_clients):
            per_client[i].extend(chunks[i])

    for c in per_client:
        random.shuffle(c)

    return per_client


# ========================================================================
# MODEL
# ========================================================================
class ResNet18_Pretrained(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = self.model.fc.in_features
        self.model.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.model(x)


def freeze_except_deep(model):
    for name, p in model.model.named_parameters():
        if (name.startswith("layer3") or
            name.startswith("layer4") or
            name.startswith("fc")):
            p.requires_grad = True
        else:
            p.requires_grad = False


def get_trainable_params(model):
    return [p.detach().cpu().clone() for p in model.parameters()
            if p.requires_grad]


def set_trainable_params(model, params):
    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            p.data = params[idx].clone().to(p.device)
            idx += 1


def zero_like_trainable(model):
    return [torch.zeros_like(p).cpu() for p in model.parameters()
            if p.requires_grad]


# ========================================================================
# CLIENT UPDATE (SCAFFOLD-Lite)
# ========================================================================
def client_update(gpu_id, client_id, train_idx, trainset,
                  global_params, c_local, c_global, lr,
                  num_classes, result_queue):
    """
    Funzione eseguita su GPU worker.
    """
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    # Build model
    model = ResNet18_Pretrained(num_classes).to(device)
    freeze_except_deep(model)
    set_trainable_params(model, global_params)

    loader = DataLoader(
        Subset(trainset, train_idx),
        batch_size=BATCH_SIZE, shuffle=True)

    old_params = get_trainable_params(model)

    opt = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.9, weight_decay=5e-4
    )

    E = len(loader)

    # TRAIN
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()

            i = 0
            for p in model.parameters():
                if p.requires_grad:
                    p.grad += (c_global[i].to(device) -
                               c_local[i].to(device))
                    i += 1

            opt.step()

    new_params = get_trainable_params(model)

    # Update c_local
    delta_c = []
    for new, old, cl, cg in zip(new_params, old_params, c_local, c_global):
        diff = new - old
        dc = BETA * (diff / max(E, 1))
        delta_c.append(dc)
        cl += dc

    # Send result back to server
    result_queue.put((client_id, new_params, c_local, delta_c))


# ========================================================================
# EVALUATION
# ========================================================================
def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ========================================================================
# FEDERATED SERVER LOOP
# ========================================================================
def run_federated(dataset_name, gpus):

    print("\n=====================================")
    print(f"DATASET = {dataset_name}")
    print("=====================================\n")

    # ----------------------------
    # Load dataset
    # ----------------------------
    if dataset_name == "CIFAR10":
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=16),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),
                                 (0.229,0.224,0.225)),
            transforms.RandomErasing(p=0.25),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),
                                 (0.229,0.224,0.225)),
        ])
        trainset = datasets.CIFAR10("./data", train=True, download=True,
                                    transform=transform_train)
        testset  = datasets.CIFAR10("./data", train=False, download=True,
                                    transform=transform_test)
        labels = trainset.targets

    elif dataset_name == "CIFAR100":
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=16),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),
                                 (0.229,0.224,0.225)),
            transforms.RandomErasing(p=0.25),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),
                                 (0.229,0.224,0.225)),
        ])
        trainset = datasets.CIFAR100("./data", train=True, download=True,
                                     transform=transform_train)
        testset  = datasets.CIFAR100("./data", train=False, download=True,
                                     transform=transform_test)
        labels = trainset.targets

    elif dataset_name == "SVHN":
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=16),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),
                                 (0.229,0.224,0.225)),
            transforms.RandomErasing(p=0.25),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),
                                 (0.229,0.224,0.225)),
        ])
        trainset = datasets.SVHN("./data", split='train', download=True,
                                 transform=transform_train)
        testset  = datasets.SVHN("./data", split='test', download=True,
                                 transform=transform_test)
        labels = trainset.labels

    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    # =================================================================
    # LOOP SU α
    # =================================================================
    for alpha in DIRICHLET_ALPHAS:

        print(f"\n=== DATASET {dataset_name} | α={alpha} ===")

        splits = dirichlet_split(labels, NUM_CLIENTS, alpha)

        # global model
        device0 = "cuda:0"
        global_model = ResNet18_Pretrained(num_classes).to(device0)
        freeze_except_deep(global_model)
        global_params = get_trainable_params(global_model)

        c_global = zero_like_trainable(global_model)
        c_local = [zero_like_trainable(global_model) for _ in range(NUM_CLIENTS)]

        # multiprocessing queues
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        # spawn worker processes
        workers = []
        for gpu_id in range(gpus):
            p = ctx.Process(
                target=worker_loop,
                args=(gpu_id, task_queue, result_queue, trainset,
                      num_classes))
            p.start()
            workers.append(p)

        # ------------------------------
        # FEDERATED ROUNDS
        # ------------------------------
        for rnd in range(1, NUM_ROUNDS + 1):
            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_INIT * 0.1
            print(f"\n--- ROUND {rnd}/{NUM_ROUNDS} ---")

            # assign clients to task queue
            for cid in range(NUM_CLIENTS):
                task_queue.put(
                    ("run_client",
                     cid,
                     splits[cid],
                     global_params,
                     c_local[cid],
                     c_global,
                     lr)
                )

            # collect all results
            results_received = 0
            new_params_all = [None]*NUM_CLIENTS
            delta_c_sum = zero_like_trainable(global_model)

            while results_received < NUM_CLIENTS:
                try:
                    cid, new_params, new_c_local, delta_c = \
                        result_queue.get(timeout=9999)
                except Empty:
                    continue

                new_params_all[cid] = new_params
                c_local[cid] = new_c_local
                for i, dc in enumerate(delta_c):
                    delta_c_sum[i] += dc

                results_received += 1

            # aggregate average
            avg_params = []
            for i in range(len(global_params)):
                avg_params.append(torch.stack(
                    [p[i] for p in new_params_all]
                ).mean(dim=0))

            global_params = avg_params
            set_trainable_params(global_model, global_params)

            # update c_global
            for i in range(len(c_global)):
                c_global[i] += delta_c_sum[i] / NUM_CLIENTS

            # evaluate
            acc = evaluate(global_model, testloader, device0)
            print(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")

        # stop workers
        for _ in range(len(workers)):
            task_queue.put(("stop",))
        for p in workers:
            p.join()

        print(f"\n✔ FINISHED dataset={dataset_name}, alpha={alpha}\n")


# ========================================================================
# WORKER LOOP (ESEGUITO SU OGNI GPU)
# ========================================================================
def worker_loop(gpu_id, task_queue, result_queue, trainset, num_classes):
    while True:
        msg = task_queue.get()
        if msg[0] == "stop":
            return

        _, cid, train_idx, global_params, c_local, c_global, lr = msg

        client_update(
            gpu_id, cid, train_idx, trainset,
            global_params, c_local, c_global, lr,
            num_classes,
            result_queue
        )


# ========================================================================
# MAIN
# ========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    gpus = args.gpus
    set_seed(SEED)

    for ds in ["CIFAR10", "CIFAR100", "SVHN"]:
        run_federated(ds, gpus=gpus)


if __name__ == "__main__":
    main()
