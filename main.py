#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# ============================================================
# LOGGING (debug → stderr, accuracy → stdout)
# ============================================================

DEBUG = True

def log_debug(msg):
    if DEBUG:
        print(msg, file=sys.stderr, flush=True)

def log_accuracy(msg):
    print(msg, file=sys.stdout, flush=True)


# ============================================================
# CONFIG
# ============================================================

NUM_CLIENTS = 10
DIRICHLET_ALPHAS = [0.5, 0.1, 0.05]
NUM_ROUNDS = 100
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LR_INIT = 0.01
LR_DECAY_ROUND = 50
BETA = 0.01
SEED = 42


# ============================================================
# UTILS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dirichlet_split(labels, n_clients, alpha):
    labels = np.array(labels)
    per_client = [[] for _ in range(n_clients)]
    classes = np.unique(labels)

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for i in range(n_clients):
            per_client[i].extend(chunks[i])

    for c in per_client:
        random.shuffle(c)

    return per_client


# ============================================================
# MODEL
# ============================================================

class ResNet18_Pretrained(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        log_debug(f"[MODEL] Creating ResNet18 with {num_classes} classes")
        try:
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1
            )
            log_debug("[MODEL] Loaded IMAGENET1K_V1 weights")
        except Exception:
            self.model = models.resnet18(pretrained=True)
            log_debug("[MODEL] Loaded pretrained=True (legacy API)")

        in_f = self.model.fc.in_features
        self.model.fc = nn.Linear(in_f, num_classes)

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)


# ============================================================
# CLIENT UPDATE
# ============================================================

def client_update(model, device, train_idx, trainset,
                  c_local, c_global, lr):

    log_debug(f"[CLIENT_UPDATE] Start | device={device} | lr={lr} | n_samples={len(train_idx)}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    old_params = [p.detach().clone().cpu() for p in trainable]

    loader = DataLoader(
        Subset(trainset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    opt = optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=5e-4)
    E = len(loader)
    log_debug(f"[CLIENT_UPDATE] Dataloader length E={E}")

    for ep in range(LOCAL_EPOCHS):
        log_debug(f"[CLIENT_UPDATE] Epoch {ep+1}/{LOCAL_EPOCHS}")
        for it, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()

            for i, p in enumerate(trainable):
                p.grad += (c_global[i].to(device) - c_local[i].to(device))

            opt.step()

            if DEBUG and it % 50 == 0:
                log_debug(f"[CLIENT_UPDATE] Iter {it}, loss={loss.item():.4f}")

    new_params = [p.detach().clone().cpu() for p in trainable]

    delta_c = []
    for i in range(len(new_params)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / max(E, 1))
        delta_c.append(dc)
        c_local[i] += dc

    log_debug("[CLIENT_UPDATE] Done")
    return new_params, c_local, delta_c


# ============================================================
# WORKER LOOP
# ============================================================

def worker_loop(gpu_id, job_q, res_q, trainset, num_classes):

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    log_debug(f"[WORKER {gpu_id}] Started on {device}")

    model = ResNet18_Pretrained(num_classes).to(device)
    log_debug(f"[WORKER {gpu_id}] Model ready")

    while True:
        log_debug(f"[WORKER {gpu_id}] Waiting for job...")
        job = job_q.get()
        log_debug(f"[WORKER {gpu_id}] Job received: {job[0]}")

        if job[0] == "STOP":
            log_debug(f"[WORKER {gpu_id}] STOP received → exit")
            return

        _, cid, train_idx, global_sd, c_local, c_global, lr = job

        log_debug(f"[WORKER {gpu_id}] Loading global weights for client {cid}")
        model.load_state_dict(global_sd)

        log_debug(f"[WORKER {gpu_id}] Running update for client {cid}")
        new_params, new_c_local, delta_c = client_update(
            model, device, train_idx, trainset, c_local, c_global, lr
        )

        log_debug(f"[WORKER {gpu_id}] Sending results for client {cid}")
        res_q.put((cid, new_params, new_c_local, delta_c))


# ============================================================
# DATASET
# ============================================================

def load_dataset(name):
    log_debug(f"[DATASET] Loading {name}")

    if name == "CIFAR10":
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=16),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        train = datasets.CIFAR10("./data", train=True, download=True,
                                 transform=transform_train)
        test = datasets.CIFAR10("./data", train=False, download=True,
                                transform=transform_test)
        labels = train.targets

    elif name == "CIFAR100":
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224, padding=16),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        train = datasets.CIFAR100("./data", train=True, download=True,
                                  transform=transform_train)
        test = datasets.CIFAR100("./data", train=False, download=True,
                                 transform=transform_test)
        labels = train.targets

    else:
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        train = datasets.SVHN("./data", split="train", download=True,
                              transform=transform_train)
        test = datasets.SVHN("./data", split="test", download=True,
                             transform=transform_test)
        labels = train.labels

    log_debug(f"[DATASET] Loaded {name} | train={len(train)}, test={len(test)}")
    return train, test, labels, num_classes


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ============================================================
# FEDERATED TRAINING LOOP
# ============================================================

def federated_run(dataset_name, gpus):

    print(f"\n========== DATASET: {dataset_name} ==========\n", flush=True)

    log_debug("[FED] Loading dataset")
    trainset, testset, labels, num_classes = load_dataset(dataset_name)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    for alpha in DIRICHLET_ALPHAS:

        print(f"\n==== α = {alpha} ====\n", flush=True)

        log_debug(f"[FED] Dirichlet split α={alpha}")
        splits = dirichlet_split(labels, NUM_CLIENTS, alpha)

        device0 = "cuda:0"
        log_debug("[FED] Creating global model")
        global_model = ResNet18_Pretrained(num_classes).to(device0)
        global_sd = global_model.state_dict()

        trainable_params = [p for p in global_model.parameters() if p.requires_grad]

        log_debug("[FED] Initializing SCAFFOLD buffers")
        c_global = [torch.zeros_like(p).cpu() for p in trainable_params]
        c_local = [[torch.zeros_like(p).cpu() for p in trainable_params]
                   for _ in range(NUM_CLIENTS)]

        ctx = mp.get_context("fork")
        job_q = ctx.Queue()
        res_q = ctx.Queue()

        log_debug(f"[FED] Starting {gpus} workers")
        workers = []
        for gpu in range(gpus):
            p = ctx.Process(
                target=worker_loop,
                args=(gpu, job_q, res_q, trainset, num_classes)
            )
            p.start()
            workers.append(p)
        log_debug("[FED] Workers started")

        # ROUNDS
        for rnd in range(1, NUM_ROUNDS + 1):

            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_INIT * 0.1
            print(f"--- ROUND {rnd}/{NUM_ROUNDS} ---", flush=True)
            log_debug(f"[FED] ROUND {rnd} | lr={lr}")

            log_debug("[FED] Preparing global state dict")
            global_sd_cpu = {k: v.cpu() for k, v in global_sd.items()}

            log_debug("[FED] Enqueue client jobs")
            for cid in range(NUM_CLIENTS):
                job_q.put((
                    "RUN",
                    cid,
                    splits[cid],
                    global_sd_cpu,
                    c_local[cid],
                    c_global,
                    lr
                ))
                log_debug(f"[FED] Enqueued client {cid}")

            log_debug("[FED] Collecting client results")
            new_params_accum = None

            for it in range(NUM_CLIENTS):
                log_debug(f"[FED] Waiting result {it+1}/{NUM_CLIENTS}")
                cid, new_params, new_c_local, delta_c = res_q.get()
                log_debug(f"[FED] Received result for client {cid}")
                c_local[cid] = new_c_local

                if new_params_accum is None:
                    new_params_accum = [torch.zeros_like(p) for p in new_params]

                for i in range(len(new_params)):
                    new_params_accum[i] += new_params[i]

                for i in range(len(c_global)):
                    c_global[i] += delta_c[i] / NUM_CLIENTS

            log_debug("[FED] Aggregating results")
            avg_params = [p / NUM_CLIENTS for p in new_params_accum]

            idx = 0
            with torch.no_grad():
                for name, param in global_model.named_parameters():
                    if param.requires_grad:
                        param.copy_(avg_params[idx].to(param.device))
                        idx += 1

            global_sd = global_model.state_dict()

            acc = evaluate(global_model, testloader, device0)
            log_accuracy(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")

        log_debug("[FED] Stopping workers")
        for _ in range(gpus):
            job_q.put(("STOP",))
        for p in workers:
            p.join()
        log_debug("[FED] Workers joined")


# ============================================================
# MAIN
# ============================================================

def main():
    log_debug("[MAIN] Starting")
    set_seed(SEED)

    mp.set_start_method("fork", force=True)
    log_debug("[MAIN] mp start_method = fork")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    log_debug(f"[MAIN] Using {args.gpus} GPUs")

    for ds in ["CIFAR10", "CIFAR100", "SVHN"]:
        federated_run(ds, args.gpus)

    log_debug("[MAIN] Finished all datasets")


if __name__ == "__main__":
    main()



