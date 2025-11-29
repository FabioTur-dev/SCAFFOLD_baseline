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
from multiprocessing import shared_memory

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


# ============================================================
# LOGGING
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
            log_debug("[MODEL] Loaded pretrained=True (legacy)")

        in_f = self.model.fc.in_features
        self.model.fc = nn.Linear(in_f, num_classes)

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)


# ============================================================
# SHARED MEMORY HANDLERS
# ============================================================

def create_shared_tensor(t):
    """
    Create a shared memory block for a tensor t (CPU float32).
    """
    flat = t.contiguous().view(-1).numpy()
    shm = shared_memory.SharedMemory(create=True, size=flat.nbytes)
    np_sh = np.ndarray(flat.shape, dtype=flat.dtype, buffer=shm.buf)
    np_sh[:] = flat[:]
    return shm, flat.shape, t.shape


def read_shared_tensor(shm_name, flat_shape, real_shape):
    """
    Read a shared tensor from shared memory.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    np_arr = np.ndarray(flat_shape, dtype=np.float32, buffer=shm.buf)
    t = torch.from_numpy(np_arr.copy()).view(real_shape)
    return t


def update_shared_tensor(shm_name, flat_shape, t):
    """
    Write an updated tensor t back to the shared memory block.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    np_arr = np.ndarray(flat_shape, dtype=np.float32, buffer=shm.buf)
    flat_new = t.contiguous().view(-1).numpy()
    np_arr[:] = flat_new[:]


# ============================================================
# CLIENT UPDATE
# ============================================================

def client_update(model, device, train_idx, trainset,
                  c_local, c_global, lr):

    log_debug(f"[CLIENT_UPDATE] lr={lr} | samples={len(train_idx)}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    old_params = [p.detach().clone().cpu() for p in trainable]

    loader = DataLoader(
        Subset(trainset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    opt = optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=5e-4)
    E = len(loader)

    for ep in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            loss.backward()

            for i, p in enumerate(trainable):
                p.grad += (c_global[i].to(device) - c_local[i].to(device))

            opt.step()

    new_params = [p.detach().clone().cpu() for p in trainable]

    delta_c = []
    for i in range(len(new_params)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / max(E, 1))
        delta_c.append(dc)
        c_local[i] += dc

    return new_params, c_local, delta_c


# ============================================================
# WORKER PROCESS
# ============================================================

def worker_loop(gpu_id, job_q, res_q, trainset, num_classes,
                shm_info_list):

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    log_debug(f"[WORKER {gpu_id}] Started on {device}")

    # Load model ONCE
    model = ResNet18_Pretrained(num_classes).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]

    while True:
        msg = job_q.get()

        if msg[0] == "STOP":
            log_debug(f"[WORKER {gpu_id}] STOP received")
            return

        _, cid, train_idx, c_local, c_global, lr = msg

        # --- Load global params from shared memory ---
        for i, (shm_name, flat_shape, real_shape) in enumerate(shm_info_list):
            w = read_shared_tensor(shm_name, flat_shape, real_shape)
            trainable[i].data.copy_(w.to(device))

        # --- Update client ---
        new_params, new_c_local, delta_c = client_update(
            model, device, train_idx,
            trainset,
            c_local, c_global, lr
        )

        # --- Return result ---
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

    log_debug(f"[DATASET] Loaded {name}, train={len(train)}, test={len(test)}")
    return train, test, labels, num_classes


# ============================================================
# EVALUATE
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ============================================================
# FEDERATED LOOP
# ============================================================

def federated_run(dataset_name, gpus):

    print(f"\n========== DATASET: {dataset_name} ==========\n", flush=True)

    trainset, testset, labels, num_classes = load_dataset(dataset_name)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    for alpha in DIRICHLET_ALPHAS:

        print(f"\n==== α = {alpha} ====\n", flush=True)

        splits = dirichlet_split(labels, NUM_CLIENTS, alpha)

        device0 = "cuda:0"
        global_model = ResNet18_Pretrained(num_classes).to(device0)
        trainable_params = [p for p in global_model.parameters() if p.requires_grad]

        # =====================================================
        # CREATE SHARED MEMORY FOR ALL GLOBAL PARAMETERS
        # =====================================================
        shm_info_list = []
        for p in trainable_params:
            shm, flat_shape, real_shape = create_shared_tensor(p.detach().cpu())
            shm_info_list.append((shm.name, flat_shape, real_shape))

        c_global = [torch.zeros_like(p).cpu() for p in trainable_params]
        c_local = [[torch.zeros_like(p).cpu() for p in trainable_params]
                   for _ in range(NUM_CLIENTS)]

        # =====================================================
        # START WORKERS
        # =====================================================
        ctx = mp.get_context("spawn")
        job_q = ctx.Queue()
        res_q = ctx.Queue()

        workers = []
        for gpu in range(gpus):
            p = ctx.Process(
                target=worker_loop,
                args=(gpu, job_q, res_q, trainset, num_classes,
                      shm_info_list)
            )
            p.start()
            workers.append(p)

        # =====================================================
        # ROUNDS
        # =====================================================
        for rnd in range(1, NUM_ROUNDS + 1):

            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_INIT * 0.1
            print(f"--- ROUND {rnd}/{NUM_ROUNDS} ---", flush=True)

            # PUT JOBS
            for cid in range(NUM_CLIENTS):
                job_q.put((
                    "RUN",
                    cid,
                    splits[cid],
                    c_local[cid],
                    c_global,
                    lr
                ))

            # COLLECT
            new_params_accum = None
            for _ in range(NUM_CLIENTS):
                cid, new_params, new_c_local, delta_c = res_q.get()
                c_local[cid] = new_c_local

                if new_params_accum is None:
                    new_params_accum = [
                        torch.zeros_like(p) for p in new_params
                    ]

                for i in range(len(new_params)):
                    new_params_accum[i] += new_params[i]

                for i in range(len(c_global)):
                    c_global[i] += delta_c[i] / NUM_CLIENTS

            # AGGREGATE
            avg_params = [p / NUM_CLIENTS for p in new_params_accum]

            # UPDATE GLOBAL MODEL
            idx = 0
            with torch.no_grad():
                for p in trainable_params:
                    p.copy_(avg_params[idx].to(device0))
                    idx += 1

            # WRITE UPDATED PARAMS TO SHARED MEMORY
            idx = 0
            for p in trainable_params:
                update_shared_tensor(
                    shm_info_list[idx][0],
                    shm_info_list[idx][1],
                    p.detach().cpu()
                )
                idx += 1

            # EVAL
            acc = evaluate(global_model, testloader, device0)
            log_accuracy(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")

        # STOP WORKERS
        for _ in range(gpus):
            job_q.put(("STOP",))
        for p in workers:
            p.join()


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    for ds in ["CIFAR10", "CIFAR100", "SVHN"]:
        federated_run(ds, args.gpus)


if __name__ == "__main__":
    main()



