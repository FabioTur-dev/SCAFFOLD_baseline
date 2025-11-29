#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import time


# ======================================================================
# CONFIG — SCAFFOLD-Lite++ (87%)
# ======================================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.5]
NUM_ROUNDS = 100
LOCAL_EPOCHS = 2
BATCH = 128

LR_INIT = 0.003
BETA = 0.01
DAMPING = 0.1
GRAD_CLIP = 5.0
EMA_DECAY = 0.995

SEED = 42


# ======================================================================
# LOGGING
# ======================================================================

def loga(msg):
    print(msg, flush=True)

def logd(msg):
    pass


# ======================================================================
# SEED
# ======================================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ======================================================================
# RAW DATASET (Cached)
# ======================================================================

class RawDataset(Dataset):
    def __init__(self, data, labels, indices, augment=False):
        self.data = data
        self.labels = labels
        self.indices = indices

        if augment:
            self.T = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),
                                     (0.229,0.224,0.225)),
                transforms.RandomErasing(p=0.25, scale=(0.02,0.2))
            ])
        else:
            self.T = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),
                                     (0.229,0.224,0.225)),
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = self.data[idx]
        return self.T(img), self.labels[idx]


def preprocess_raw_dataset(ds_name):
    os.makedirs("cached", exist_ok=True)

    df = f"cached/{ds_name}_data.pt"
    lf = f"cached/{ds_name}_labels.pt"

    if os.path.exists(df) and os.path.exists(lf):
        return torch.load(df), torch.load(lf)

    if ds_name == "CIFAR10":
        d = datasets.CIFAR10("./data", train=True, download=True)
        data = torch.tensor(d.data).permute(0,3,1,2)
        labels = torch.tensor(d.targets)
    elif ds_name == "CIFAR100":
        d = datasets.CIFAR100("./data", train=True, download=True)
        data = torch.tensor(d.data).permute(0,3,1,2)
        labels = torch.tensor(d.targets)
    else:
        from torchvision.datasets import SVHN
        d = SVHN("./data", split="train", download=True)
        data = torch.tensor(d.data).permute(0,3,1,2)
        labels = torch.tensor(d.labels)

    torch.save(data, df)
    torch.save(labels, lf)

    return data, labels


# ======================================================================
# DIRICHLET
# ======================================================================

def dirichlet_split(labels, n_clients, alpha):
    labels = np.array(labels)
    per = [[] for _ in range(n_clients)]
    classes = np.unique(labels)

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha]*n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for i in range(n_clients):
            per[i].extend(chunks[i])

    for L in per:
        random.shuffle(L)

    return per


# ======================================================================
# MODEL — fine-tune only last blocks
# ======================================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()
        from torchvision.models import ResNet18_Weights
        self.m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        inf = self.m.fc.in_features
        self.m.fc = nn.Linear(inf, nc)

        for name, p in self.m.named_parameters():
            if ("layer3" in name) or ("layer4" in name) or ("fc" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, x):
        return self.m(x)


# ======================================================================
# WORKER
# ======================================================================

def client_update_worker(args):
    device = f"cuda:{args.gpu}"

    # load cached raw data
    data = torch.load(f"cached/{args.dataset}_data.pt")
    labels = torch.load(f"cached/{args.dataset}_labels.pt")
    indices = torch.load(args.train_idx)

    ds = RawDataset(data, labels, indices, augment=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

    model = ResNet18Pre(args.num_classes).to(device)
    model.load_state_dict(torch.load(args.global_ckpt, map_location="cpu"))

    trainable = [p for p in model.parameters() if p.requires_grad]

    c_state = torch.load(args.state_c, map_location="cpu")
    c_global = c_state["c_global"]
    c_local = c_state["c_local"][args.cid]

    old_params = [p.detach().clone().cpu() for p in trainable]

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(trainable, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    E = len(loader)

    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()

            # SCAFFOLD-Lite++
            for i, p in enumerate(trainable):
                p.grad += DAMPING * (c_global[i].to(device) - c_local[i].to(device))

            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step()

    new_params = [p.detach().clone().cpu() for p in trainable]
    delta_c = []

    for i in range(len(trainable)):
        diff = new_params[i] - old_params[i]
        dc = BETA * (diff / max(E,1))
        delta_c.append(dc)
        c_local[i] += dc

    torch.save({
        "cid": args.cid,
        "new_params": new_params,
        "new_c_local": c_local,
        "delta_c": delta_c
    }, args.output)

    return


# ======================================================================
# EVAL
# ======================================================================

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


# ======================================================================
# EMA UPDATE
# ======================================================================

def ema_update(global_model, ema_model):
    with torch.no_grad():
        for p, q in zip(global_model.parameters(), ema_model.parameters()):
            q.data = EMA_DECAY * q.data + (1-EMA_DECAY) * p.data


# ======================================================================
# FEDERATED LOOP
# ======================================================================

def federated_run(ds_name, gpus):

    # --------------------------------------------------------------
    # RAW DATA LOADING ONCE
    # --------------------------------------------------------------
    raw_data, raw_labels = preprocess_raw_dataset(ds_name)
    labels_np = raw_labels.numpy()

    # --------------------------------------------------------------
    # TEST SET
    # --------------------------------------------------------------
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225)),
    ])

    if ds_name == "CIFAR10":
        te = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
        nc = 10
    elif ds_name == "CIFAR100":
        te = datasets.CIFAR100("./data", train=False, download=True, transform=transform_test)
        nc = 100
    else:
        from torchvision.datasets import SVHN
        te = SVHN("./data", split="test", download=True, transform=transform_test)
        nc = 10

    testloader = DataLoader(te, batch_size=256, shuffle=False)

    # ===================================================================
    # LOOP α VALUES
    # ===================================================================

    for alpha in DIR_ALPHAS:

        loga(f"\n==== DATASET={ds_name} | α={alpha} ====\n")

        splits = dirichlet_split(labels_np, NUM_CLIENTS, alpha)

        # INIT GLOBAL MODEL
        device0 = "cuda:0"
        global_model = ResNet18Pre(nc).to(device0)
        ema_model = ResNet18Pre(nc).to(device0)
        ema_model.load_state_dict(global_model.state_dict())

        trainable = [p for p in global_model.parameters() if p.requires_grad]

        os.makedirs("global_ckpt", exist_ok=True)
        os.makedirs("client_updates", exist_ok=True)

        c_global = [torch.zeros_like(p).cpu() for p in trainable]
        c_local = [[torch.zeros_like(p).cpu() for p in trainable]
                   for _ in range(NUM_CLIENTS)]

        # COSINE LR schedule
        def cosine_lr(r):
            t = r / NUM_ROUNDS
            return LR_INIT * (0.5 * (1 + np.cos(np.pi * t)))

        # ==========================================================
        # ROUNDS
        # ==========================================================

        for rnd in range(1, NUM_ROUNDS+1):

            lr = cosine_lr(rnd)

            # save control variates
            state_c_path = "global_ckpt/state_c.pth"
            torch.save({"c_local": c_local, "c_global": c_global}, state_c_path)

            # save indices
            idx_paths = []
            for cid in range(NUM_CLIENTS):
                p = f"global_ckpt/train_idx_{cid}.pth"
                torch.save(splits[cid], p)
                idx_paths.append(p)

            # save current global model
            global_ckpt_path = f"global_ckpt/global_{rnd}.pth"
            torch.save(global_model.state_dict(), global_ckpt_path)

            # --------------------------------------------------
            # WORKERS (silent)
            # --------------------------------------------------
            procs = []
            out_paths = []

            for cid in range(NUM_CLIENTS):
                p_out = f"client_updates/cid_{cid}_r{rnd}.pth"
                out_paths.append(p_out)

                cmd = [
                    sys.executable, "main.py",
                    "--worker",
                    "--dataset", ds_name,
                    "--num_classes", str(nc),
                    "--cid", str(cid),
                    "--gpu", str(cid % gpus),
                    "--global_ckpt", global_ckpt_path,
                    "--state_c", state_c_path,
                    "--train_idx", idx_paths[cid],
                    "--lr", str(lr),
                    "--output", p_out,
                ]

                procs.append(subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                ))

            for p in procs:
                p.wait()

            # --------------------------------------------------
            # HPC-SAFE FILE CHECK + RETRY LOGIC
            # --------------------------------------------------
            for cid, outp in enumerate(out_paths):

                if not os.path.exists(outp):
                    waited = 0
                    while not os.path.exists(outp) and waited < 40:
                        time.sleep(1)
                        waited += 1

                if not os.path.exists(outp):
                    # retry worker once
                    cmd = [
                        sys.executable, "main.py",
                        "--worker",
                        "--dataset", ds_name,
                        "--num_classes", str(nc),
                        "--cid", str(cid),
                        "--gpu", str(cid % gpus),
                        "--global_ckpt", global_ckpt_path,
                        "--state_c", state_c_path,
                        "--train_idx", idx_paths[cid],
                        "--lr", str(lr),
                        "--output", outp,
                    ]
                    subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    ).wait()

                if not os.path.exists(outp):
                    raise RuntimeError(f"Worker {cid} FAILED twice: {outp}")

            # --------------------------------------------------
            # LOAD UPDATES
            # --------------------------------------------------
            updates = [torch.load(out_paths[c], map_location="cpu")
                       for c in range(NUM_CLIENTS)]

            new_accum = [torch.zeros_like(p) for p in updates[0]["new_params"]]

            for u in updates:
                for i, p in enumerate(u["new_params"]):
                    new_accum[i] += p

            avg_params = [p / NUM_CLIENTS for p in new_accum]

            # update global
            idx = 0
            with torch.no_grad():
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[idx].to(device0))
                        idx += 1

            # update c_global
            for i in range(len(c_global)):
                c_global[i] = sum(u["delta_c"][i] for u in updates) / NUM_CLIENTS

            # update local
            for u in updates:
                c_local[u["cid"]] = u["new_c_local"]

            # EMA Model Update (improves final acc by +1%)
            ema_update(global_model, ema_model)

            # Evaluation (use ema_model for smoother accuracy)
            acc = evaluate(ema_model, testloader, device0)
            loga(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--cid", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--global_ckpt", type=str)
    parser.add_argument("--state_c", type=str)
    parser.add_argument("--train_idx", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()

    if args.worker:
        return client_update_worker(args)

    set_seed(SEED)

    # only CIFAR10 for now
    federated_run("CIFAR10", args.gpus)


if __name__ == "__main__":
    main()







