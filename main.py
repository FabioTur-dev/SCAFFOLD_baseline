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
import time

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models


# ==============================================================
# CONFIG (VERSIONE CHE TI FACEVA 81% AL ROUND 7)
# ==============================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.5]
NUM_ROUNDS = 100
LOCAL_EPOCHS = 2
BATCH = 128

LR_INIT = 0.01        # <- identico alla versione funzionante
LR_DECAY_ROUND = 40

BETA = 0.01           # scaffold-lite stable
SEED = 42

# ==============================================================
# LOGGING — solo accuracy
# ==============================================================

def loga(msg):
    print(msg, flush=True)

def logd(msg):
    pass


# ==============================================================
# SEED
# ==============================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ==============================================================
# RAW DATASET (VERSIONE ORIGINALE)
# ==============================================================

from torchvision.transforms.functional import to_pil_image

class RawDataset(Dataset):
    def __init__(self, data, labels, indices, augment):
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
        img = to_pil_image(self.data[idx])      # FIX fondamentale
        return self.T(img), self.labels[idx]


# ==============================================================
# DIRICHLET SPLIT (identico)
# ==============================================================

def dirichlet_split(labels, n_clients, alpha):
    labels = np.array(labels)
    per = [[] for _ in range(n_clients)]
    classes = np.unique(labels)

    for c in classes:
        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)
        p = np.random.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(p) * len(idx)).astype(int)
        chunks = np.split(idx, cuts[:-1])
        for i in range(n_clients):
            per[i].extend(chunks[i])

    for cl in per:
        random.shuffle(cl)

    return per


# ==============================================================
# MODEL — VERSIONE ORIGINALE
# ==============================================================

class ResNet18Pre(nn.Module):
    def __init__(self, nc):
        super().__init__()

        try:
            from torchvision.models import ResNet18_Weights
            self.m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except:
            self.m = models.resnet18(pretrained=True)

        in_f = self.m.fc.in_features
        self.m.fc = nn.Linear(in_f, nc)

        # VERSIONE ORIGINALE: addestriamo tutto (niente freeze extra)
        for p in self.m.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.m(x)


# ==============================================================
# PREPROCESS RAW DATASET UNA SOLA VOLTA
# ==============================================================

def preprocess_raw_dataset(ds_name):
    os.makedirs("cached", exist_ok=True)
    fp_data = f"cached/{ds_name}_data.pt"
    fp_lab = f"cached/{ds_name}_labels.pt"

    if os.path.exists(fp_data) and os.path.exists(fp_lab):
        return torch.load(fp_data), torch.load(fp_lab)

    if ds_name == "CIFAR10":
        d = datasets.CIFAR10("./data", train=True, download=True)
        data = torch.tensor(d.data).permute(0,3,1,2)
        labels = torch.tensor(d.targets)

    torch.save(data, fp_data)
    torch.save(labels, fp_lab)
    return data, labels


# ==============================================================
# WORKER — identico alla versione che funzionava
# ==============================================================

def client_update_worker(args):

    device = f"cuda:{args.gpu}"

    data = torch.load(f"cached/{args.dataset}_data.pt")
    labels = torch.load(f"cached/{args.dataset}_labels.pt")
    indices = torch.load(args.train_idx)

    ds = RawDataset(data, labels, indices, augment=True)

    loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                        num_workers=2, pin_memory=True)

    model = ResNet18Pre(args.num_classes).to(device)
    model.load_state_dict(torch.load(args.global_ckpt, map_location="cpu"))

    trainable = [p for p in model.parameters() if p.requires_grad]

    c_state = torch.load(args.state_c, map_location="cpu")
    c_global = c_state["c_global"]
    c_local = c_state["c_local"][args.cid]

    old_params = [p.detach().clone().cpu() for p in trainable]

    opt = optim.SGD(trainable, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    E = len(loader)

    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()

            # scaffold correction
            for i, p in enumerate(trainable):
                p.grad += (c_global[i].to(device) - c_local[i].to(device))

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


# ==============================================================
# EVALUATE
# ==============================================================

def evaluate(model, loader, device):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total


# ==============================================================
# FEDERATED LOOP
# ==============================================================

def federated_run(ds_name, gpus):

    raw_data, raw_labels = preprocess_raw_dataset(ds_name)

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225)),
    ])

    te = datasets.CIFAR10("./data", train=False, download=True,
                          transform=transform_test)

    testloader = DataLoader(te, batch_size=256, shuffle=False)
    labels_np = raw_labels.numpy()
    nc = 10

    for alpha in DIR_ALPHAS:

        loga(f"\n==== DATASET={ds_name} | α={alpha} ====\n")

        splits = dirichlet_split(labels_np, NUM_CLIENTS, alpha)

        device0 = "cuda:0"
        global_model = ResNet18Pre(nc).to(device0)
        trainable = [p for p in global_model.parameters() if p.requires_grad]

        os.makedirs("global_ckpt", exist_ok=True)
        os.makedirs("client_updates", exist_ok=True)

        global_ckpt_path = "global_ckpt/global_round_0.pth"
        torch.save(global_model.state_dict(), global_ckpt_path)

        c_global = [torch.zeros_like(p).cpu() for p in trainable]
        c_local = [[torch.zeros_like(p).cpu() for p in trainable]
                   for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS+1):

            lr = LR_INIT if rnd <= LR_DECAY_ROUND else LR_INIT*0.1

            # save control variates
            state_c_path = "global_ckpt/state_c.pth"
            torch.save({"c_local": c_local, "c_global": c_global}, state_c_path)

            # save indices
            idx_paths = []
            for cid in range(NUM_CLIENTS):
                pth = f"global_ckpt/train_idx_{cid}.pth"
                torch.save(splits[cid], pth)
                idx_paths.append(pth)

            # save global model
            global_ckpt_path = f"global_ckpt/global_round_{rnd-1}.pth"
            torch.save(global_model.state_dict(), global_ckpt_path)

            # launch workers
            procs = []
            out_paths = []

            for cid in range(NUM_CLIENTS):

                gpu = cid % gpus
                outp = f"client_updates/cid_{cid}_r{rnd}.pth"
                out_paths.append(outp)

                cmd = [
                    sys.executable, "main.py",
                    "--worker",
                    "--dataset", ds_name,
                    "--num_classes", str(nc),
                    "--cid", str(cid),
                    "--gpu", str(gpu),
                    "--global_ckpt", global_ckpt_path,
                    "--state_c", state_c_path,
                    "--train_idx", idx_paths[cid],
                    "--lr", str(lr),
                    "--output", outp
                ]

                procs.append(subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                ))

            for p in procs:
                p.wait()

            # ======================================================
            # HPC-SAFE LOAD OF CLIENT FILES (UNICA PATCH RICHIESTA)
            # ======================================================
            updates = []

            for cid in range(NUM_CLIENTS):
                outp = out_paths[cid]

                wait_time = 0
                while not os.path.exists(outp) and wait_time < 30:
                    time.sleep(1)
                    wait_time += 1

                if not os.path.exists(outp):
                    # retry worker
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
                        "--output", outp
                    ]
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    ).wait()

                    wait_time = 0
                    while not os.path.exists(outp) and wait_time < 30:
                        time.sleep(1)
                        wait_time += 1

                if not os.path.exists(outp):
                    raise RuntimeError(f"Worker {cid} FAILED after retry.")

                updates.append(torch.load(outp, map_location="cpu"))

            # ======================================================

            # aggregate
            new_acc = None
            for u in updates:
                if new_acc is None:
                    new_acc = [torch.zeros_like(p) for p in u["new_params"]]
                for i,p in enumerate(u["new_params"]):
                    new_acc[i] += p

            avg_params = [p / NUM_CLIENTS for p in new_acc]

            idxp = 0
            with torch.no_grad():
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[idxp].to(device0))
                        idxp += 1

            # update control variates
            for i in range(len(c_global)):
                c_global[i] = sum(u["delta_c"][i] for u in updates) / NUM_CLIENTS

            for u in updates:
                c_local[u["cid"]] = u["new_c_local"]

            # eval
            acc = evaluate(global_model, testloader, device0)
            loga(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")


# ==============================================================
# MAIN
# ==============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--cid", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--global_ckpt")
    parser.add_argument("--state_c")
    parser.add_argument("--train_idx")
    parser.add_argument("--output")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()

    if args.worker:
        return client_update_worker(args)

    set_seed(SEED)

    federated_run("CIFAR10", args.gpus)


if __name__ == "__main__":
    main()








