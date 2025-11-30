#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import to_pil_image


# ==============================================================
# GLOBAL CONFIG
# ==============================================================

NUM_CLIENTS = 10
DIR_ALPHAS = [0.5, 0.1, 0.05]
NUM_ROUNDS = 50
LOCAL_EPOCHS = 2
BATCH = 128

SEED = 42
GRAD_CLIP = 5.0


# ==============================================================
# LOG
# ==============================================================

def loga(msg):
    print(msg, flush=True)


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
# RAW DATASET WRAPPER
# ==============================================================

class RawDataset(Dataset):
    def __init__(self, data, labels, indices, augment, resize_size):
        self.data = data
        self.labels = labels
        self.indices = indices

        if augment:
            self.T = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),

                transforms.Resize(resize_size),

                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),

                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            ])
        else:
            self.T = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = to_pil_image(self.data[idx])
        img = self.T(img)
        return img, self.labels[idx]


# ==============================================================
# DIRICHLET SPLIT
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
# MODEL (layer3+layer4+fc sbloccati)
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

        for name, p in self.m.named_parameters():
            if "layer3" in name or "layer4" in name or "fc" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, x):
        return self.m(x)


# ==============================================================
# PREPROCESS RAW
# ==============================================================

def preprocess_raw_dataset(ds_name):
    os.makedirs("cached", exist_ok=True)
    data_file = f"cached/{ds_name}_train_raw.pt"
    label_file = f"cached/{ds_name}_train_labels.pt"

    if os.path.exists(data_file) and os.path.exists(label_file):
        return torch.load(data_file), torch.load(label_file)

    if ds_name == "CIFAR10":
        d = datasets.CIFAR10("./data", train=True, download=True)
        data = torch.tensor(d.data).permute(0, 3, 1, 2)
        labels = torch.tensor(d.targets)
    else:
        d = datasets.CIFAR100("./data", train=True, download=True)
        data = torch.tensor(d.data).permute(0, 3, 1, 2)
        labels = torch.tensor(d.targets)

    torch.save(data, data_file)
    torch.save(labels, label_file)
    return data, labels


# ==============================================================
# CLIENT WORKER
# ==============================================================

def client_update_worker(args, CONFIG):

    resize_size = CONFIG["resize"]
    device = f"cuda:{args.gpu}"

    data = torch.load(f"cached/{args.dataset}_train_raw.pt")
    labels = torch.load(f"cached/{args.dataset}_train_labels.pt")
    indices = torch.load(args.train_idx)

    ds = RawDataset(data, labels, indices, augment=True, resize_size=resize_size)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

    model = ResNet18Pre(args.num_classes).to(device)
    model.load_state_dict(torch.load(args.global_ckpt, map_location="cpu"))

    trainable = [p for p in model.parameters() if p.requires_grad]

    c_state = torch.load(args.state_c, map_location="cpu")
    c_global = c_state["c_global"]
    c_local = c_state["c_local"][args.cid]

    old_params = [p.detach().clone().cpu() for p in trainable]

    opt = optim.SGD(trainable, lr=args.lr,
                    momentum=CONFIG["momentum"],
                    weight_decay=CONFIG["wd"])
    loss_fn = nn.CrossEntropyLoss()

    E = len(loader)

    for _ in range(LOCAL_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()

            for i, p in enumerate(trainable):
                p.grad += CONFIG["damping"] * (c_global[i].to(device) - c_local[i].to(device))

            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            opt.step()

    new_params = [p.detach().clone().cpu() for p in trainable]

    delta_c = []
    for i in range(len(trainable)):
        diff = new_params[i] - old_params[i]
        dc = CONFIG["beta"] * (diff / max(E, 1))
        delta_c.append(dc)
        c_local[i] += dc

    torch.save({
        "cid": args.cid,
        "new_params": new_params,
        "new_c_local": c_local,
        "delta_c": delta_c,
    }, args.output)


# ==============================================================
# EVALUATION
# ==============================================================

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


# ==============================================================
# CONFIG MODES (B and C)
# ==============================================================

def get_config_for_mode(mode):

    if mode == "B":
        return {
            "resize": 160,
            "lr_init": 0.006,
            "lr_decay": 0.002,
            "decay_round": 25,
            "momentum": 0.9,
            "wd": 5e-4,
            "damping": 0.05,
            "beta": 0.01,
            "keepalive": 5.0,
        }

    if mode == "C":
        return {
            "resize": 224,
            "lr_init": 0.0045,
            "lr_decay": 0.0015,
            "decay_round": 25,
            "momentum": 0.9,
            "wd": 5e-4,
            "damping": 0.10,
            "beta": 0.01,
            "keepalive": 5.0,
        }

    raise ValueError("Unknown mode:", mode)


# ==============================================================
# FEDERATED LOOP
# ==============================================================

def federated_run(ds_name, gpus, CONFIG):

    raw_data, raw_labels = preprocess_raw_dataset(ds_name)

    transform_test = transforms.Compose([
        transforms.Resize(CONFIG["resize"]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    if ds_name == "CIFAR10":
        te = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
        nc = 10
    else:
        te = datasets.CIFAR100("./data", train=False, download=True, transform=transform_test)
        nc = 100

    testloader = DataLoader(te, batch_size=256, shuffle=False)

    labels_np = raw_labels.numpy()

    for alpha in DIR_ALPHAS:

        loga(f"\n==== DATASET={ds_name} | α={alpha} ====\n")

        splits = dirichlet_split(labels_np, NUM_CLIENTS, alpha)

        device0 = "cuda:0"
        global_model = ResNet18Pre(nc).to(device0)
        trainable = [p for p in global_model.parameters() if p.requires_grad]

        os.makedirs("global_ckpt", exist_ok=True)
        os.makedirs("client_updates", exist_ok=True)

        torch.save(global_model.state_dict(), "global_ckpt/global_round_0.pth")

        c_global = [torch.zeros_like(p).cpu() for p in trainable]
        c_local = [[torch.zeros_like(p).cpu() for p in trainable] for _ in range(NUM_CLIENTS)]

        for rnd in range(1, NUM_ROUNDS + 1):

            lr = CONFIG["lr_init"] if rnd <= CONFIG["decay_round"] else CONFIG["lr_decay"]

            state_c_path = "global_ckpt/state_c.pth"
            torch.save({"c_local": c_local, "c_global": c_global}, state_c_path)

            for _ in range(300):
                if os.path.exists(state_c_path) and os.path.getsize(state_c_path) > 1024:
                    break
                time.sleep(0.01)

            idx_paths = []
            for cid in range(NUM_CLIENTS):
                p = f"global_ckpt/train_idx_{cid}.pth"
                torch.save(splits[cid], p)
                idx_paths.append(p)

            global_path = f"global_ckpt/global_round_{rnd-1}.pth"
            torch.save(global_model.state_dict(), global_path)

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
                    "--global_ckpt", global_path,
                    "--state_c", state_c_path,
                    "--train_idx", idx_paths[cid],
                    "--lr", str(lr),
                    "--output", outp,
                    "--mode", CONFIG["mode"]
                ]

                procs.append(subprocess.Popen(cmd))

            # KEEP ALIVE
            while True:
                if all(p.poll() is not None for p in procs):
                    break
                loga(f"[SERVER] Waiting workers at round {rnd}...")
                time.sleep(CONFIG["keepalive"])

            # AGGREGATION
            updates = [torch.load(out_paths[c], map_location="cpu")
                       for c in range(NUM_CLIENTS)]

            new_accum = None
            for u in updates:
                if new_accum is None:
                    new_accum = [torch.zeros_like(p) for p in u["new_params"]]
                for i, p in enumerate(u["new_params"]):
                    new_accum[i] += p

            avg_params = [p / NUM_CLIENTS for p in new_accum]

            with torch.no_grad():
                pi = 0
                for p in global_model.parameters():
                    if p.requires_grad:
                        p.copy_(avg_params[pi].to(device0))
                        pi += 1

            for i in range(len(c_global)):
                c_global[i] = sum(u["delta_c"][i] for u in updates) / NUM_CLIENTS

            for u in updates:
                c_local[u["cid"]] = u["new_c_local"]

            acc = evaluate(global_model, testloader, device0)
            loga(f"[ROUND {rnd}] ACC = {acc*100:.2f}%")


# ==============================================================
# MAIN
# ==============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--cid", type=int)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--global_ckpt", type=str)
    parser.add_argument("--state_c", type=str)
    parser.add_argument("--train_idx", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--mode", type=str, default="B")
    args = parser.parse_args()

    CONFIG = get_config_for_mode(args.mode)
    CONFIG["mode"] = args.mode

    if args.worker:
        return client_update_worker(args, CONFIG)

    set_seed(SEED)

    federated_run("CIFAR10", args.gpus, CONFIG)


if __name__ == "__main__":
    main()











