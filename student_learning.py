import os, certifi, random
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import json
from dotenv import load_dotenv
from pymongo import MongoClient
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)

LABELS = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement",
]

LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

DB_NAME = "distillena"
COL_NAME = "teacher_softlabels_v2"

#Human label dataset for validation
COL_VAL_NAME = "PERSUADE"
TEXT_FIELD = "discourse_text"
LABEL_FIELD = "discourse_type"

class HumanDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tok = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]

        text = (r.get(TEXT_FIELD) or "").strip()
        gold = (r.get(LABEL_FIELD) or "").strip()

        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        y = LABELS.index(gold) if gold in LABELS else -1

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y": torch.tensor(y, dtype=torch.long),
        }

def load_human_val_records(col_val, per_label=20, seed=None):
    # avoid referencing SEED before it is defined
    if seed is None:
        seed = 2026
    rng = random.Random(seed)
    all_records = []

    for lab in LABELS:
        cur = col_val.find(
            {LABEL_FIELD: lab, TEXT_FIELD: {"$type": "string", "$ne": ""}},
            {TEXT_FIELD: 1, LABEL_FIELD: 1, "_id": 0}
        )

        docs = list(cur)
        rng.shuffle(docs)
        docs = docs[:per_label]
        all_records.extend(docs)

    rng.shuffle(all_records)
    return all_records

#########


TEACHERS = [
    {"provider": "anthropic", "model": "claude-opus-4-6"},
#     {"provider": "openai", "model": "gpt-5.2"},
#     {"provider": "google", "model": "gemini-3-pro-preview"},
 ]

STUDENTS = [
    "distilbert/distilbert-base-uncased",
    "microsoft/MiniLM-L12-H384-uncased",
    "distilbert/distilbert-base-multilingual-cased",
    "FacebookAI/xlm-roberta-base",
]

#hyperparameters
MAX_LEN = 64
BATCH_SIZE = 16
LR = 5e-6
EPOCHS = 5
# ALPHA = 0.7 # change here for testing different alpha values (e.g., 0.5, 0.3, 0.0 for no hard loss)
ALPHAS = [0.0, 0.2, 0.5, 0.7, 1.0]
SEED = 2026

RESULTS_PATH = "student_results.jsonl"



def teacher_key(t: dict) -> str:
    return f"{t.get('provider','unknown')}/{t.get('model','unknown')}"

def normalize_labels_dict(probs: dict) -> np.ndarray:
    vec = np.array([float(probs.get(l, 0.0)) for l in LABELS], dtype=np.float32)
    s = float(vec.sum())
    if s <= 0:
        vec[:] = 1.0 / len(LABELS)
    else:
        vec /= s
    return vec

class DistillDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tok = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        text = (r.get("text") or "").strip()
        concept = (r.get("concept") or "").strip()   # generated_label(hard label)
        probs = r.get("probs") or {}                # teacher soft labels(probs)

        enc = self.tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        y = LABELS.index(concept) if concept in LABELS else -1
        tvec = normalize_labels_dict(probs)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y": torch.tensor(y, dtype=torch.long),
            "tvec": torch.tensor(tvec, dtype=torch.float32),
        }

def load_records_for_teacher(col, teacher_cfg, limit):
    q = {
        "teacher.provider": teacher_cfg["provider"],
        "teacher.model": teacher_cfg["model"],
        "text": {"$type": "string", "$ne": ""},
        "concept": {"$type": "string", "$ne": ""},
        "probs": {"$type": "object"},
    }
    proj = {"text": 1, "concept": 1, "probs": 1, "_id": 0}
    cur = col.find(q, proj)
    if limit:
        cur = cur.limit(int(limit))
    return list(cur)



@torch.no_grad()
def evaluate_hard(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        y = batch["y"].cpu().numpy().tolist()

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

        for yt, yp in zip(y, pred):
            if yt == -1:
                continue
            y_true.append(yt)
            y_pred.append(yp)

    if not y_true:
        return {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1)}





def train_one(student_name, train_records, device, alpha):
    tok = AutoTokenizer.from_pretrained(student_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        student_name,
        num_labels=len(LABELS),
    ).to(device)

    # ====== NEW: pre-tokenize once (big speedup) ======
    texts = [(r.get("text") or "").strip() for r in train_records]
    concepts = [(r.get("concept") or "").strip() for r in train_records]
    probs_list = [r.get("probs") or {} for r in train_records]

    enc = tok(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    y = torch.tensor(
        [LABELS.index(c) if c in LABELS else -1 for c in concepts],
        dtype=torch.long
    )

    tvec = torch.tensor(
        [normalize_labels_dict(p) for p in probs_list],
        dtype=torch.float32
    )

    class PreTokDistillDataset(Dataset):
        def __init__(self, enc, y, tvec):
            self.input_ids = enc["input_ids"]
            self.attn = enc["attention_mask"]
            self.y = y
            self.tvec = tvec

        def __len__(self):
            return self.input_ids.size(0)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attn[idx],
                "y": self.y[idx],
                "tvec": self.tvec[idx],
            }

    train_ds = PreTokDistillDataset(enc, y, tvec)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
    kld = torch.nn.KLDivLoss(reduction="batchmean")

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for batch in train_loader:
            optim.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            yb = batch["y"].to(device)
            tvecb = batch["tvec"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attn).logits

            logp = torch.log_softmax(logits, dim=-1)
            distill_loss = kld(logp, tvecb)
            hard_loss = ce(logits, yb)

            loss = (1 - alpha) * distill_loss + alpha * hard_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            losses.append(loss.item())

        print(f"[epoch {ep}] loss={float(np.mean(losses)):.4f}")

    return model, tok





def append_result(row: dict):
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")



def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    load_dotenv()
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI not found")

    client = MongoClient(uri, tls=True, tlsCAFile=certifi.where())
    db = client[DB_NAME]

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # (A) human validation (PERSUADE) — 固定同一份，讓不同模型可比
    col_val = db[COL_VAL_NAME]
    val_records = load_human_val_records(col_val, per_label=30, seed=SEED)
    print("PERSUADE val size:", len(val_records))

    val_ds_cache = {}  

    # (B) training data source
    src = db[COL_NAME]  # teacher_softlabels_v2

    # optional: clear previous results
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)

    for teacher_cfg in TEACHERS:
        for student_name in STUDENTS:
            for alpha in ALPHAS:

                train_records = load_records_for_teacher(src, teacher_cfg, limit=None)

                print(f"== TRAIN student={student_name} <- teacher={teacher_key(teacher_cfg)} | alpha={alpha} ==")
                model, tok = train_one(student_name, train_records, device, alpha)

                val_ds = HumanDataset(val_records, tok)
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
                m = evaluate_hard(model, val_loader, device)

                row = {
                    "teacher": teacher_key(teacher_cfg),
                    "student": student_name,
                    "alpha": alpha,
                    "n_train": len(train_records),
                    "n_human_val": len(val_records),
                    **m,
                }
                print("[human-val]", row)
                append_result(row)

            # cleanup (important when looping many runs)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    print("\n[ok] all done ->", RESULTS_PATH)


if __name__ == "__main__":
    main()