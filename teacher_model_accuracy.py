from pymongo import MongoClient
import os, certifi
from dotenv import load_dotenv

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
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

LABELS = [l.strip() for l in LABELS]  # 放在 LABELS 定義後


DB_NAME = "distillena"
COL_NAME = "teacher_softlabels_v2"
OUT_CSV = "teacher_metrics_all_in_one.csv"


def normalize_teacher(t) -> str:
    if isinstance(t, dict):
        return f"{t.get('provider','unknown')}/{t.get('model','unknown')}"
    return str(t)


def main():
    load_dotenv()
    uri = os.environ["MONGODB_URI"]

    client = MongoClient(uri, tls=True, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    col = db[COL_NAME]

    docs = list(col.find(
        {"teacher": {"$ne": None}, "concept": {"$ne": None}, "pred": {"$ne": None}},
        {"teacher": 1, "concept": 1, "pred": 1, "_id": 0}
    ))
    df = pd.DataFrame(docs)
    if df.empty:
        raise ValueError(f"{COL_NAME} collection is empty or no valid documents found.")

    df["teacher"] = df["teacher"].apply(normalize_teacher)
    df["concept"] = df["concept"].astype(str).str.strip()
    df["pred"] = df["pred"].astype(str).str.strip()

    rows = []


    for teacher, g in df.groupby("teacher"):
        y_true = g["concept"].str.strip().tolist()
        y_pred = g["pred"].str.strip().tolist()

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        rows.append({
            "teacher": teacher,
            "metric_scope": "overall_macro",
            "label": "__ALL__",
            "n": len(g),
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": len(g),
        })

    # ===== (B) per-label metrics per teacher (many rows per teacher) =====
    for teacher, g in df.groupby("teacher"):
        y_true = g["concept"].str.strip().tolist()
        y_pred = g["pred"].str.strip().tolist()

        rep = classification_report(
            y_true, y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0
        )

        for lab in LABELS:
            r = rep.get(lab, {})
            rows.append({
                "teacher": teacher,
                "metric_scope": "per_label",
                "label": lab,
                "n": len(g),
                "accuracy": None,  # per-label 不定義 accuracy（避免誤解）
                "precision": r.get("precision", 0.0),
                "recall": r.get("recall", 0.0),
                "f1": r.get("f1-score", 0.0),
                "support": r.get("support", 0),
            })

    out = pd.DataFrame(rows)

    # 排序：先 overall 再 per_label；teacher 按 overall accuracy 由高到低
    teacher_order = (
        out[(out["metric_scope"] == "overall_macro")]
        .sort_values("accuracy", ascending=False)["teacher"]
        .tolist()
    )
    out["teacher"] = pd.Categorical(out["teacher"], categories=teacher_order, ordered=True)
    out["metric_scope"] = pd.Categorical(out["metric_scope"], categories=["overall_macro", "per_label"], ordered=True)
    out = out.sort_values(["teacher", "metric_scope", "label"]).reset_index(drop=True)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[ok] saved: {OUT_CSV}")

    # ===== (C) confusion matrices: 只印在 terminal（不寫檔） =====
    for teacher, g in df.groupby("teacher"):
        y_true = g["concept"].tolist()
        y_pred = g["pred"].tolist()

        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in LABELS], columns=[f"pred:{l}" for l in LABELS])

        print(f"\n=== Confusion matrix :: {teacher} ===")
        print(cm_df.to_string())


if __name__ == "__main__":
    main()