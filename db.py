import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
import hashlib

load_dotenv()

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.strip().encode("utf-8")).hexdigest()

uri = os.environ["MONGODB_URI"]

client = MongoClient(uri, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=8000)
db = client["distillena"]
col = db["generated_texts"]

# 防重複（建議只要跑一次就好，重跑也沒差）
col.create_index("hash", unique=True)

text = "This is a test sentence for DistillENA (Stage 1)."

doc = {
    "task": "argumentative_style",
    "label": "claim",
    "text": text,
    "hash": sha1_text(text),
    "source": {"role": "generator", "provider": "manual", "model": "none"},
    "created_at": datetime.utcnow(),
}

try:
    r = col.insert_one(doc)
    print("Inserted:", r.inserted_id)
except Exception as e:
    print("Skipped (duplicate or error):", e)