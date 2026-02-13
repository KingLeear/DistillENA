import os
import json
import time
import hashlib
import certifi
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import TEXT, MongoClient, UpdateOne
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from pymongo.errors import OperationFailure

load_dotenv()

MONGO_DB = "distillena"
SRC_COL = "generated_text"
OUT_COL = "teacher_softlabels_v2"


TEACHER_PROVIDER = "openai"
# multiple teacher models
TEACHER_MODELS = [
    {"provider": "openai", "model": "gpt-5.2"},
    {"provider": "anthropic", "model": "claude-opus-4-6"},
    {"provider": "google", "model": "gemini-3-pro-preview"},
]


BATCH_LIMIT = None
SLEEP_SECONDS = 0.15

LABEL_LIST = [
    "Lead",
    "Position",
    "Claim",
    "Counterclaim",
    "Rebuttal",
    "Evidence",
    "Concluding Statement ",
]

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.strip().encode("utf-8")).hexdigest()

def get_mongo():
    uri = os.environ["MONGODB_URI"]
    client = MongoClient(uri, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=8000)
    return client[MONGO_DB]

def build_prompt(text: str) -> str:
    return f"""

You are a classification model.

You MUST output valid JSON.
DO NOT include explanations, comments, markdown, or extra text.
DO NOT wrap the output in ```.

You are classifying a student-written sentence into the following labels:

Lead: "The introduction begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis."
Position: "An opinion on the main question",
Claim: "A claim that supports the position."
Counterclaim: "A claim that refutes another claim or gives an opposing reason to the position.",
Rebuttal: "A claim that refutes a counterclaim."
Evidence: "Ideas or examples that support claims, counterclaims, or rebuttals."
Concluding Statement: "A concluding statement that restates the claims."

Decision logic (follow strictly):
1) First decide the sentence FUNCTION, not keywords.
2) If it is an opening hook without stating a stance → Lead.
3) If it clearly states a stance/opinion/should-claim on the main issue → Position.
4) If it gives a reason that supports the stance (often starts with "because", "one reason", "this is why") → Claim.
5) If it presents an opposing view → Counterclaim.
6) If it refutes that opposing view → Rebuttal.
7) If it provides a concrete example/data/source/event/case → Evidence.
8) If it summarizes/closes/reaffirms at the end → Concluding Statement.
9) If multiple apply, distribute probability (but keep the most plausible label highest).

Output rules:
- Output ONE JSON object
- Keys MUST be exactly these labels (spelling and capitalization must match)
- Every key MUST appear exactly once
- Values MUST be floats between 0 and 1
- All values MUST sum to exactly 1 (small rounding error is acceptable)
- Do NOT include any other keys

Correct output example (FORMAT ONLY — values are illustrative):

{
  {"Lead": 0.10,
  "Position": 0.15,
  "Claim": 0.35,
  "Counterclaim": 0.10,
  "Rebuttal": 0.10,
  "Evidence": 0.15,
  "Concluding Statement": 0.05}
}
Please DO close the brackets!
If the sentence weakly matches multiple labels, distribute probabilities accordingly.
Never output text outside the JSON object.

Sentence to classify:{text}

""".strip()

def teacher_softlabel_openai(client: OpenAI, text: str, model: str | None = None) -> dict:
    prompt = build_prompt(text)
    use_model = model or TEACHER_MODELS[0]["model"]
    resp = client.chat.completions.create(
        model=use_model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = (resp.choices[0].message.content or "").strip()
    probs = json.loads(content)

    for k in LABEL_LIST:
        if k not in probs:
            raise ValueError(f"Missing label: {k}")

    probs = {k: float(probs[k]) for k in LABEL_LIST}
    s = sum(probs.values())
    if not (0.98 <= s <= 1.02):
        raise ValueError(f"Prob sum not ~1: {s}")

    return probs

def main():
    print(">>> teacher_label.py started (writing to teacher_softlabels)")

    db = get_mongo()
    src = db[SRC_COL]
    out = db[OUT_COL]

    def safe_create_index(coll, *args, **kwargs):
        try:
            coll.create_index(*args, **kwargs)
        except OperationFailure as e:

            if getattr(e, "code", None) == 86:
                print(f"Index conflict for {coll.name} {args}; continuing.")
            else:
                raise

    safe_create_index(src, "hash")
    safe_create_index(out, "source_id")
    safe_create_index(out, [("teacher.provider", 1), ("teacher.model", 1)])
    safe_create_index(
        out,
        [("source_id", 1), ("teacher.provider", 1), ("teacher.model", 1)],
        unique=True,
    )

    # instantiate clients for all teacher providers
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    gemini_client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
)

    def teacher_softlabel_dispatch(model_cfg: dict, text: str) -> dict:
        provider = model_cfg["provider"]
        model = model_cfg["model"]

        prompt = build_prompt(text)

        if provider == "openai":
            resp = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = (resp.choices[0].message.content or "").strip()

        elif provider == "anthropic":
            msg = anthropic_client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = "".join(block.text for block in msg.content if block.type == "text").strip()

        elif provider == "google":
            r = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0.0, "maxOutputTokens": 2048,  "responseMimeType": "application/json"},
            )
            content = (r.text or "").strip()


        else:
            raise ValueError(f"Unknown teacher provider: {provider}")

        # extract JSON from content (models may wrap response in markdown or text)
        # try direct parse first
        try:
            probs = json.loads(content)

            for k in LABEL_LIST:
                if k not in probs:
                    probs[k] = 0.0

            # Renormalize to sum to 1
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
            else:
                # All zeros; give equal weight
                probs = {k: 1.0 / len(LABEL_LIST) for k in LABEL_LIST}
            
        except json.JSONDecodeError:
            # try extracting JSON from markdown code block
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                probs = json.loads(json_str)
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
                probs = json.loads(json_str)
            else:
    # safer JSON extraction (brace matching instead of greedy regex)
                start = content.find("{")
                if start == -1:
                    raise ValueError(f"Could not parse JSON from response: {content[:200]}")

                depth = 0
                for i in range(start, len(content)):
                    if content[i] == "{":
                        depth += 1
                    elif content[i] == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = content[start:i+1]
                            break
                else:
                    # if never properly closed, auto-close
                    json_str = content[start:] + "}" * depth

                json_str = re.sub(r",\s*}", "}", json_str)  # remove trailing comma

                try:
                    probs = json.loads(json_str)
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse JSON from response: {content[:200]}")

        for k in LABEL_LIST:
            if k not in probs:
                raise ValueError(f"Missing label: {k}")
        probs = {k: float(probs[k]) for k in LABEL_LIST}
        s = sum(probs.values())
        if not (0.98 <= s <= 1.02):
            raise ValueError(f"Prob sum not ~1: {s}")
        return probs

    cursor = src.find(
        {"text": {"$type": "string", "$ne": ""}},
        {"_id": 1, "text": 1, "label": 1, "concept": 1, "hash": 1, "provider": 1, "model": 1, "temperature": 1}
    )

    if BATCH_LIMIT:
        cursor = cursor.limit(BATCH_LIMIT)

    n_ok, n_skip, n_fail = 0, 0, 0

    for d in cursor:
        source_id = d["_id"]
        text = (d.get("text") or "").strip()
        if not text:
            continue
        h = d.get("hash") or sha1_text(text)

        # call each teacher model/provider
        for model_cfg in TEACHER_MODELS:
            provider = model_cfg["provider"]
            model = model_cfg["model"]

            exists = out.find_one(
                {"source_id": source_id, "teacher.provider": provider, "teacher.model": model},
                {"_id": 1}
            )
            if exists:
                n_skip += 1
                continue

            try:
                probs = teacher_softlabel_dispatch(model_cfg, text)
                pred = max(probs.items(), key=lambda kv: kv[1])[0]
                conf = probs[pred]
            except Exception as e:
                n_fail += 1
                print(f"[fail] {source_id} :: {provider}/{model} :: {type(e).__name__} {e}")
                continue

            doc = {
                "source_id": source_id,
                "hash": h,
                "text": text,
                "concept": d.get("concept"),
                "generator_model": d.get("model"),
                "teacher": {
                    "provider": model_cfg["provider"],
                    "model": model_cfg["model"],
                },
                "probs": probs,
                "pred": pred,
                "confidence": conf,
            }

            try:
                out.insert_one(doc)
                n_ok += 1
                print(f"[ok] saved {n_ok} :: {provider}/{model} pred={pred} conf={conf:.3f}")
            except Exception as e:
                n_skip += 1
                print(f"[skip] duplicate or insert error :: {type(e).__name__} {e}")

            time.sleep(SLEEP_SECONDS)

    print(f">>> done. ok={n_ok}, skip={n_skip}, fail={n_fail}")

if __name__ == "__main__":
    main()
