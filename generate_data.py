import os
import hashlib
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient
import certifi
import re


from openai import OpenAI
from anthropic import Anthropic
from google import genai




load_dotenv()

#define your concept here
LABEL_SPECS = {
    "Position": {
        "concept": "Position",
        "definition": "An opinion or conclusion on the main question",
        "examples": [
            " In my opinion, every individual has an obligation to think seriously about important matters, although this might be difficult."
            "Drivers should not be able to use phones while operating a vehicle. ",
            "However, this device is taken to areas that it is irresponsible and dangerous. "
        ]
    },
    "Lead": {
        "concept": "Lead",
        "definition": "The introduction begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis.",
        "examples": [
            "What would you choose, thousands of screaming fans beckoning you to perform your guitar solo or a quiet shelf in a library with only a couple hundred pages detailing your life. It is the sad choice between being a celebrity on one hand and a hero on the other.",
            "Modern humans today are always on their phone. They are always on their phone more than 5 hours a day no stop .All they do is text back and forward and just have group Chats on social media. They even do it while driving. ",
            "The ability to stay connected to people we know despite distance was originally brought to fruition by the use of letters. This system was found to be rather slow and new pathways were searched for until the invention of the telegram; the people thought it was an invention of the millennia, but after time it too was thought of as slow until the invention of the telephone. Today, a telephone is in the hand or pocket of a majority of the seven billion people on planet earth "

        ]
    },


    "Claim": {
        "concept": "Claim",
        "definition": "A claim that supports the position.",
        "examples": [
            "The next reason why I agree that every individual has an obligation to think seriously about important matters is that this simple task can help each person get ahead in life and be successful."
            "Drivers who used their phone while operating a vehicle are most likely to get into an accident that could be fatal. "
            "The last reason you shouldn't drive while using a cellphone because it can cause fatal injuries that might have long-term or short-term affects. "
        ]
    },

    "Counterclaim": {
        "concept": "Counterclaim",
        "definition": "A claim that refutes another claim or gives an opposing reason to the position.",
        "examples": [
            "Some may argue that obligating every individual to think seriously is not necessary and even annoying as some people may choose to just follow the great thinkers of the nation.",
            "While it is dangerous to be interacting with your phone while driving some people have no choice in the matter. "
            "Now, I know what you probably saying 'But what if everyone you ask gives you incorrect advice?'"
            
        ]
    },

    "Rebuttal": {
        "concept": "Rebuttal",
        "definition": "A claim that refutes a counterclaim.",
        "examples": [
            "Even though people can follow others' steps without thinking seriously in some situations, the ability to think critically for themselves is a very important survival skill."
            "what are the odds that seven of your close friends do not no what you are talking about? "
        ]
    },

    "Evidence": {
        "concept": "Evidence",
        "definition": "Ideas or examples that support claims, counterclaims, or rebuttals.",
        "examples": [
            "For instance, the presidential debate is currently going on. In order to choose the right candidate, voters need to research all sides of both candidates and think seriously to make a wise decision for the good of the whole nation."
            "My family could not decide where to go for vacation, so I went and asked my uncle, who used to traveled the world for photography, for his opinion. After talking for a long time he gave me a great deal of information on where and where to not go for vacation. "
        ]
    },

    "Concluding Statement ": {
        "concept": "Concluding Statement ",
        "definition": "A concluding statement that restates the claims.",
        "examples": [
            "To sum up, thinking seriously is important in making decisions because each decision has an outcome that affects lives. It is also important because if you think seriously it can help you succeed",
            "The odds are in your favor; and the on the off chance that the odds are not in you favor? Just expand your focus group, ask new people, be confident. You just need to ask more people. Now go ask those questions, and get some answers. "
            "In conclusion, asking advice can help someone make a better decision because they can see the persons point of view, more experienced, and they can see which advice is better. Asking for advice is important because they can see which¬†advice is better and can see other peoples perspective. This affects you because its on you to ask for advice and make the right decision! "
        ]
    },


}


#define the model here
MODEL_LIST = [
    {"provider": "anthropic", "model": "claude-opus-4-6"},
    {"provider": "google", "model": "gemini-3-pro-preview"},
    {"provider": "openai", "model": "gpt-5.2"}
    ]


TASK = "argumentative_style"
N_SAMPLES = 10
TEMPERATURE = 0.9


uri = os.environ["MONGODB_URI"]
client_db = MongoClient(
    uri,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=8000
)

db = client_db["distillena"]
col = db["generated_text"]
try:
    col.create_index("hash", unique=True)
except Exception as e:
    try:
        from pymongo.errors import OperationFailure
    except Exception:
        OperationFailure = None

    if OperationFailure and isinstance(e, OperationFailure) and getattr(e, "code", None) == 86:
        print("Index 'hash' already exists (conflict); continuing.")
    else:
        raise



#api_setting
client_gpt = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
gemini_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


def build_prompt(label_name: str, spec: dict) -> str:
    examples_text = "\n".join(f"- {ex}".strip() for ex in spec.get("examples", []) if ex)

    prompt = f"""
        You are a student writing genetator, trying to write 20 sentences that match the following condition, seperate them with ";". DO NOT number the sentence
        you must vary topic, tone, and structure across everyday student essay contexts, anything, be very creative.
        Return ONLY the sentence. No explanation.

        Label: {label_name}
        Concept: {spec.get("concept","")}
        Definition: {spec.get("definition","")}

        Example sentences:
        {examples_text}

        Task:
        Write a new sentence with 12–25 words that fits the label above, 
        as it might naturally appear in a student’s written response. 
""".strip()

    return str(prompt)  


#encoding
def sha1_text(s: str) -> str:
    """
    Generate a SHA-1 hash for a text string.
    Used for deduplication of generated samples.
    """
    return hashlib.sha1(s.strip().encode("utf-8")).hexdigest()


def generate_text(model_cfg: dict, prompt: str, temperature: float) -> str:
    provider = model_cfg["provider"]
    model = model_cfg["model"]

    if provider == "openai":
        resp = client_gpt.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            frequency_penalty = 0.8,
            presence_penalty = 0.5
        )
        return (resp.choices[0].message.content or "").strip()

    if provider == "anthropic":
        msg = anthropic_client.messages.create(
            model=model,
            max_tokens=256,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in msg.content if block.type == "text"
        ).strip()

    if provider == "google":
        r = gemini_client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": temperature},
        )
        return (r.text or "").strip()

    raise ValueError(f"Unknown provider: {provider}")



def main():
    print(">>> script started")
    print("MODEL_LIST len =", len(MODEL_LIST))
    print("MODEL_LIST =", MODEL_LIST)

    for model_cfg in MODEL_LIST:
        provider = model_cfg["provider"]
        model = model_cfg["model"]
        print(f"\n### MODEL: {provider} / {model} ###")

        for label_name, spec in LABEL_SPECS.items():
            prompt = build_prompt(label_name, spec)
            print(f"\n=== Generating: {label_name} ===")

            for i in range(N_SAMPLES):
                try:
                    raw = generate_text(model_cfg, prompt, TEMPERATURE)
                    print(f"  [ok] {provider}/{model} {label_name} iter {i+1}/{N_SAMPLES}")
                    print(f"      Generated {len(raw)} chars")
                except Exception as e:
                    print(f"  [fail] {provider}/{model} {label_name} iter {i+1}/{N_SAMPLES}")
                    print(f"      Error: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                sentences = [s.strip() for s in re.split(r";\s*", raw) if s.strip()]
                # and i want to save each sentence separately, not the whole text blob
                

                
                for text in sentences:
                    doc = {
                        "label": label_name,
                        "concept": spec["concept"],
                        "text": text,
                        "hash": sha1_text(text),
                        "model": model_cfg["model"],
                        "temperature": TEMPERATURE,
                    }

                    try:
                        col.insert_one(doc)
                        print("  -> saved")
                    except Exception as e:
                        print("  -> skipped:", e)

if __name__ == "__main__":
    main()