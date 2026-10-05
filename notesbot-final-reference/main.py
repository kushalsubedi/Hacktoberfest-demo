import hashlib
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"


def load_chunks(path="notes.md"):
    text = open(path, encoding="utf-8").read()
    parts = text.split("\n## ")
    return [parts[0].strip()] + ["## " + p.strip() for p in parts[1:]]


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (sum(x * x for x in a) ** 0.5 * sum(x * x for x in b) ** 0.5)


def embed_cached(chunks, cache_path="embeddings.json"):
    key = hashlib.sha256("\n".join(chunks).encode()).hexdigest()
    if os.path.exists(cache_path):
        data = json.load(open(cache_path))
        if data.get("key") == key:
            return data["vectors"]
    vectors = embed(chunks)
    json.dump({"key": key, "vectors": vectors}, open(cache_path, "w"))
    return vectors


chunks = load_chunks()
chunk_vectors = embed_cached(chunks)


def retrieve(question, k=3):
    q_vec = embed([question])[0]
    scored = sorted(zip(chunks, chunk_vectors),
                    key=lambda cv: similarity(q_vec, cv[1]), reverse=True)
    return [c for c, _ in scored[:k]]


SYSTEM = ("You answer questions about the CodeHub student club using the "
          "provided notes. You may combine facts across sections and do simple "
          "reasoning or arithmetic from them — e.g. an academic year is two "
          "semesters, so a yearly cost is twice the semester fee — and you may "
          "summarize or infer general points (like what the club is about or "
          "its values) from specific sections. Only say you don't know when "
          "the notes offer nothing relevant. Never invent facts not grounded "
          "in the notes.")


def answer(question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"NOTES:\n{context}\n\nQUESTION: {question}"},
        ],
    )
    return reply.choices[0].message.content


print(f"NotesBot ready — {len(chunks)} chunks loaded. Ask me about the club!")
while True:
    question = input("\nYou: ").strip()
    if question.lower() in ("quit", "exit", ""):
        break
    context_chunks = retrieve(question)
    print("  [sources:", " | ".join(c.splitlines()[0] for c in context_chunks) + "]")
    print("Bot:", answer(question, context_chunks))
