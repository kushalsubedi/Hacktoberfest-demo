import os

from dotenv import load_dotenv
from openai import OpenAI


import json
import time
from pathlib import Path

KB_FILE = Path("knowledge_base") / "memories.jsonl"


load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-3.8-flash"
EMBED_MODEL = "gemini-embedding-001"


def embed(text: str) -> list[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))


def remember(note: str) -> dict:
    """Save a note to the long-term knowledge base. Survives restarts."""
    KB_FILE.parent.mkdir(exist_ok=True)
    with KB_FILE.open("a") as f:
        f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M"),
                            "note": note, "vec": embed(note)}) + "\n")
    return {"saved": note}


def recall(query: str) -> dict:
    """Return the top 5 memories most relevant to the query."""
    if not KB_FILE.exists():
        return {"memories": []}
    with KB_FILE.open() as f:
        memories = [json.loads(line) for line in f if line.strip()]
    qvec = embed(query)
    scored = sorted(memories, key=lambda m: cosine(qvec, m["vec"]), reverse=True)
    return {"memories": [{"when": m["when"], "note": m["note"]} for m in scored[:5]]}

TOOLS = {"remember": remember, "recall": recall}

tools = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a note to the long-term knowledge base (names, "
                           "preferences, facts about the user). It survives restarts.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search the knowledge base for memories relevant to a "
                           "query. Call it at the start of a conversation to "
                           "recognise returning users.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string",
                               "description": "What you want to look up, e.g. 'user name'"}},
                "required": ["query"],
            },
        },
    },
]

messages = [{"role": "system", "content":
             "You are MemBot, a friendly assistant. Be brief. "
             "At the start of a conversation call recall to check if you know the user. "
             "When you learn their name or anything worth keeping, call remember. "
             "If a tool returns an error, say so honestly."}]
def run_tool(call) -> dict:
    name = call.function.name
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        args = json.loads(call.function.arguments)
        return TOOLS[name](**args)
    except Exception as e:
        return {"error": str(e)}

print("MemBot: Hi! What's your name?")
while True:
    user = input("You: ").strip()
    if user.lower() in ("quit", "exit"):
        break
    messages.append({"role": "user", "content": user})
    reply = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    msg = reply.choices[0].message

    steps = 0
    while msg.tool_calls and steps < 5:
        steps += 1
        messages.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            print(f"  [running {call.function.name}({call.function.arguments})]")
            result = run_tool(call)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)})
        reply = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        msg = reply.choices[0].message

    messages.append({"role": "assistant", "content": msg.content})
    print("MemBot:", msg.content)



