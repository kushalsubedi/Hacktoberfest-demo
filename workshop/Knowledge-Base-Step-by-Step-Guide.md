# Give Your AI a Memory: Step-by-Step Guide

Hacktoberfest 2026 · Kushal Subedi · Oct 6, 2026

## Introduction: the problems, and the tools that solve them

By itself, a large language model is just a very good text predictor — and that comes with three built-in problems. Each one has a standard fix, and this workshop builds all three:

| Problem with a plain LLM | The fix | Where you build it |
| --- | --- | --- |
| It can't *do* anything — no internet, no files, no actions. It will happily invent trivia questions (hallucination). | **Tools / function calling** — your code does the action, the model decides when | QuizBot |
| It doesn't know *your* data — your notes, your docs, anything after its training cutoff. | **RAG** — retrieve the relevant text and put it in the prompt | NotesBot |
| It's **stateless** — close the program and it forgets everything, including your name. | **A knowledge base** — memory stored outside the model, with tools to read and write it | MemBot (this guide) |

### Terminology

- **LLM** — large language model; a function from text to text. It predicts, it doesn't know or remember.
- **Prompt / system prompt** — the text you send. The system prompt is the standing instruction that sets behaviour ("You are MemBot…").
- **Context window** — everything the model can see for one call: system prompt + conversation so far. If it isn't in there, the model doesn't know it.
- **Stateless** — the model keeps nothing between calls. "Memory" in a chat is just the growing `messages` list being re-sent every turn.
- **Hallucination** — the model confidently inventing facts. Tools and RAG both fight it by grounding answers in real data.
- **Tool (function calling)** — an ordinary function in your code, described to the model. The model *asks* for it to run; your code runs it and sends back the result.
- **Agent** — a loop where the model can call tools, read the results, and decide what to do next.
- **RAG (retrieval-augmented generation)** — fetch the text relevant to the question, put it in the prompt, then generate the answer from it.
- **Knowledge base** — persistent storage outside the model (here: one JSONL file in a folder) plus the tools to read and write it.

## What we're building

The third and final part, after QuizBot and NotesBot. We build **MemBot**: the smallest possible bot with a **knowledge base** — a folder on disk where the AI saves what it learns about you, so it still knows it after you close the program.

The whole session is one trick, shown twice:

- MemBot asks **"What's your name?"** You tell it. You quit. You start it again and ask **"what was my name?"** — and it has no idea (Step 2). An LLM is stateless: close the program and everything is gone.
- Then we give it two tools, `remember` and `recall`, backed by one file in a `knowledge_base/` folder (Steps 3–4). Restart, ask again — **it remembers** (Step 5).
- Finally we bolt the same two tools onto QuizBot and NotesBot (Step 6), because the fix is ten lines and works anywhere.

Every step follows the same rhythm:

1. **Add** the few lines shown (only the new code, never the whole file).
2. **Run** `uv run main.py` and look at the output.
3. **Try it**: a small experiment that changes one thing, so you see the idea for yourself.
4. **Takeaway:** the one sentence to remember.

| Step | You add | What you learn | Time |
| --- | --- | --- | --- |
| 0 | Setup with `uv` | Same drill as QuizBot | 3 min |
| 1 | A chat loop that asks your name | In-session memory is just a list | 5 min |
| 2 | Nothing — restart and ask again | The list dies with the process | 4 min |
| 3 | `remember` and `recall` functions | A knowledge base is a file + two functions | 6 min |
| 4 | Tool descriptions + the tool loop | Same wiring as QuizBot, new tools | 6 min |
| 5 | Nothing — restart and ask again | The payoff: memory across restarts | 4 min |
| 6 | The same tools in QuizBot & NotesBot | The pattern is portable | 6 min |

## Step 0: Setup with uv

Same drill as QuizBot and NotesBot. If you still have your Gemini key in a `.env` from those, copy it over.

```bash
uv init membot
cd membot
uv add openai python-dotenv
```

**Create `.env`** with one line (and add `.env` to `.gitignore`):

```
GEMINI_API_KEY=paste-your-key-here
```

**Takeaway:** three commands and a key — you've done this twice already.

## Step 1: A chat loop that asks your name

Replace everything in `main.py` with:

```python
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-3.8-flash"

messages = [{"role": "system", "content":
             "You are MemBot, a friendly assistant. Be brief."}]

print("MemBot: Hi! What's your name?")
while True:
    user = input("\nYou: ")
    if user in ("quit", "exit"):
        break
    messages.append({"role": "user", "content": user})
    reply = client.chat.completions.create(model=MODEL, messages=messages)
    msg = reply.choices[0].message
    messages.append({"role": "assistant", "content": msg.content})
    print("MemBot:", msg.content)
```

**Run it:**

```
$ uv run main.py
MemBot: Hi! What's your name?
You: I'm Kushal
MemBot: Nice to meet you, Kushal!
You: what's my name?
MemBot: Your name is Kushal!
```

It "remembers" within the conversation — because every turn is appended to `messages` and the whole list is sent on every call. You saw this in QuizBot Step 3.

**Try it:** ask three unrelated questions, then ask your name again. Still knows it — it's all in the list.

**Takeaway:** in-session memory is nothing magical: it's a Python list that grows.

## Step 2: Restart it — total amnesia

Add nothing. Quit MemBot (`quit`), then start it again:

```
$ uv run main.py
MemBot: Hi! What's your name?
You: what was my name?
MemBot: I'm sorry, I don't know your name — you haven't told me yet!
```

It's not being difficult. The `messages` list was a variable in a process that no longer exists. The model itself stores nothing between calls — every conversation starts from a blank slate.

**Try it:** restart twice more. Same amnesia every time. This is every LLM, every provider — not a Gemini quirk.

**Takeaway:** an LLM is stateless; when the process dies, the memory dies with it.

## Step 3: A knowledge base is a file + two functions

The fix lives *outside* the model: a folder on disk, and two ordinary Python functions. Add above `messages`:

```python
import json
import time
from pathlib import Path

KB_FILE = Path("knowledge_base") / "memories.jsonl"


def remember(note: str) -> dict:
    """Save a note to the long-term knowledge base. Survives restarts."""
    KB_FILE.parent.mkdir(exist_ok=True)
    with KB_FILE.open("a") as f:
        f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M"), "note": note}) + "\n")
    return {"saved": note}


def recall() -> dict:
    """Return everything saved in the knowledge base."""
    if not KB_FILE.exists():
        return {"memories": []}
    with KB_FILE.open() as f:
        return {"memories": [json.loads(line) for line in f if line.strip()]}
```

No AI here — just like `get_trivia` in QuizBot, a tool is ordinary code.

**Try it** before wiring anything, straight from Python:

```bash
uv run python -c "from main import remember, recall; remember('test note'); print(recall())"
```

You'll see your note come back — and a new `knowledge_base/memories.jsonl` file you can open in any editor. One JSON object per line. That file *is* the knowledge base.

**Takeaway:** a knowledge base can be this small: append to a file, read the file back.

## Step 4: Describe the tools and wire the loop

Same wiring as QuizBot Steps 5–7, with our two new tools. Add above `messages`:

```python
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
            "description": "Read everything in the knowledge base. Call it at the "
                           "start of a conversation to recognise returning users.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_tool(call) -> dict:
    name = call.function.name
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        args = json.loads(call.function.arguments)
        return TOOLS[name](**args)
    except Exception as e:
        return {"error": str(e)}
```

Tell MemBot to use them — replace the system message:

```python
messages = [{"role": "system", "content":
             "You are MemBot, a friendly assistant. Be brief. "
             "At the start of a conversation call recall to check if you know the user. "
             "When you learn their name or anything worth keeping, call remember. "
             "If a tool returns an error, say so honestly."}]
```

And replace the body of the chat loop (everything after `messages.append({"role": "user", ...})`) with the tool loop you know from QuizBot Step 7:

```python
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
```

**Run it** and introduce yourself:

```
$ uv run main.py
MemBot: Hi! What's your name?
You: I'm Kushal, and I love science
  [running recall({})]
  [running remember({"note": "The user's name is Kushal and they love science"})]
MemBot: Great to meet you, Kushal! Noted — science fan.
You: quit
```

**Try it:** open `knowledge_base/memories.jsonl`. Your name is in there, in plain text, written by the model *choosing* to call `remember`.

**Takeaway:** the model never touches the disk — it asks, your code writes. Same contract as every tool.

## Step 5: The payoff — restart and ask again

Add nothing. Restart, and ask the exact question that failed in Step 2:

```
$ uv run main.py
MemBot: Hi! What's your name?
You: what was my name?
  [running recall({})]
MemBot: You're Kushal — the science fan! Welcome back.
```

Same model, still stateless. What changed: before answering, it *pulled* its past out of a file, because we gave it a tool to do so and a system prompt that says to use it.

**Try it:** delete the `knowledge_base` folder and restart — amnesia is back. The memory was never in the model; it was always in the folder.

**Takeaway:** "an AI that remembers you" = stateless model + a store outside it + tools to read and write that store.

## Step 6: Add it to our two projects

The pattern is portable: the same two functions, two descriptions, and one sentence in the system prompt drop into anything with a tool loop.

**QuizBot:** paste `remember`, `recall`, their two `tools` entries, add them to `TOOLS`, and extend Quizzy's system prompt with:

```
At the start of a conversation call recall to recognise returning players,
and use remember to save player names, final scores and preferences.
```

Now Quizzy greets returning players by name and taunts them with last session's score.

**NotesBot:** same drop-in — and notice what you've built. `recall` returns *everything*, which is fine while the file is small. When the knowledge base grows too big for the prompt, you'll want to fetch only the *relevant* memories for the current question… chunking, embeddings, retrieval — exactly what NotesBot does to `notes.md`. A knowledge base at scale **is** RAG over your own memories; the model deciding when to search it is what people call *agentic RAG*.

**Takeaway:** QuizBot taught tools, NotesBot taught retrieval, and a knowledge base is where the two meet.
