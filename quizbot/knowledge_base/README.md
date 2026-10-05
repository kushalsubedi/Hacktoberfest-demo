# Knowledge Base

This folder is Quizzy's long-term memory. Every note the bot saves lands in
`memories.jsonl` — one JSON object per line, written by the `remember` tool and
read back by the `recall` tool (see `src/quizbot/knowledge.py`).

## The problem: LLMs have no memory

An LLM is stateless. It only "knows" what is inside the current conversation's
message list. The moment you close the program, that list is gone.

**Without this knowledge base:**

```
$ quizbot
You: hi, I'm Kushal, I love science questions
Quizzy: Welcome Kushal! Science it is! ...
You: quit

$ quizbot          <- restart
You: hi, it's me again
Quizzy: Welcome, mystery contestant! What's your name?   <- total amnesia
```

Nothing carried over — not the player's name, not their score, not that they
prefer science. The conversation history lived only in a Python list
(`messages`) that died with the process.

## How this solves it

We give the model two tools backed by a file on disk:

| Tool | What it does |
|------|--------------|
| `remember(note)` | Appends a timestamped note to `memories.jsonl` |
| `recall()` | Returns every saved note |

The system prompt tells Quizzy to call `recall` at the start of a conversation
and `remember` whenever it learns something worth keeping (names, final
scores, preferences).

**With the knowledge base:**

```
$ quizbot          <- restart, same as before
You: hi, it's me again
  [running recall({})]
Quizzy: Kushal! Back for more science, I presume? Last time you scored 6 —
        think you can beat it?
```

The model is still stateless — nothing changed about the LLM itself. What
changed is that it can now *pull* its past out of a file before answering,
and *push* new facts in as they happen.

## Why this matters beyond a quiz bot

This is the same pattern every "AI that remembers you" uses: persistent
storage outside the model + tools to read and write it. Here the store is a
plain JSONL file and `recall` returns everything, which is fine at this
scale. When the store grows too big to fit in the prompt, you add search —
retrieve only the *relevant* notes (keyword or vector search). At that point
this stops being plain tool use and becomes retrieval-augmented generation
(RAG) done agentically: the model decides when to search its own memory.
