# Build Your Own RAG System: Step-by-Step Guide

Hacktoberfest 2026 · Kushal Subedi · Oct 6, 2026

## What we're building

We build **NotesBot** live: a tiny RAG (Retrieval-Augmented Generation) system that answers questions from a markdown file the AI has never seen. This is the same idea behind "chat with your PDF", documentation bots and company knowledge assistants — in about 60 lines of Python, no vector database, no framework.

It lives in two files: `notes.md` (the knowledge) and `main.py` (the code). We use Gemini and `uv`, exactly like QuizBot.

The story of the session, in one line each:

- The model **doesn't know** your data (Step 1).
- So you **put your data in the prompt** (Step 2) — that's all RAG is.
- But whole files don't scale, so you **chunk** (Step 3), **embed** (Step 4) and **retrieve** only what's relevant (Step 5).
- Then you **generate** an answer from the retrieved chunks (Step 6) and wrap it in a chat loop with guardrails (Step 7).

Every step follows the same rhythm:

1. **Add** the few lines shown (only the new code, never the whole file).
2. **Run** `uv run main.py` and look at the output.
3. **Try it**: a small experiment that changes one thing, so you see the idea for yourself.
4. **Takeaway**: the one sentence to remember.

| Step | You add | What you learn | Time |
| --- | --- | --- | --- |
| 0 | Setup with `uv` + `notes.md` | Project setup, the knowledge file | 6 min |
| 1 | One question the model can't answer | LLMs don't know your data | 5 min |
| 2 | The whole file in the prompt | RAG is just "right text in the prompt" | 6 min |
| 3 | Split the notes into chunks | Chunking | 5 min |
| 4 | Embeddings + a similarity function | Meaning as numbers | 8 min |
| 5 | Retrieve the top chunks for a question | Search by meaning — no AI answer yet | 7 min |
| 6 | Retrieved chunks + question → answer | The full RAG pipeline | 7 min |
| 7 | Chat loop, sources, "I don't know" | Grounding and guardrails | 6 min |

Tip for the live session: keep this guide on a second screen and type the code yourself rather than pasting. Typing slows you to the pace attendees can follow.

## Step 0: Setup with uv

Same drill as QuizBot. If `uv` and your Gemini key are already set up, this step is three commands and one copy-paste.

**Install uv** (once):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Get a Gemini key** (once): [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → sign in → create key → copy.

**Create the project:**

```bash
uv init notesbot
cd notesbot
uv add openai python-dotenv
```

**Create `.env`** in the `notesbot` folder (and add `.env` to `.gitignore`):

```bash
GEMINI_API_KEY=paste-your-key-here
```

**Create `notes.md`** — the knowledge file. This is a handbook for a fictional student club, full of specific facts no AI model was ever trained on. Paste this in exactly:

```markdown
# CodeHub Student Club Handbook

CodeHub is the student coding club. This handbook is the single source of
truth for how the club runs. Last updated for the 2026/27 academic year.

## Membership

Membership costs Rs. 500 per semester, payable at the front desk of the
computer lab. Members get access to all weekly meetups, the equipment loan
program, and priority registration for the annual hackathon. To join, fill
out the form at the lab desk and bring your student ID card.

## Weekly Meetups

Meetups run every Friday from 4:00 to 6:00 PM in Room B204. The first hour
is a talk or workshop by a member or guest; the second hour is open project
time. No registration needed — just show up. Snacks are provided on the
last Friday of each month.

## Annual Hackathon

The hackathon runs for 36 hours during the second weekend of March. Teams
have a maximum of 4 members and at least half of each team must be current
club members. Registration closes two weeks before the event. The top three
teams receive sponsored prizes, and the winning project gets presented at
the department open day.

## Equipment Loans

Members can borrow a laptop, Arduino kit, or Raspberry Pi for up to 3 days
at a time. Leave your student ID as a deposit at the lab desk. Late returns
suspend borrowing rights for one month. Equipment must be returned in the
condition it was lent — report any damage immediately rather than hiding it.

## Project Showcase

Every semester ends with a project showcase in the main auditorium. Any
member can present, solo or in a team. Presentations are 5 minutes plus
2 minutes of questions. Sign-up opens one month before the showcase and
slots are first-come, first-served, capped at 20 projects.

## Code of Conduct

Be respectful in all club spaces, online and offline. No harassment of any
kind is tolerated. First violation gets a written warning from the
committee; a second violation means removal from the club with no refund
of the membership fee.

## Committee

The committee has five roles: President, Vice President, Treasurer,
Events Lead, and Tech Lead. Elections happen in the first week of the
autumn semester, and any member who has been in the club for at least one
full semester can run. Contact the committee at codehub@campus.example.
```

**Takeaway:** the knowledge lives in a plain markdown file, the key lives in `.env`, and `uv` handles everything else.

### Check: your folder should look like this

```text
notesbot/
├── .env              ← your Gemini key (never share it)
├── .gitignore        ← add .env here
├── .venv/            ← made by uv, don't touch
├── main.py           ← THE file you edit in every step
├── notes.md          ← the knowledge file (paste from above)
├── pyproject.toml
└── uv.lock
```

**Where code goes in `main.py`.** By Step 7, the file reads top to bottom like this:

```text
main.py
├── 1. imports + client + models     (Step 1)
├── 2. load_chunks()                 (Step 3)
├── 3. embed() + similarity()        (Step 4)
├── 4. chunks + chunk_vectors        (Step 5)
├── 5. retrieve()                    (Step 5)
├── 6. SYSTEM + answer()             (Step 6)
└── 7. while True: chat loop         (Step 7)
```

If you get lost, compare against this map: functions sit at the top, the chat loop is always last.

## Step 1: One question the model can't answer

First, let's prove the problem exists.

**Replace** everything in `main.py` with:

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

question = "How much does CodeHub student club membership cost per semester?"
reply = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": question}],
)
print(reply.choices[0].message.content)
```

**Run** `uv run main.py`.

The model either admits it has no idea which CodeHub you mean, or — worse — confidently invents a price. Both are useless. The answer (Rs. 500) is sitting in `notes.md`, one folder away, and the model can't see it.

**Try it:** ask "When does hackathon registration close?" Same failure. Now ask "What is the capital of France?" — that it knows, because it was in the training data.

**Takeaway:** an LLM only knows its training data. Your files, your company's docs, yesterday's news — invisible. Everything that follows is about fixing exactly this.

## Step 2: The whole file in the prompt

The fix is almost embarrassingly simple: put the notes *in the prompt*.

**Replace** the `question = ...` line and everything below it with:

```python
notes = open("notes.md", encoding="utf-8").read()

question = "How much does CodeHub student club membership cost per semester?"
reply = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": f"Answer the question using these notes.\n\n"
                                          f"NOTES:\n{notes}\n\nQUESTION: {question}"}],
)
print(reply.choices[0].message.content)
```

**Run** it. Correct answer: Rs. 500 per semester, paid at the lab desk.

This already *is* retrieval-augmented generation: we retrieved text (all of it) and the model generated an answer from it. Every RAG system, however fancy, ends at this exact moment — relevant text pasted above a question.

So why isn't the tutorial over? Scale. Our handbook is one page. A company wiki is 50,000 pages: it won't fit in the context window, you'd pay to process all of it on *every* question, and models get worse at finding facts buried in huge prompts. We need to send only the *relevant* part.

**Try it:** change the question to something about equipment loans — still correct, because everything is in the prompt. That luxury is what we're about to lose and win back.

**Takeaway:** RAG = put the right text in the prompt. The whole engineering problem is the word *right*.

## Step 3: Split the notes into chunks

To send only the relevant part, the file first needs *parts*. Our markdown already has natural seams: the `## ` headings.

**Replace** the `notes = ...` line and everything below it with:

```python
def load_chunks(path="notes.md"):
    text = open(path, encoding="utf-8").read()
    parts = text.split("\n## ")
    return [parts[0].strip()] + ["## " + p.strip() for p in parts[1:]]


chunks = load_chunks()
print(f"{len(chunks)} chunks")
for c in chunks:
    print("-", c.splitlines()[0])
```

**Run** it:

```text
8 chunks
- # CodeHub Student Club Handbook
- ## Membership
- ## Weekly Meetups
- ## Annual Hackathon
- ## Equipment Loans
- ## Project Showcase
- ## Code of Conduct
- ## Committee
```

Each chunk is one self-contained topic. That's the entire art of chunking: pieces small enough to be cheap, big enough to still make sense alone. Real systems split PDFs by paragraph or fixed token counts with overlap — same idea, messier seams.

**Try it:** `print(chunks[3])` to see one full chunk. Notice it would answer any hackathon question on its own, without the rest of the file.

**Takeaway:** chunking turns one big document into retrievable pieces; good boundaries (headings, paragraphs) keep each piece meaningful.

## Step 4: Embeddings — meaning as numbers

Now the magic ingredient. To find which chunk matches a question, we need to compare *meaning*, not keywords — "How much do I pay?" should match the Membership chunk even though it shares no words with it.

An **embedding model** turns any text into a long list of numbers (a vector) where *similar meanings land close together*. It's a second, cheaper model next to the chat model.

**Add** below the client setup (keep `MODEL` where it is):

```python
EMBED_MODEL = "gemini-embedding-001"


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (sum(x * x for x in a) ** 0.5 * sum(x * x for x in b) ** 0.5)
```

`similarity` is cosine similarity in three lines of plain Python: 1.0 means identical direction (same meaning), near 0 means unrelated.

### The theory in two pictures

**Picture 1: an embedding is a point in space.** The embedding model maps every text to a vector — a list of ~3,000 numbers, which you can think of as coordinates of a point in a 3,000-dimensional space. The model was trained so that *texts with similar meaning land near each other*. Squashed down to 2D, our three example sentences look like this:

```text
 meaning-
 axis 2
   ▲
   │                       ● "How do I borrow a laptop?"
   │                     ● "What is the process for
   │                        taking equipment home?"
   │        (close together = similar meaning,
   │         even with zero shared words)
   │
   │   ● "Momo is the best dumpling."
   │     (far away = different topic)
   │
   └──────────────────────────────────▶ meaning-axis 1
```

The axes don't mean anything by themselves — no single number is "the food dimension". Meaning lives in the *geometry*: which points sit near which. That's the whole trick, and it turns "find relevant text" into "find the nearest points".

**Picture 2: cosine similarity is the angle between two vectors.** Each point is also an arrow from the origin. To compare two texts we ask: do their arrows point the same way? Cosine similarity is literally the cosine of the angle between them:

```text
  similar meaning:                unrelated meaning:
  small angle, cos ≈ 1            big angle, cos ≈ 0

      ▲ laptop                        ▲ laptop
     ╱▲ equipment                     │
    ╱╱                                │   ╱ momo
   ╱╱  θ ≈ 15°                        │  ╱
  ╱╱   cos θ ≈ 0.96                   │ ╱   θ ≈ 75°
 ●                                    ●╱    cos θ ≈ 0.26
```

The scale:

| cos θ | angle | meaning |
| --- | --- | --- |
| 1.0 | 0° — same direction | same meaning |
| ~0.5 | ~60° | loosely related |
| ~0.0 | 90° — perpendicular | unrelated |
| -1.0 | 180° — opposite | opposite (rare with real embeddings) |

The formula, and how it maps onto the three lines of `similarity`:

```text
                 A · B           ← dot:  sum(x * y for x, y in zip(a, b))
cos θ  =  ─────────────────
           |A| × |B|             ← the two sum(x*x) ** 0.5 terms
```

The numerator (dot product) is big when the vectors point the same way; dividing by the two lengths normalizes it, so only *direction* counts, not *length*. That's why cosine beats plain distance here: a long paragraph and a short question get vectors of different magnitudes, but if they're about the same thing they point the same way — and cosine only measures the pointing.

**Replace** the printing code at the bottom with:

```python
v = embed(["How do I borrow a laptop?",
           "What is the process for taking equipment home?",
           "Momo is the best dumpling."])
print("vector length:", len(v[0]), "| first numbers:", [round(x, 3) for x in v[0][:4]])
print("laptop vs equipment:", round(similarity(v[0], v[1]), 3))
print("laptop vs momo:     ", round(similarity(v[0], v[2]), 3))
```

**Run** it. The two equipment questions score high with each other despite sharing almost no words; the dumpling sentence scores clearly lower. The model measured *meaning*.

**Try it:** replace the momo sentence with "Where can I get a laptop on loan?" and watch the score jump near the top.

**Takeaway:** embeddings turn text into vectors where similar meaning = nearby numbers, so "find relevant text" becomes arithmetic.

## Step 5: Retrieval — search by meaning

Combine Steps 3 and 4: embed every chunk once, embed the question, and keep the closest chunks. This is the R in RAG — and notice there's no chat model anywhere in it.

**Replace** the experiment code at the bottom with:

```python
chunks = load_chunks()
chunk_vectors = embed(chunks)


def retrieve(question, k=2):
    q_vec = embed([question])[0]
    scored = sorted(zip(chunks, chunk_vectors),
                    key=lambda cv: similarity(q_vec, cv[1]), reverse=True)
    return [c for c, _ in scored[:k]]


for c in retrieve("How much do I need to pay to join?"):
    print("MATCH:", c.splitlines()[0])
```

**Run** it. `## Membership` comes out on top — for a question that never says "membership", "fee", or "cost" in the chunk's words.

The `k` in `retrieve(question, k=2)` is the only knob here: **how many of the best-scoring chunks to keep**. Too small and the answer might sit in chunk #k+1, just below the cut; too big and you're pasting the whole file again — more cost, more noise, no gain. For an 8-chunk handbook, 2 is plenty. Real systems typically retrieve 3–10 and let a reranker trim further.

A **vector database** (Pinecone, Chroma, pgvector...) is exactly this `sorted(...)` line made fast for millions of chunks, plus persistence so you don't re-embed on every start. For 8 chunks, a Python list *is* the vector database.

**Try it:** change the question to "Can my team of 6 enter the hackathon?" and check the top match. Then try "What happens if I'm rude to someone?" — Code of Conduct, found purely by meaning.

**Takeaway:** retrieval = embed the question, rank chunks by similarity, take the top few. A vector database is this loop at scale.

## Step 6: Generation — the full pipeline

Now bolt Step 5's output onto Step 2's prompt. Retrieved chunks in, grounded answer out: that's the complete RAG pipeline.

**Replace** the `for c in retrieve(...)` test at the bottom with:

```python
SYSTEM = ("You answer questions about the CodeHub student club using ONLY the "
          "provided notes. If the notes don't contain the answer, say you don't "
          "know — never guess.")


def answer(question):
    context = "\n\n---\n\n".join(retrieve(question))
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"NOTES:\n{context}\n\nQUESTION: {question}"},
        ],
    )
    return reply.choices[0].message.content


print(answer("How much does membership cost per semester?"))
```

**Run** it. Correct answer again — but this time the model saw only the 2 most relevant chunks, not the whole file. Same quality as Step 2 at a fraction of the size, and it would still work if the handbook were 10,000 pages.

Say the pipeline out loud once, because it's the whole field in one breath:

**question → embed → rank chunks → top-k → prompt → answer**

**Try it:** inside `answer`, add `print(">>> context sent:", context[:200])` before the API call to watch exactly what the model receives. Seeing the prompt demystifies the entire system.

**Takeaway:** RAG = retrieval (Step 5) + generation (Step 2). You've now built both halves yourself.

## Step 7: Chat loop, sources, and "I don't know"

Finally, make it a tool: a loop, visible sources, and proof the guardrail works.

**Replace** the final `print(answer(...))` line with:

```python
print(f"NotesBot ready — {len(chunks)} chunks loaded. Ask me about the club!")
while True:
    question = input("\nYou: ").strip()
    if question.lower() in ("quit", "exit", ""):
        break
    sources = [c.splitlines()[0] for c in retrieve(question)]
    print("  [sources:", " | ".join(sources) + "]")
    print("Bot:", answer(question))
```

**Run** it and ask away:

- "When are the weekly meetups?" → Friday 4–6 PM, B204, with the sources line showing *which* chunks it used.
- "What happens if I return a laptop late?" → borrowing suspended for a month.
- Now the important one: **"Who won the hackathon last year?"** The notes don't say — and the bot says it doesn't know, because the system prompt forbids guessing. Comment out the `SYSTEM` message and ask again: watch it invent a winner. That difference — grounded vs. hallucinated — is *why* companies use RAG instead of trusting the model's memory.

**Try it:** edit `notes.md` (change the membership fee to Rs. 700), restart, and ask again. The bot's knowledge updated without retraining anything. That's the other superpower of RAG: knowledge you can edit with a text editor.

**Takeaway:** grounding ("answer only from the notes, else say I don't know") turns a creative text generator into a trustworthy tool — and updating its knowledge is just editing a file.

## The terminology, decoded

Every term in this guide exists because of one specific problem. Here's the chain — each row's problem is created by the row above it:

| Term | What it is | The problem it solves |
| --- | --- | --- |
| **RAG** | Retrieval-Augmented Generation: find the right text, paste it above the question, let the model answer from it | The model only knows its training data — your files are invisible to it (Step 1) |
| **Context window** | The maximum text a model can take in one prompt | It's why you can't just paste everything: a 50,000-page wiki doesn't fit, and you'd pay for all of it on every question (Step 2) |
| **Chunk** | One self-contained piece of a document (here: one `##` section) | You can't send *part* of a file unless the file has parts. Too big = paying for irrelevant text; too small = pieces that make no sense alone (Step 3) |
| **Embedding** | A vector (~3,000 numbers) a model produces from text, where similar meanings land at nearby points | Keyword search fails: "How much do I pay?" shares zero words with the Membership chunk. You need to match *meaning*, not spelling (Step 4) |
| **Vector** | A list of numbers, treated as a point (or arrow) in space | The form meaning has to take before you can do arithmetic on it |
| **Cosine similarity** | The angle between two vectors: 1.0 = same meaning, ~0 = unrelated | Given two embeddings, you need one number saying "how similar". It compares *direction, not length*, so a short question can match a long paragraph (Step 4) |
| **Similarity check** | Scoring the question's vector against every chunk's vector | Turns "which chunk is relevant?" into "which number is biggest?" — plain arithmetic, no AI judgment needed (Step 5) |
| **k (top-k)** | How many of the best-scoring chunks you keep and send | The cutoff between "enough context to answer" and "pasting the whole file again". Too small misses the answer; too big re-creates the scale problem (Step 5) |
| **Retrieval** | embed the question → rank all chunks by similarity → keep the top k | The R in RAG: finds the *right* text automatically, for any question (Step 5) |
| **Vector database** | Pinecone, Chroma, pgvector...: the ranking loop made fast, plus stored embeddings | Our `sorted()` scans every chunk on every question — fine for 8, hopeless for 8 million. Also avoids re-embedding on every restart (Step 5) |
| **Grounding** | "Answer ONLY from the notes; if they don't say, say you don't know" | Without it the model fills gaps with confident inventions (**hallucinations**). Grounding ties every answer to real retrieved text (Step 7) |
| **Hallucination** | A fluent, confident, made-up answer | Not solved but *exposed* by RAG: with sources shown and guessing forbidden, a wrong answer becomes a visible "the notes don't say" instead of a silent lie (Step 7) |

The one-breath version: the model doesn't know your data (**RAG**), you can't send all of it (**context window**), so you split it (**chunks**), turn meaning into numbers (**embeddings**), score them (**cosine similarity**), keep the best few (**top-k retrieval**), and force the answer to stick to them (**grounding**).

## Where to go from here

You built, from scratch: chunking, embeddings, cosine similarity, retrieval, grounded generation. Everything else in the RAG world is an upgrade to one of those boxes:

| You built | Production systems use |
| --- | --- |
| Split on `## ` | Token-based splitters with overlap, PDF/HTML parsers |
| Python list + `sorted()` | Vector databases (Chroma, Pinecone, pgvector) |
| Top-2 by cosine | Hybrid search (keywords + vectors), rerankers |
| One `notes.md` | Thousands of docs, re-embedded on change |
| "Say I don't know" | Citations, answer-checking, evaluation suites |

Ideas to try this week: point it at your own lecture notes; load several `.md` files into `chunks`; combine it with QuizBot so Quizzy quizzes you *from your notes* — retrieval becomes a tool the agent calls. That's how the two sessions connect: tools give an AI *abilities*, RAG gives it *knowledge*.
