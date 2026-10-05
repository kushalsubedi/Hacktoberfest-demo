import html
import json
import os
import random

import requests
from dotenv import load_dotenv
from openai import OpenAI

from quizbot.knowledge import recall, remember

load_dotenv(override=True)
MODEL = "gemini-3.8-flash"

CATEGORIES = {"general": 9, "film": 11, "music": 12, "video games": 15, "science": 17,
              "computers": 18, "sports": 21, "geography": 22, "history": 23, "animals": 27}


def get_trivia(category: str = "general", difficulty: str = "easy") -> dict:
    """Fetch one real multiple-choice trivia question."""
    params = {"amount": 1, "type": "multiple", "difficulty": difficulty,
              "category": CATEGORIES.get(category, 9)}
    data = requests.get("https://opentdb.com/api.php", params=params, timeout=15).json()
    if data.get("response_code") != 0:
        return {"error": "No question available right now. The trivia site allows "
                         "one request every 5 seconds, so try again shortly."}
    q = data["results"][0]
    options = [html.unescape(a) for a in q["incorrect_answers"] + [q["correct_answer"]]]
    random.shuffle(options)
    return {"question": html.unescape(q["question"]), "options": options,
            "correct_answer": html.unescape(q["correct_answer"]), "difficulty": difficulty}


SCORES = {}


def add_points(player: str, points: int) -> dict:
    """Give a player points and return the scoreboard."""
    if not 1 <= points <= 3:
        return {"error": "points must be between 1 and 3"}
    SCORES[player] = SCORES.get(player, 0) + points
    return SCORES


TOOLS = {"get_trivia": get_trivia, "add_points": add_points,
         "remember": remember, "recall": recall}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_trivia",
            "description": "Fetch one real multiple-choice trivia question. Use it for every question you ask.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_points",
            "description": "Award points after a correct answer: easy 1, medium 2, hard 3.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player": {"type": "string"},
                    "points": {"type": "integer"},
                },
                "required": ["player", "points"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a note to the long-term knowledge base (player names, scores, "
                           "preferences, running jokes). It survives restarts.",
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
            "description": "Read everything in the knowledge base. Call it at the start of a "
                           "conversation so you recognise returning players.",
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


def main() -> None:
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    messages = [{"role": "system", "content":
                 "You are Quizzy, an over-the-top game show host running a trivia game. "
                 "Get every question from get_trivia; never invent questions. "
                 "Show the options labelled A to D, and never reveal the answer before a player guesses. "
                 "When a player answers correctly, call add_points. Keep it short and fun. "
                 "At the start of a conversation call recall to recognise returning players, "
                 "and use remember to save player names, final scores and preferences. "
                 "If a tool returns an error, say so honestly."}]

    while True:
        try:
            user = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("quit", "exit"):
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
        print("Quizzy:", msg.content)


if __name__ == "__main__":
    main()
