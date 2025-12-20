import os
import json
import re
from typing import List, Dict, Any, Optional

import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_TRAINER = os.getenv("OLLAMA_MODEL_TRAINER", "qwen2.5:14b-instruct")
MODEL_EVAL = os.getenv("OLLAMA_MODEL_EVAL", MODEL_TRAINER)

client = ollama.Client(host=OLLAMA_HOST)

# -------------------------
# Prompts
# -------------------------
TRAINER_SYSTEM = """\
You are a friendly English tutor in a casual text chat.

Your goal is to keep a natural, interesting conversation AND gently help the student practice specific vocabulary.

Rules:
- First respond naturally to what the student said (like a real person).
- Then gently guide the student to practice.
- Use 1–2 target words naturally in your reply (not as a list).
- Ask the student to use 1–2 target words in the next message.
- Do NOT turn this into a lesson or explanation.
- Keep the tone friendly and conversational.
- Do not introduce many new words.

At the end of your reply, add a valid JSON block EXACTLY between:

<<<JSON>>>
{ ... }
<<<ENDJSON>>>

The JSON must contain:
- used_target_words: array of word ids you used
- requested_student_words: array of word ids the student should use next
- micro_task: short description of what the student should do next
- suggested_next_mode: one of ["casual", "roleplay", "story"]
"""

EVAL_SYSTEM = """\
You analyze the student's message.
You do NOT talk to the student.
Return ONLY valid JSON. No explanations.

Tasks:
1. Detect which target words the student used.
2. Mark correctness: correct, partial, or wrong.
3. List new vocabulary words the student used that are NOT target words.
4. Detect possible interests or topics (max 2).

Rules:
- Do not invent words.
- Ignore names and places.
- If unsure, use low confidence.
"""

JSON_BLOCK_RE = re.compile(r"<<<JSON>>>\s*(\{.*?\})\s*<<<ENDJSON>>>", re.DOTALL)


# -------------------------
# Helpers
# -------------------------
def ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    resp = client.chat(
        model=model,
        messages=messages,
        options={
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 512,
        },
    )
    return resp["message"]["content"]


def parse_trainer_output(text: str) -> tuple[str, dict]:
    """
    Splits assistant reply into (assistant_text, meta_json).
    If JSON is missing or broken, returns empty meta.
    """
    m = JSON_BLOCK_RE.search(text)
    if not m:
        return text.strip(), {}

    assistant_text = (text[: m.start()] + text[m.end():]).strip()
    raw = m.group(1).strip()

    try:
        meta = json.loads(raw)
        if not isinstance(meta, dict):
            return assistant_text, {}
        return assistant_text, meta
    except Exception:
        return assistant_text, {}


def parse_eval_json(text: str) -> dict:
    """
    Evaluator should return pure JSON.
    We also do best-effort extraction.
    """
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"(\{.*\})", t, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(1))
        except Exception:
            return {}


def pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


# -------------------------
# Test data (target words)
# -------------------------
TARGET_WORDS = [
    {"id": "w1", "lemma": "although", "gloss": "хотя"},
    {"id": "w2", "lemma": "manage", "gloss": "справляться"},
    {"id": "w3", "lemma": "issue", "gloss": "проблема"},
]


def build_trainer_user_payload(
    student_level: str,
    mode: str,
    target_words: List[dict],
    conversation_context: List[Dict[str, str]],
) -> str:
    payload = {
        "student_level": student_level,
        "mode": mode,
        "target_words": target_words,
        "conversation_context": conversation_context[-10:],
    }
    return pretty(payload)


def build_eval_user_payload(
    target_words: List[dict],
    expected_word_ids: List[str],
    student_message: str,
) -> str:
    payload = {
        "target_words": [{"id": w["id"], "lemma": w["lemma"]} for w in target_words],
        "expected_word_ids": expected_word_ids,
        "student_message": student_message,
    }
    return pretty(payload)


# -------------------------
# Modes
# -------------------------
def run_trainer_only():
    print("\n[Trainer-only mode]\nType 'exit' to quit.\n")
    chat: List[Dict[str, str]] = []
    mode = "casual"

    while True:
        student = input("You: ").strip()
        if not student:
            continue
        if student.lower() in {"exit", "quit"}:
            break

        # We store normal chat history for display/continuity
        chat.append({"role": "user", "content": student})

        # We provide structured context to trainer (as user message payload)
        trainer_user = build_trainer_user_payload(
            student_level="A2",
            mode=mode,
            target_words=TARGET_WORDS,
            conversation_context=chat,
        )

        out = ollama_chat(
            model=MODEL_TRAINER,
            messages=[
                {"role": "system", "content": TRAINER_SYSTEM},
                {"role": "user", "content": trainer_user},
            ],
        )

        assistant_text, meta = parse_trainer_output(out)
        print("\nTutor:", assistant_text)
        if meta:
            print("\n[Trainer JSON]:\n", pretty(meta))

        chat.append({"role": "assistant", "content": assistant_text})

        # optionally update mode
        if meta.get("suggested_next_mode") in {"casual", "roleplay", "story"}:
            mode = meta["suggested_next_mode"]


def run_evaluator_only():
    print("\n[Evaluator-only mode]\nType 'exit' to quit.\n")
    expected = ["w1", "w2"]

    while True:
        student = input("Student msg: ").strip()
        if not student:
            continue
        if student.lower() in {"exit", "quit"}:
            break

        eval_user = build_eval_user_payload(TARGET_WORDS, expected, student)
        out = ollama_chat(
            model=MODEL_EVAL,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM},
                {"role": "user", "content": eval_user},
            ],
            temperature=0.2,
        )
        data = parse_eval_json(out)
        print("\n[Eval JSON]:\n", pretty(data))


def run_full_loop():
    print("\n[Trainer + Evaluator loop]\nType 'exit' to quit.\n")
    chat: List[Dict[str, str]] = []
    mode = "roleplay"
    expected_word_ids: List[str] = []

    while True:
        student = input("You: ").strip()
        if not student:
            continue
        if student.lower() in {"exit", "quit"}:
            break

        chat.append({"role": "user", "content": student})

        # Evaluator (only if we have expected words or longer text)
        do_eval = (len(student) >= 25) and (bool(expected_word_ids) or any(w["lemma"] in student.lower() for w in TARGET_WORDS))
        if do_eval:
            eval_user = build_eval_user_payload(TARGET_WORDS, expected_word_ids, student)
            eval_out = ollama_chat(
                model=MODEL_EVAL,
                messages=[
                    {"role": "system", "content": EVAL_SYSTEM},
                    {"role": "user", "content": eval_user},
                ],
                temperature=0.2,
            )
            eval_data = parse_eval_json(eval_out)
            print("\n[Eval]:\n", pretty(eval_data))
        else:
            eval_data = {}

        # Trainer
        trainer_user = build_trainer_user_payload(
            student_level="A2",
            mode=mode,
            target_words=TARGET_WORDS,
            conversation_context=chat,
        )
        trainer_out = ollama_chat(
            model=MODEL_TRAINER,
            messages=[
                {"role": "system", "content": TRAINER_SYSTEM},
                {"role": "user", "content": trainer_user},
            ],
            temperature=0.7,
        )

        assistant_text, meta = parse_trainer_output(trainer_out)
        print("\nTutor:", assistant_text)
        if meta:
            print("\n[Trainer JSON]:\n", pretty(meta))

        chat.append({"role": "assistant", "content": assistant_text})

        # Next expected words (1–2)
        req = meta.get("requested_student_words", [])
        if isinstance(req, list):
            expected_word_ids = [x for x in req if isinstance(x, str)][:2]

        # Update mode if suggested
        if meta.get("suggested_next_mode") in {"casual", "roleplay", "story"}:
            mode = meta["suggested_next_mode"]


# -------------------------
# Entry
# -------------------------
def main():
    print("Select mode:")
    print("1) trainer-only")
    print("2) evaluator-only")
    print("3) full loop (trainer + evaluator)")
    choice = input("> ").strip()

    if choice == "1":
        run_trainer_only()
    elif choice == "2":
        run_evaluator_only()
    elif choice == "3":
        run_full_loop()
    else:
        print("Unknown choice.")


if __name__ == "__main__":
    main()
