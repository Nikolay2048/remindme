from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import os
import ollama

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = ollama.Client(host=OLLAMA_HOST)

RE_RU = re.compile(r"[А-Яа-яЁё]+")
RE_WORD = re.compile(r"[A-Za-z']+")


def detect_has_ru(text: str) -> bool:
    return bool(RE_RU.search(text or ""))


def normalize_token(t: str) -> str:
    t = (t or "").strip().lower()
    # strip punctuation on edges
    return t.strip(" \t\n\r.,!?;:\"()[]{}<>")

def extract_en_tokens(text: str) -> List[str]:
    return [normalize_token(x) for x in RE_WORD.findall(text or "") if normalize_token(x)]


@dataclass
class FocusWord:
    text: str
    cefr: str = "UNKNOWN"
    mastery: int = 0
    status: str = "new"


@dataclass
class TurnAnalysis:
    has_ru: bool
    unknown_ru_fragments: List[str]
    used_focus_words: List[str]
    en_tokens: List[str]


@dataclass
class ChatTurn:
    ts: float
    role: str  # "student" | "bot"
    text: str
    analysis: Optional[Dict[str, Any]] = None


class SimpleVocabStore:
    """
    MVP-заглушка.
    Потом заменишь на реальную выборку из user_vocabulary.
    """
    def __init__(self):
        self.focus_words: List[FocusWord] = [
            FocusWord("recommend", "B1", 25, "learning"),
            FocusWord("instead", "A2", 40, "learning"),
            FocusWord("however", "B1", 20, "learning"),
            FocusWord("actually", "A2", 55, "learning"),
            FocusWord("improve", "A2", 30, "learning"),
        ]

    def select_focus_words(self, k: int = 5) -> List[FocusWord]:
        return self.focus_words[:k]

    def update_from_analysis(self, analysis: TurnAnalysis) -> Dict[str, Any]:
        # MVP: возвращаем апдейты, чтобы потом писать в БД
        updates = {"used": analysis.used_focus_words, "unknown_ru": analysis.unknown_ru_fragments}
        return updates


class TutorEngine:
    def __init__(self, model: str = "qwen2.5:14b-instruct", client=None):
        self.model = model
        self.client = client or ollama_client

    def build_system_prompt(self, topic: str, focus_words: List[FocusWord]) -> str:
        fw = ", ".join([f.text for f in focus_words])
        return (
            "You are a friendly English tutor in a casual text chat.\n"
            "Goals:\n"
            "1) Keep the conversation natural and engaging.\n"
            "2) Provide at most 2 corrections per student message (only the most important ones).\n"
            "3) Encourage the student to use focus words naturally.\n\n"
            "Rules:\n"
            "- First answer the meaning. Then (optional) corrections. Then ask ONE short question.\n"
            "- If the student uses Russian words, ask what they meant and offer 2 English options.\n"
            "- Keep replies short (3-8 lines).\n\n"
            f"Topic: {topic}\n"
            f"Focus words to practice: {fw}\n"
        )

    def reply(
        self,
        *,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        repeat_penalty: float = 1.15,
        max_tokens: int = 300,
    ) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "num_predict": max_tokens,
            },
        )
        return (resp["message"]["content"] or "").strip()


class Analyzer:
    def analyze_student_text(self, text: str, focus_words: List[FocusWord]) -> TurnAnalysis:
        text = text or ""
        has_ru = detect_has_ru(text)
        # MVP: просто выдернем русские фрагменты
        unknown_ru = list(set(RE_RU.findall(text))) if has_ru else []

        tokens = extract_en_tokens(text)
        focus_set = {f.text.lower() for f in focus_words}
        used_focus = sorted({t for t in tokens if t in focus_set})

        return TurnAnalysis(
            has_ru=has_ru,
            unknown_ru_fragments=unknown_ru[:10],
            used_focus_words=used_focus,
            en_tokens=tokens[:200],
        )


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    log_path = "chat_log.jsonl"

    topic = "daily life + hobbies"
    vocab = SimpleVocabStore()
    focus_words = vocab.select_focus_words(k=5)

    tutor = TutorEngine(model="qwen2.5:14b-instruct")
    analyzer = Analyzer()

    system_prompt = tutor.build_system_prompt(topic, focus_words)

    print("Console English Tutor (Qwen 14B). Type /exit to quit.\n")
    print("Focus words:", ", ".join([w.text for w in focus_words]))
    print("Topic:", topic)
    print("-" * 60)

    chat_messages: List[Dict[str, str]] = []
    turn_idx = 0

    while True:
        student = input("\nYou: ").strip()
        if not student:
            continue
        if student.lower() in ("/exit", "exit", "quit"):
            print("Bye!")
            break

        # analyze
        analysis = analyzer.analyze_student_text(student, focus_words)
        vocab_updates = vocab.update_from_analysis(analysis)

        # log student
        st_turn = ChatTurn(ts=time.time(), role="student", text=student, analysis=asdict(analysis))
        append_jsonl(log_path, asdict(st_turn))

        # add to conversation context
        chat_messages.append({"role": "user", "content": student})

        # occasionally nudge with focus words request (every 2 turns)
        turn_idx += 1
        if turn_idx % 2 == 0:
            nudge = (
                "In your next message, please try to naturally use: "
                + ", ".join([w.text for w in focus_words[:2]])
                + "."
            )
            chat_messages.append({"role": "user", "content": nudge})

        # generate reply
        bot_text = tutor.reply(system_prompt=system_prompt, messages=chat_messages)

        # log bot
        bt_turn = ChatTurn(ts=time.time(), role="bot", text=bot_text, analysis={"vocab_updates": vocab_updates})
        append_jsonl(log_path, asdict(bt_turn))

        # print
        print("\nTutor:", bot_text)

        # keep context short (MVP)
        if len(chat_messages) > 12:
            chat_messages = chat_messages[-12:]


if __name__ == "__main__":
    print("Using Ollama host:", OLLAMA_HOST)
    print("Models:", ollama_client.list())
    main()
