from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# =========================================================
# Config
# =========================================================

MODEL_NAME = "qwen2.5:14b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"
MEMORY_FILE = Path("user_memory.json")


# =========================================================
# Persistent user memory
# =========================================================

def _default_user_memory(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "english_level": "A2-B1",
        "interests": ["movies", "VR games", "machine learning", "music"],
        "target_words": [
            {
                "word": "apparently",
                "meaning_ru": "по-видимому, как оказалось",
                "examples": 0,
                "correct_uses": 0,
                "incorrect_uses": 0,
                "mastery": 0.0,
            },
            {
                "word": "tough",
                "meaning_ru": "трудный, жёсткий, крепкий",
                "examples": 0,
                "correct_uses": 0,
                "incorrect_uses": 0,
                "mastery": 0.0,
            },
            {
                "word": "work out",
                "meaning_ru": "тренироваться; получиться, сработать",
                "examples": 0,
                "correct_uses": 0,
                "incorrect_uses": 0,
                "mastery": 0.0,
            },
        ],
        "common_mistakes": [],
        "new_words_seen": [],
        "last_topics": [],
    }


def load_user_memory(user_id: str) -> dict[str, Any]:
    if not MEMORY_FILE.exists():
        return _default_user_memory(user_id)

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = {}

    return data.get(user_id, _default_user_memory(user_id))


def save_user_memory(user_id: str, memory: dict[str, Any]) -> None:
    if MEMORY_FILE.exists():
        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}

    data[user_id] = memory
    MEMORY_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================================================
# Structured outputs
# =========================================================

ALLOWED_ISSUE_TYPES = {
    "none",
    "grammar",
    "semantic",
    "collocation",
    "register",
    "naturalness",
}


class GrammarError(BaseModel):
    fragment: str = ""
    issue: str = ""
    fix: str = ""


class TargetWordCheck(BaseModel):
    word: str = ""
    used: bool = False
    correct: bool = False
    issue_type: Literal["none", "grammar", "semantic", "collocation", "register", "naturalness"] = "none"
    explanation: str = ""
    better_example: str = ""

    @field_validator("issue_type", mode="before")
    @classmethod
    def normalize_issue_type(cls, v):
        if v is None:
            return "none"
        if isinstance(v, str):
            value = v.strip().lower()
            if not value:
                return "none"
            if value in ALLOWED_ISSUE_TYPES:
                return value
        return "none"


class AnalysisResult(BaseModel):
    student_intent: str = ""
    overall_quality: Literal["good", "mixed", "poor"] = "mixed"
    has_grammar_errors: bool = False
    grammar_errors: list[GrammarError] = Field(default_factory=list)
    corrected_sentence: str = ""
    target_word_checks: list[TargetWordCheck] = Field(default_factory=list)
    new_unknown_words: list[str] = Field(default_factory=list)
    should_correct_explicitly: bool = True
    encouragement_mode: Literal["light", "normal", "supportive"] = "normal"

    @field_validator("overall_quality", mode="before")
    @classmethod
    def normalize_quality(cls, v):
        allowed = {"good", "mixed", "poor"}
        if isinstance(v, str) and v.strip().lower() in allowed:
            return v.strip().lower()
        return "mixed"

    @field_validator("encouragement_mode", mode="before")
    @classmethod
    def normalize_encouragement_mode(cls, v):
        allowed = {"light", "normal", "supportive"}
        if isinstance(v, str) and v.strip().lower() in allowed:
            return v.strip().lower()
        return "normal"


class TutorReply(BaseModel):
    short_feedback: str = ""
    reply_text: str = ""


# =========================================================
# LangGraph state
# =========================================================

class TutorState(TypedDict, total=False):
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    user_memory: dict[str, Any]
    analysis: dict[str, Any]
    reply: str


# =========================================================
# Local LLM / connectivity
# =========================================================

def check_ollama(base_url: str = OLLAMA_BASE_URL) -> None:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Ollama is not reachable at {base_url}. "
            f"Start Ollama first. Original error: {e}"
        )


def make_llm() -> ChatOllama:
    check_ollama(OLLAMA_BASE_URL)
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )


llm = make_llm()


# =========================================================
# Helpers
# =========================================================

def normalize_word(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def format_target_words(target_words: list[dict[str, Any]]) -> str:
    lines = []
    for item in target_words:
        lines.append(
            f"- {item['word']} — {item.get('meaning_ru', '')}; "
            f"mastery={item.get('mastery', 0):.2f}; "
            f"correct={item.get('correct_uses', 0)}; "
            f"incorrect={item.get('incorrect_uses', 0)}"
        )
    return "\n".join(lines)


def pick_active_target_words(user_memory: dict[str, Any], n: int = 3) -> list[dict[str, Any]]:
    words = sorted(
        user_memory.get("target_words", []),
        key=lambda x: (x.get("mastery", 0.0), x.get("examples", 0)),
    )
    return words[:n]


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from model output:\n{text}")


def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0", ""}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def clean_analysis_payload(raw: dict[str, Any], active_words: list[dict[str, Any]]) -> dict[str, Any]:
    data = dict(raw) if isinstance(raw, dict) else {}

    if not isinstance(data.get("student_intent"), str):
        data["student_intent"] = ""

    if not isinstance(data.get("corrected_sentence"), str):
        data["corrected_sentence"] = ""

    data["has_grammar_errors"] = coerce_bool(data.get("has_grammar_errors"), False)
    data["should_correct_explicitly"] = coerce_bool(data.get("should_correct_explicitly"), True)

    if not isinstance(data.get("new_unknown_words"), list):
        data["new_unknown_words"] = []
    else:
        data["new_unknown_words"] = [str(x).strip() for x in data["new_unknown_words"] if str(x).strip()]

    if not isinstance(data.get("grammar_errors"), list):
        data["grammar_errors"] = []
    else:
        cleaned_errors = []
        for item in data["grammar_errors"]:
            if not isinstance(item, dict):
                continue
            cleaned_errors.append(
                {
                    "fragment": str(item.get("fragment", "")).strip(),
                    "issue": str(item.get("issue", "")).strip(),
                    "fix": str(item.get("fix", "")).strip(),
                }
            )
        data["grammar_errors"] = cleaned_errors

    if not isinstance(data.get("target_word_checks"), list):
        data["target_word_checks"] = []

    cleaned_checks = []
    for item in data["target_word_checks"]:
        if not isinstance(item, dict):
            continue
        issue_type = str(item.get("issue_type", "")).strip().lower()
        if issue_type not in ALLOWED_ISSUE_TYPES:
            issue_type = "none"

        cleaned_checks.append(
            {
                "word": str(item.get("word", "")).strip(),
                "used": coerce_bool(item.get("used"), False),
                "correct": coerce_bool(item.get("correct"), False),
                "issue_type": issue_type,
                "explanation": str(item.get("explanation", "")).strip(),
                "better_example": str(item.get("better_example", "")).strip(),
            }
        )

    # гарантируем наличие записей по активным target words
    existing_words = {normalize_word(x["word"]) for x in cleaned_checks if x.get("word")}
    for tw in active_words:
        word = tw["word"]
        if normalize_word(word) not in existing_words:
            cleaned_checks.append(
                {
                    "word": word,
                    "used": False,
                    "correct": False,
                    "issue_type": "none",
                    "explanation": "The word was not used in the learner message.",
                    "better_example": "",
                }
            )

    data["target_word_checks"] = cleaned_checks[: max(3, len(active_words))]
    return data


def clean_tutor_reply_payload(raw: dict[str, Any]) -> dict[str, Any]:
    data = dict(raw) if isinstance(raw, dict) else {}
    short_feedback = str(data.get("short_feedback", "")).strip()
    reply_text = str(data.get("reply_text", "")).strip()

    if not short_feedback:
        short_feedback = "Good attempt."
    if not reply_text:
        reply_text = "Let’s keep practicing. Can you say that again in a slightly different way?"

    return {
        "short_feedback": short_feedback,
        "reply_text": reply_text,
    }


def update_word_stats(user_memory: dict[str, Any], analysis: AnalysisResult, tutor_reply_text: str) -> None:
    target_words = user_memory.get("target_words", [])
    target_word_map = {normalize_word(w["word"]): w for w in target_words}

    for check in analysis.target_word_checks:
        key = normalize_word(check.word)
        if key not in target_word_map:
            continue

        item = target_word_map[key]
        if check.used:
            if check.correct:
                item["correct_uses"] = item.get("correct_uses", 0) + 1
            else:
                item["incorrect_uses"] = item.get("incorrect_uses", 0) + 1

        total = item.get("correct_uses", 0) + item.get("incorrect_uses", 0)
        if total > 0:
            item["mastery"] = round(item.get("correct_uses", 0) / total, 2)

    reply_lower = normalize_word(tutor_reply_text)
    for item in target_words:
        if normalize_word(item["word"]) in reply_lower:
            item["examples"] = item.get("examples", 0) + 1


def update_common_mistakes(user_memory: dict[str, Any], analysis: AnalysisResult) -> None:
    mistakes = user_memory.setdefault("common_mistakes", [])
    for err in analysis.grammar_errors:
        mistakes.append(
            {
                "fragment": err.fragment,
                "issue": err.issue,
                "fix": err.fix,
            }
        )
    user_memory["common_mistakes"] = mistakes[-20:]


def update_new_words(user_memory: dict[str, Any], analysis: AnalysisResult) -> None:
    existing = set(user_memory.get("new_words_seen", []))
    for word in analysis.new_unknown_words:
        if word not in existing:
            user_memory.setdefault("new_words_seen", []).append(word)


def update_topics(user_memory: dict[str, Any], analysis: AnalysisResult) -> None:
    topics = user_memory.setdefault("last_topics", [])
    if analysis.student_intent.strip():
        topics.append(analysis.student_intent.strip())
    user_memory["last_topics"] = topics[-10:]


def get_last_user_message(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    raise ValueError("No user message found.")


# =========================================================
# LLM calls
# =========================================================

def run_json_prompt(schema_hint: dict[str, Any], system_prompt: str, user_prompt: str, retries: int = 2) -> dict[str, Any]:
    last_error = None

    for attempt in range(retries + 1):
        full_user_prompt = f"""
Return ONLY valid JSON.
NO explanations.
NO markdown.
NO text before JSON.
NO text after JSON.

Follow this JSON schema shape exactly:
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}

Task:
{user_prompt}
""".strip()

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=full_user_prompt),
                ]
            )
            return extract_json_object(str(response.content))
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Model failed to return valid JSON after retries. Last error: {last_error}")


# =========================================================
# Graph nodes
# =========================================================

def load_profile(state: TutorState) -> TutorState:
    user_memory = load_user_memory(state["user_id"])
    return {"user_memory": user_memory}


def analyze_message(state: TutorState) -> TutorState:
    user_memory = state["user_memory"]
    active_words = pick_active_target_words(user_memory, n=3)
    target_words_text = format_target_words(active_words)
    last_user_message = get_last_user_message(state["messages"])

    schema_hint = {
        "student_intent": "short phrase",
        "overall_quality": "good",
        "has_grammar_errors": True,
        "grammar_errors": [
            {
                "fragment": "I goed",
                "issue": "wrong past tense",
                "fix": "I went"
            }
        ],
        "corrected_sentence": "Yesterday I worked out at the gym and it was very tough, but apparently I did well.",
        "target_word_checks": [
            {
                "word": "apparently",
                "used": True,
                "correct": True,
                "issue_type": "none",
                "explanation": "The word is used correctly.",
                "better_example": "Apparently, he forgot about the meeting."
            },
            {
                "word": "tough",
                "used": True,
                "correct": True,
                "issue_type": "none",
                "explanation": "The word is used naturally.",
                "better_example": "It was a tough workout."
            },
            {
                "word": "work out",
                "used": True,
                "correct": True,
                "issue_type": "none",
                "explanation": "The phrase is used correctly.",
                "better_example": "I worked out yesterday."
            }
        ],
        "new_unknown_words": ["deadline"],
        "should_correct_explicitly": True,
        "encouragement_mode": "normal"
    }

    user_prompt = f"""
Analyze the learner's message.

Learner profile:
- level: {user_memory.get("english_level", "unknown")}
- interests: {", ".join(user_memory.get("interests", []))}
- recent mistakes: {json.dumps(user_memory.get("common_mistakes", [])[-5:], ensure_ascii=False)}

Current target words:
{target_words_text}

Requirements:
1. Check grammar.
2. Check naturalness.
3. Check whether target words are used correctly in context.
4. corrected_sentence must be a natural corrected version of the whole learner sentence.
5. Only put truly useful items into new_unknown_words.
6. Keep student_intent short.
7. issue_type must always be one of:
   "none", "grammar", "semantic", "collocation", "register", "naturalness".
8. If there is no issue, use "none".
9. Never return an empty string for issue_type.
10. Return checks for all target words, even if they were not used.

Learner message:
{last_user_message}
""".strip()

    raw = run_json_prompt(
        schema_hint=schema_hint,
        system_prompt="You are a precise English tutor analyzer.",
        user_prompt=user_prompt,
        retries=2,
    )

    cleaned = clean_analysis_payload(raw, active_words)

    try:
        analysis = AnalysisResult.model_validate(cleaned)
    except ValidationError as e:
        raise RuntimeError(f"Analysis validation failed after cleanup: {e}\nRaw cleaned data: {json.dumps(cleaned, ensure_ascii=False, indent=2)}")

    return {"analysis": analysis.model_dump()}


def generate_reply(state: TutorState) -> TutorState:
    user_memory = state["user_memory"]
    analysis = AnalysisResult.model_validate(state["analysis"])
    active_words = pick_active_target_words(user_memory, n=3)
    target_words_text = format_target_words(active_words)
    last_user_message = get_last_user_message(state["messages"])

    schema_hint = {
        "short_feedback": "A better version would be: 'Yesterday I worked out at the gym.'",
        "reply_text": "That sounds like a tough workout. Apparently, you still managed to do well. What exercises did you do?"
    }

    user_prompt = f"""
Write a natural tutor reply.

Learner profile:
- level: {user_memory.get("english_level", "unknown")}
- interests: {", ".join(user_memory.get("interests", []))}

Current target words:
{target_words_text}

Learner message:
{last_user_message}

Analysis:
{analysis.model_dump_json(indent=2)}

Rules:
1. Keep the tone natural and friendly.
2. Keep correction short.
3. If there is a mistake, show a better version briefly.
4. If a target word was used incorrectly, explain briefly and naturally.
5. Use 1 or 2 target words naturally in your own reply when suitable.
6. End with one short follow-up question.
7. reply_text must be in English.
8. short_feedback must be in English.
9. Return JSON only.
""".strip()

    raw = run_json_prompt(
        schema_hint=schema_hint,
        system_prompt="You are a natural conversational English tutor.",
        user_prompt=user_prompt,
        retries=2,
    )

    cleaned = clean_tutor_reply_payload(raw)
    tutor_reply = TutorReply.model_validate(cleaned)
    final_text = f"{tutor_reply.short_feedback}\n\n{tutor_reply.reply_text}".strip()

    return {
        "reply": final_text,
        "messages": [AIMessage(content=final_text)],
    }


def update_memory_node(state: TutorState) -> TutorState:
    user_id = state["user_id"]
    user_memory = state["user_memory"]
    analysis = AnalysisResult.model_validate(state["analysis"])
    reply_text = state["reply"]

    update_word_stats(user_memory, analysis, reply_text)
    update_common_mistakes(user_memory, analysis)
    update_new_words(user_memory, analysis)
    update_topics(user_memory, analysis)

    save_user_memory(user_id, user_memory)
    return {"user_memory": user_memory}


# =========================================================
# Build graph
# =========================================================

def build_graph():
    builder = StateGraph(TutorState)

    builder.add_node("load_profile", load_profile)
    builder.add_node("analyze_message", analyze_message)
    builder.add_node("generate_reply", generate_reply)
    builder.add_node("update_memory", update_memory_node)

    builder.add_edge(START, "load_profile")
    builder.add_edge("load_profile", "analyze_message")
    builder.add_edge("analyze_message", "generate_reply")
    builder.add_edge("generate_reply", "update_memory")
    builder.add_edge("update_memory", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


# =========================================================
# CLI
# =========================================================

def print_user_progress(user_id: str) -> None:
    memory = load_user_memory(user_id)
    print("\n=== USER MEMORY SNAPSHOT ===")
    print(f"Level: {memory.get('english_level')}")
    print(f"Interests: {', '.join(memory.get('interests', []))}")
    print("Target words:")
    for w in memory.get("target_words", []):
        print(
            f"  - {w['word']}: mastery={w.get('mastery', 0):.2f}, "
            f"correct={w.get('correct_uses', 0)}, "
            f"incorrect={w.get('incorrect_uses', 0)}, "
            f"examples={w.get('examples', 0)}"
        )
    print(f"Recent topics: {memory.get('last_topics', [])[-5:]}")
    print("=== END SNAPSHOT ===\n")


def chat():
    user_id = "alex"
    thread_id = "alex-session-1"
    config = {"configurable": {"thread_id": thread_id}}

    print("Local English Tutor Bot")
    print(f"Model: {MODEL_NAME}")
    print("Commands: exit, memory\n")

    while True:
        user_text = input("You: ").strip()

        if not user_text:
            continue

        if user_text.lower() == "exit":
            print("Bye.")
            break

        if user_text.lower() == "memory":
            print_user_progress(user_id)
            continue

        try:
            result = graph.invoke(
                {
                    "user_id": user_id,
                    "messages": [HumanMessage(content=user_text)],
                },
                config=config,
            )
            print(f"\nTutor: {result['reply']}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    chat()