from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# ============================================================
# Structured output schemas
# ============================================================

class NewVocabularyCandidate(BaseModel):
    text: str = Field(..., description="Surface form in lower case")
    lemma: str | None = None
    translation_text: str | None = None
    item_type: Literal["WORD", "PHRASE"] = "WORD"
    is_valid_in_context: bool
    reason: str | None = None
    corrected_usage: str | None = None


class DetectedMistake(BaseModel):
    original: str
    corrected: str
    explanation: str


class ConversationLLMOutput(BaseModel):
    reply_text: str
    used_target_words: list[str] = Field(default_factory=list)
    new_user_vocabulary: list[NewVocabularyCandidate] = Field(default_factory=list)
    mistakes: list[DetectedMistake] = Field(default_factory=list)
    detected_topics: list[str] = Field(default_factory=list)


# ============================================================
# In-memory models
# ============================================================

@dataclass
class ChatMessage:
    role: str
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ChatSession:
    session_id: str
    user_id: int
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True


@dataclass
class UserVocabularyItem:
    text: str
    lemma: str | None = None
    translation_text: str | None = None
    item_type: str = "WORD"
    contexts: list[str] = field(default_factory=list)
    source: list[str] = field(default_factory=lambda: ["chat"])
    mastery: int = 0
    seen_count: int = 0
    correct_count: int = 0
    success_streak: int = 0
    status: str = "new"
    active_knowledge: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_seen_at: datetime | None = None


@dataclass
class UserProfile:
    interests: dict[str, float] = field(default_factory=dict)
    style_preferences: dict[str, float] = field(default_factory=dict)
    current_topic: str | None = None
    topic_history: list[str] = field(default_factory=list)


# ============================================================
# Text utils
# ============================================================

EN_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]*")

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "am", "i", "you",
    "he", "she", "it", "we", "they", "to", "of", "in", "on", "at", "for",
    "with", "this", "that", "yes", "no", "ok", "okay", "hi", "hello",
    "very", "good", "bad", "nice", "thing", "stuff", "was", "were", "be",
    "have", "has", "had", "do", "did", "does", "my", "your", "his", "her",
    "their", "our", "me", "him", "them", "us", "as", "if", "so", "just",
    "really", "maybe", "also", "then", "than", "from", "into", "about",
    "after", "before", "because", "today", "yesterday", "tomorrow"
}


def extract_english_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in EN_TOKEN_RE.finditer(text)]


def candidate_new_words(text: str) -> list[str]:
    tokens = extract_english_tokens(text)
    result: list[str] = []
    for token in tokens:
        if len(token) <= 2:
            continue
        if token in STOP_WORDS:
            continue
        result.append(token)
    return sorted(set(result))


# ============================================================
# In-memory repositories / services
# ============================================================

class InMemoryChatMemoryService:
    def __init__(self):
        self.sessions_by_id: dict[str, ChatSession] = {}
        self.active_session_by_user: dict[int, str] = {}

    def get_or_create_active_session(self, user_id: int) -> str:
        session_id = self.active_session_by_user.get(user_id)
        if session_id and session_id in self.sessions_by_id:
            return session_id

        new_session_id = str(uuid.uuid4())
        session = ChatSession(session_id=new_session_id, user_id=user_id)
        self.sessions_by_id[new_session_id] = session
        self.active_session_by_user[user_id] = new_session_id
        return new_session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        session = self.sessions_by_id[session_id]
        session.messages.append(ChatMessage(role=role, content=content))
        session.updated_at = datetime.utcnow()

    def get_recent_messages(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        session = self.sessions_by_id[session_id]
        recent = session.messages[-limit:]
        return [{"role": m.role, "content": m.content} for m in recent]


class InMemoryVocabularyRepository:
    def __init__(self):
        self.data: dict[int, dict[str, UserVocabularyItem]] = {}

    def _user_bucket(self, user_id: int) -> dict[str, UserVocabularyItem]:
        if user_id not in self.data:
            self.data[user_id] = {}
        return self.data[user_id]

    def exists(self, user_id: int, text_value: str) -> bool:
        return text_value.lower() in self._user_bucket(user_id)

    def upsert_new_word(
        self,
        *,
        user_id: int,
        text_value: str,
        lemma: str | None,
        translation_text: str | None,
        context: str,
        source: str = "chat",
    ) -> None:
        text_value = text_value.lower().strip()
        bucket = self._user_bucket(user_id)
        now = datetime.utcnow()

        if text_value not in bucket:
            bucket[text_value] = UserVocabularyItem(
                text=text_value,
                lemma=lemma.lower() if lemma else None,
                translation_text=translation_text,
                contexts=[context],
                source=[source],
                mastery=5,
                seen_count=1,
                correct_count=1,
                success_streak=1,
                status="learning",
                active_knowledge=False,
                created_at=now,
                updated_at=now,
                last_seen_at=now,
            )
            return

        item = bucket[text_value]
        if lemma and not item.lemma:
            item.lemma = lemma.lower()
        if translation_text and not item.translation_text:
            item.translation_text = translation_text
        if context not in item.contexts:
            item.contexts.append(context)
        if source not in item.source:
            item.source.append(source)

        item.seen_count += 1
        item.correct_count += 1
        item.success_streak += 1
        item.mastery = min(100, item.mastery + 5)
        item.status = "known" if item.mastery >= 70 else "learning"
        item.active_knowledge = item.correct_count >= 3
        item.last_seen_at = now
        item.updated_at = now

    def list_user_vocabulary(self, user_id: int) -> list[UserVocabularyItem]:
        return list(self._user_bucket(user_id).values())


class InMemoryUserProfileRepository:
    def __init__(self):
        self.data: dict[int, UserProfile] = {}

    def get_or_create(self, user_id: int) -> UserProfile:
        if user_id not in self.data:
            self.data[user_id] = UserProfile()
        return self.data[user_id]


class InterestProfileService:
    TOPIC_KEYWORDS: dict[str, set[str]] = {
        "vr": {"vr", "quest", "virtual", "headset", "alyx", "steamvr", "meta"},
        "games": {"game", "games", "gaming", "boss", "walkthrough", "level", "puzzle", "rpg"},
        "movies": {"movie", "film", "cinema", "series", "episode", "director", "scene"},
        "machine_learning": {"ml", "machine", "learning", "neural", "dataset", "model", "training"},
        "psychology": {"emotion", "motivation", "feeling", "anxiety", "psychology", "mind", "mood"},
        "music": {"music", "song", "songs", "piano", "synth", "synthesizer", "melody"},
        "work": {"job", "work", "office", "task", "meeting", "project", "deadline"},
        "travel": {"travel", "trip", "vacation", "flight", "hotel", "country"},
        "fitness": {"gym", "run", "running", "workout", "training", "exercise", "treadmill"},
        "english_learning": {"english", "word", "vocabulary", "grammar", "lesson", "tutor"},
    }

    STYLE_HINTS: dict[str, set[str]] = {
        "prefers_deep_topics": {"meaning", "symbolism", "emotion", "feel", "why", "deep"},
        "likes_storytelling": {"story", "character", "scene", "plot", "ending"},
        "likes_practical_examples": {"example", "real life", "practice", "work", "daily"},
    }

    def __init__(self, repo: InMemoryUserProfileRepository):
        self.repo = repo

    def update_from_text(self, user_id: int, text: str) -> None:
        profile = self.repo.get_or_create(user_id)
        lowered = text.lower()

        for topic, keywords in self.TOPIC_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in lowered)
            if hits > 0:
                profile.interests[topic] = profile.interests.get(topic, 0.0) + hits * 0.25

        for style_key, keywords in self.STYLE_HINTS.items():
            hits = sum(1 for kw in keywords if kw in lowered)
            if hits > 0:
                profile.style_preferences[style_key] = profile.style_preferences.get(style_key, 0.0) + hits * 0.15

    def apply_detected_topics(self, user_id: int, topics: list[str]) -> None:
        if not topics:
            return
        profile = self.repo.get_or_create(user_id)
        for topic in topics:
            t = topic.strip().lower()
            if not t:
                continue
            profile.interests[t] = profile.interests.get(t, 0.0) + 0.2
            profile.current_topic = t
            if t not in profile.topic_history:
                profile.topic_history.append(t)
            else:
                # move topic to the end as more recent
                profile.topic_history = [x for x in profile.topic_history if x != t] + [t]

    def top_interests(self, user_id: int, limit: int = 3) -> list[str]:
        profile = self.repo.get_or_create(user_id)
        ranked = sorted(profile.interests.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in ranked[:limit]]

    def top_style_preferences(self, user_id: int, limit: int = 3) -> list[str]:
        profile = self.repo.get_or_create(user_id)
        ranked = sorted(profile.style_preferences.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in ranked[:limit]]

    def current_topic(self, user_id: int) -> str | None:
        return self.repo.get_or_create(user_id).current_topic


# ============================================================
# Mock word selector
# ============================================================

class MockWordSelector:
    def select_for_reply(self, user_id: int, max_words: int = 2) -> list[str]:
        pool = ["effort", "avoid", "exhausted", "improve", "confident"]
        return pool[:max_words]


# ============================================================
# LangChain runner
# ============================================================

class LangChainConversationRunner:
    def __init__(
        self,
        model: str = "qwen2.5:14b-instruct",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.8,
    ):
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )
        self.structured_llm = self.llm.with_structured_output(ConversationLLMOutput)

    def run(
        self,
        *,
        recent_messages: list[dict[str, str]],
        user_text: str,
        target_words: list[str],
        candidate_tokens: list[str],
        top_interests: list[str],
        style_preferences: list[str],
        current_topic: str | None,
    ) -> ConversationLLMOutput:
        system_prompt = """
You are an English tutor chatbot and conversation partner.

Main goals:
1. Keep the dialogue natural, interesting, and engaging.
2. Continue the conversation instead of giving dry textbook replies.
3. If appropriate, use 1-2 words from target_words naturally in your reply.
4. Connect the conversation to the user's interests when it feels natural.
5. Ask follow-up questions that make the user want to continue.
6. Use examples or references related to the user's interests when helpful.
7. If the user makes a mistake, correct gently without breaking the conversation flow.
8. Detect whether the user used any candidate English words correctly in context.
9. Only mark a word as valid if it is truly used with the intended meaning in context.
10. Be conservative: if unsure, mark is_valid_in_context = false.

Important style rules:
- Do not sound like a generic tutor.
- Do not turn every reply into a lesson.
- Keep the reply natural, warm, and a bit personalized.
- Prefer one interesting follow-up question over many small questions.
- If the user's interests include games, VR, movies, ML, psychology, music, etc., use them naturally as conversational context.
- If the user sounds emotional or reflective, respond with emotional intelligence.

Rules for new_user_vocabulary:
- Include only meaningful English words or short phrases from the user's message.
- Do not include stopwords or trivial words.
- Use lower case for `text`.
- If the word is grammatically present but semantically wrong in context, set is_valid_in_context = false.
- Do not include every token from the message. Include only useful vocabulary candidates.

detected_topics:
- Return 0..3 short topic labels that match the current user message or the conversation.
- Prefer labels like: vr, games, movies, machine_learning, psychology, music, work, travel, fitness, english_learning.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
Recent dialogue:
{recent_dialogue}

Current user message:
{user_text}

target_words:
{target_words}

candidate_tokens_from_user_message:
{candidate_tokens}

top_user_interests:
{top_interests}

conversation_style_preferences:
{style_preferences}

current_topic:
{current_topic}

Return structured output only.
""")
        ])

        recent_dialogue_lines: list[str] = []
        for msg in recent_messages:
            recent_dialogue_lines.append(f"{msg['role'].upper()}: {msg['content']}")

        chain = prompt | self.structured_llm
        return chain.invoke({
            "recent_dialogue": "\n".join(recent_dialogue_lines),
            "user_text": user_text,
            "target_words": ", ".join(target_words) if target_words else "-",
            "candidate_tokens": ", ".join(candidate_tokens) if candidate_tokens else "-",
            "top_interests": ", ".join(top_interests) if top_interests else "-",
            "style_preferences": ", ".join(style_preferences) if style_preferences else "-",
            "current_topic": current_topic or "-",
        })


# ============================================================
# Conversation service
# ============================================================

class ConversationService:
    def __init__(
        self,
        *,
        chat_memory_service: InMemoryChatMemoryService,
        vocabulary_repository: InMemoryVocabularyRepository,
        user_profile_repository: InMemoryUserProfileRepository,
        interest_service: InterestProfileService,
        word_selector: MockWordSelector,
        runner: LangChainConversationRunner,
    ):
        self.chat_memory_service = chat_memory_service
        self.vocabulary_repository = vocabulary_repository
        self.user_profile_repository = user_profile_repository
        self.interest_service = interest_service
        self.word_selector = word_selector
        self.runner = runner

    def _merge_reply_with_corrections(self, llm_result: ConversationLLMOutput) -> str:
        reply = llm_result.reply_text.strip()

        if not llm_result.mistakes:
            return reply

        correction_lines: list[str] = []
        for m in llm_result.mistakes[:2]:
            correction_lines.append(
                f"Correction: instead of '{m.original}', better say '{m.corrected}'. {m.explanation}"
            )

        return reply + "\n\n" + "\n".join(correction_lines)

    def handle_user_message(self, *, user_id: int, user_text: str) -> dict:
        session_id = self.chat_memory_service.get_or_create_active_session(user_id)

        # Update user interests from raw text before model call
        self.interest_service.update_from_text(user_id, user_text)

        self.chat_memory_service.add_message(
            session_id=session_id,
            role="user",
            content=user_text,
        )

        recent_messages = self.chat_memory_service.get_recent_messages(session_id=session_id, limit=12)
        target_words = self.word_selector.select_for_reply(user_id=user_id, max_words=2)
        candidate_tokens = candidate_new_words(user_text)

        top_interests = self.interest_service.top_interests(user_id, limit=3)
        style_preferences = self.interest_service.top_style_preferences(user_id, limit=3)
        current_topic = self.interest_service.current_topic(user_id)

        llm_result = self.runner.run(
            recent_messages=recent_messages,
            user_text=user_text,
            target_words=target_words,
            candidate_tokens=candidate_tokens,
            top_interests=top_interests,
            style_preferences=style_preferences,
            current_topic=current_topic,
        )

        self.interest_service.apply_detected_topics(user_id, llm_result.detected_topics)
        final_reply = self._merge_reply_with_corrections(llm_result)

        self.chat_memory_service.add_message(
            session_id=session_id,
            role="assistant",
            content=final_reply,
        )

        saved_words: list[str] = []
        rejected_words: list[dict[str, str | None]] = []

        for item in llm_result.new_user_vocabulary:
            word_text = item.text.strip().lower()

            if len(word_text) <= 2:
                continue

            if item.is_valid_in_context:
                self.vocabulary_repository.upsert_new_word(
                    user_id=user_id,
                    text_value=word_text,
                    lemma=item.lemma,
                    translation_text=item.translation_text,
                    context=user_text,
                    source="chat",
                )
                saved_words.append(word_text)
            else:
                rejected_words.append({
                    "text": word_text,
                    "reason": item.reason,
                    "corrected_usage": item.corrected_usage,
                })

        return {
            "session_id": session_id,
            "reply": final_reply,
            "target_words": target_words,
            "saved_words": saved_words,
            "rejected_words": rejected_words,
            "mistakes": [m.model_dump() for m in llm_result.mistakes],
            "detected_topics": llm_result.detected_topics,
            "top_interests": self.interest_service.top_interests(user_id, limit=5),
        }


# ============================================================
# Demo helpers
# ============================================================

def print_user_vocabulary(repo: InMemoryVocabularyRepository, user_id: int) -> None:
    items = sorted(repo.list_user_vocabulary(user_id), key=lambda x: x.text)
    if not items:
        print("\n[User vocabulary is empty]\n")
        return

    print("\n[User vocabulary]")
    for item in items:
        print(
            f"- {item.text} | lemma={item.lemma} | translation={item.translation_text} "
            f"| mastery={item.mastery} | correct_count={item.correct_count} | status={item.status}"
        )
    print()


def print_user_profile(repo: InMemoryUserProfileRepository, user_id: int) -> None:
    profile = repo.get_or_create(user_id)

    ranked_interests = sorted(profile.interests.items(), key=lambda x: x[1], reverse=True)
    ranked_styles = sorted(profile.style_preferences.items(), key=lambda x: x[1], reverse=True)

    print("[User profile]")
    if ranked_interests:
        print("  interests:")
        for topic, score in ranked_interests[:5]:
            print(f"    - {topic}: {score:.2f}")
    else:
        print("  interests: -")

    if ranked_styles:
        print("  style_preferences:")
        for key, score in ranked_styles[:5]:
            print(f"    - {key}: {score:.2f}")
    else:
        print("  style_preferences: -")

    print(f"  current_topic: {profile.current_topic}")
    print(f"  topic_history: {profile.topic_history[-5:]}")
    print()


# ============================================================
# CLI demo
# ============================================================

def main() -> None:
    user_id = 1

    chat_memory_service = InMemoryChatMemoryService()
    vocabulary_repository = InMemoryVocabularyRepository()
    user_profile_repository = InMemoryUserProfileRepository()
    interest_service = InterestProfileService(user_profile_repository)
    word_selector = MockWordSelector()
    runner = LangChainConversationRunner(
        model="qwen2.5:14b-instruct",
        base_url="http://localhost:11434",
        temperature=0.8,
    )

    conversation_service = ConversationService(
        chat_memory_service=chat_memory_service,
        vocabulary_repository=vocabulary_repository,
        user_profile_repository=user_profile_repository,
        interest_service=interest_service,
        word_selector=word_selector,
        runner=runner,
    )

    print("English tutor chat started. Type 'exit' to stop.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        try:
            result = conversation_service.handle_user_message(
                user_id=user_id,
                user_text=user_text,
            )
        except Exception as exc:
            print(f"\n[Error] {exc}\n")
            continue

        print(f"\nBot: {result['reply']}\n")

        if result["saved_words"]:
            print(f"[Saved words] {result['saved_words']}")
        if result["rejected_words"]:
            print(f"[Rejected words] {result['rejected_words']}")
        if result["mistakes"]:
            print(f"[Mistakes] {result['mistakes']}")
        if result["detected_topics"]:
            print(f"[Detected topics] {result['detected_topics']}")
        if result["top_interests"]:
            print(f"[Top interests] {result['top_interests']}")

        print()
        print_user_profile(user_profile_repository, user_id)
        print_user_vocabulary(vocabulary_repository, user_id)


if __name__ == "__main__":
    main()