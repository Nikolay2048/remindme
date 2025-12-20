from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict

from .models import WordToken, Utterance
from ..word_processor import WordProcessor, detect_lang
from ...utils.text_norm import normalize_vocab_key


@dataclass(frozen=True)
class CefrLexicon:
    word_to_cefr: Dict[str, str]

    def lookup(self, lemma: str) -> Optional[str]:
        if not lemma:
            return None
        return self.word_to_cefr.get(lemma.lower().strip())


def default_cefr_level() -> str:
    return "UNKNOWN"


def clip_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def status_from_mastery(mastery: int) -> str:
    if mastery >= 80:
        return "known"
    if mastery >= 30:
        return "learning"
    return "new"


class VocabAnalyzer:
    """
    v1 rules:
      - vocabulary keys are LEMMAS (via WordProcessor)
      - activated: only WORD lemmas (phrases not activated in v1)
      - russian_used: RU lemmas to learn translation later
    """

    def __init__(
        self,
        *,
        wp: Optional[WordProcessor] = None,
        cefr: Optional[CefrLexicon] = None,
        low_prob: float = 0.55,
        high_prob: float = 0.75,
        long_pause_ms: int = 500,
        mastery_inc_confident: int = 3,
        mastery_dec_uncertain: int = 2,
        mastery_inc_neutral: int = 1,
        activated_min_confident_hits: int = 2,
        activated_min_mastery_delta: int = 5,
        new_candidates_limit: int = 40,
        uncertain_limit: int = 20,
        # phrases are optional (very conservative)
        phrase_whitelist: Optional[set[str]] = None,
        phrase_min_count: int = 2,
    ):
        self.wp = wp or WordProcessor(keep_stopwords=False, min_len=2)
        self.cefr = cefr

        self.low_prob = low_prob
        self.high_prob = high_prob
        self.long_pause_ms = long_pause_ms

        self.mastery_inc_confident = mastery_inc_confident
        self.mastery_dec_uncertain = mastery_dec_uncertain
        self.mastery_inc_neutral = mastery_inc_neutral

        self.activated_min_confident_hits = activated_min_confident_hits
        self.activated_min_mastery_delta = activated_min_mastery_delta

        self.new_candidates_limit = new_candidates_limit
        self.uncertain_limit = uncertain_limit

        self.phrase_whitelist = phrase_whitelist
        self.phrase_min_count = phrase_min_count

    # ----------------------------
    # Public API
    # ----------------------------

    def analyze(
        self,
        *,
        lesson_id: str,
        student_words: List[WordToken],
        teacher_words: List[WordToken],
        student_utts: List[Utterance],
        existing_user_vocab: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        existing_user_vocab: mapping text(lemma) -> row-like dict from user_vocabulary
        """
        # Build examples from utterances (most readable)
        examples = self._examples_for_lemmas(student_words, student_utts)

        # 1) Collect lemma hits (EN) + RU lemma hits
        en_hits, ru_hits = self._collect_lemma_hits(student_words)

        # 2) Optional phrase hits (kept separate; they can go into new_candidates only)
        phrase_hits = self._collect_phrase_hits(student_utts)
        # unify pool for new-candidates (EN words + phrases)
        new_pool_counts: Dict[str, int] = dict(en_hits)
        for p, cnt in phrase_hits.items():
            new_pool_counts[p] = new_pool_counts.get(p, 0) + cnt

        # 3) Uncertainty (word-level, by lemma)
        uncertain_by_lemma = self._compute_uncertain_by_lemma(student_words)

        # 4) Build upserts & lists
        upserts: List[Dict[str, Any]] = []
        new_candidates: List[Dict[str, Any]] = []
        activated: List[Dict[str, Any]] = []

        for item, used_cnt in new_pool_counts.items():
            before = existing_user_vocab.get(item, {})
            is_phrase = (" " in item)

            cefr_level = self._cefr_level(item, before.get("difficulty_level_cefr"))

            mastery_before = int(before.get("mastery", 0) or 0)
            status_before = str(before.get("status", "new") or "new")

            confident_hits, uncertain_hits = (0, 0)
            if not is_phrase:
                confident_hits, uncertain_hits = self._confidence_hits_for_lemma(item, student_words)

            mastery_delta = self._calc_mastery_delta(
                confident_hits=confident_hits,
                uncertain_hits=uncertain_hits,
                total_hits=used_cnt,
                is_phrase=is_phrase,
            )

            mastery_after = clip_int(mastery_before + mastery_delta, 0, 100)
            status_after = status_from_mastery(mastery_after)

            ex = examples.get(item)
            contexts_to_add = [e["context"] for e in (ex or [])[:2]]

            upserts.append(self._make_upsert(
                lesson_id=lesson_id,
                item=item,
                item_type=("PHRASE" if is_phrase else "WORD"),
                cefr_level=cefr_level,
                mastery_before=mastery_before,
                mastery_after=mastery_after,
                status_before=status_before,
                status_after=status_after,
                mastery_delta=mastery_delta,
                seen_inc=used_cnt,
                correct_inc=confident_hits,
                wrong_inc=uncertain_hits,
                contexts_to_add=contexts_to_add,
                before=before,
            ))

            # new candidates logic (spoken in lesson + new-ish in history)
            if self._is_new_candidate(item=item, used_cnt=used_cnt, before=before):
                new_candidates.append({
                    "item": item,
                    "item_type": "PHRASE" if is_phrase else "WORD",
                    "difficulty_level_cefr": cefr_level,
                    "student_used_count": used_cnt,
                    "example": (ex[0] if ex else None),
                })

            # activated only for WORD items that existed before
            if self._is_activated(
                item=item,
                before=before,
                mastery_delta=mastery_delta,
                status_before=status_before,
                status_after=status_after,
                confident_hits=confident_hits,
            ):
                activated.append({
                    "item": item,
                    "item_type": "WORD",
                    "difficulty_level_cefr": cefr_level,
                    "mastery_delta": mastery_delta,
                    "examples": (ex[:2] if ex else []),
                })

        # 5) Uncertain top-20
        uncertain_top20 = self._format_uncertain_top(
            uncertain_by_lemma=uncertain_by_lemma,
            examples=examples,
            existing_user_vocab=existing_user_vocab,
        )

        # 6) Russian-used list + upserts
        russian_report, russian_upserts = self._format_russian_used(
            lesson_id=lesson_id,
            ru_hits=ru_hits,
            examples=examples,
            existing_user_vocab=existing_user_vocab,
        )
        upserts.extend(russian_upserts)

        # 7) Limit/sort
        new_candidates = self._limit_new_candidates(new_candidates)
        activated = self._sort_activated(activated)

        action_items = self._make_action_items(new_candidates, uncertain_top20)

        return {
            "vocabulary": {
                "new_candidates": new_candidates,
                "activated": activated,
                "uncertain_top20": uncertain_top20,
                "russian_used": russian_report,
            },
            "action_items": action_items,
            "upserts": upserts,
            "aggregates": {
                "new_items_count": len(new_candidates),
                "activated_items_count": len(activated),
                "uncertain_items_count": len(uncertain_top20),
                "russian_items_count": len(russian_report),
            }
        }

    # ----------------------------
    # Internals
    # ----------------------------

    def _cefr_level(self, item: str, existing_level: Optional[str]) -> str:
        if existing_level and str(existing_level).strip():
            return str(existing_level)
        if self.cefr and " " not in item:  # CEFR usually for single words
            lvl = self.cefr.lookup(item)
            if lvl:
                return lvl
        return default_cefr_level()

    def _collect_lemma_hits(self, student_words: List[WordToken]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Returns (en_lemma_hits, ru_lemma_hits)
        """
        en = Counter()
        ru = Counter()

        for w in student_words:
            raw = (w.text or "").strip()
            if not raw:
                continue
            raw = normalize_vocab_key(w.text)
            lang = detect_lang(raw)
            if lang == "ru":
                lemma = self.wp.get_lexeme(raw, language="ru")
                if lemma:
                    ru[lemma] += 1
                continue

            if lang == "en":
                lemma = self.wp.get_lexeme(raw, language="en")
                if lemma:
                    en[lemma] += 1
                continue

            # mixed/unknown: try both parts via get_lexemes
            lex = self.wp.get_lexemes(raw, language=None, as_set=False)
            for lx in lex:
                # lexemes can be en or ru; detect quickly
                l = detect_lang(lx)
                if l == "ru":
                    ru[lx] += 1
                elif l == "en":
                    en[lx] += 1

        return dict(en), dict(ru)

    def _collect_phrase_hits(self, student_utts: List[Utterance]) -> Dict[str, int]:
        """
        Optional phrases: controlled by whitelist or min_count.
        We keep phrases in surface form (lowercase) because lemmatizing phrases is messy in MVP.
        """
        if self.phrase_whitelist is None and self.phrase_min_count <= 1:
            return {}

        c = Counter()
        for u in student_utts:
            t = (u.text or "").strip().lower()
            if not t:
                continue
            # only english utterances for phrase mining
            if detect_lang(t) == "ru":
                continue

            toks = [x for x in t.split() if x]
            # build 2..4 grams
            for n in (2, 3, 4):
                for i in range(0, max(0, len(toks) - n + 1)):
                    p = " ".join(toks[i:i+n]).strip(" .,!?:;\"()[]{}<>")
                    if not p:
                        continue
                    if self.phrase_whitelist is not None and p not in self.phrase_whitelist:
                        continue
                    c[p] += 1

        if self.phrase_whitelist is None:
            c = Counter({p: cnt for p, cnt in c.items() if cnt >= self.phrase_min_count})

        return dict(c)

    def _confidence_hits_for_lemma(self, lemma: str, student_words: List[WordToken]) -> Tuple[int, int]:
        """
        We need to count confidence/pauses for occurrences belonging to the lemma.
        We map each word token -> lemma again. (OK for MVP; optimize later by caching.)
        """
        confident = 0
        uncertain = 0
        for w in student_words:
            raw = (w.text or "").strip()
            if not raw or detect_lang(raw) != "en":
                continue
            l = self.wp.get_lexeme(raw, language="en")
            if l != lemma:
                continue

            low_conf = (w.prob is not None and w.prob < self.low_prob)
            long_pause = (w.pause_ms is not None and w.pause_ms >= self.long_pause_ms)

            if (w.prob is None or w.prob >= self.high_prob) and not long_pause:
                confident += 1
            elif low_conf or long_pause:
                uncertain += 1

        return confident, uncertain

    def _compute_uncertain_by_lemma(self, student_words: List[WordToken]) -> Dict[str, Dict[str, Any]]:
        agg: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "pause_before": 0,
            "low_confidence": 0,
            "score": 0.0,
        })

        for w in student_words:
            raw = (w.text or "").strip()
            if not raw:
                continue
            if detect_lang(raw) != "en":
                continue

            lemma = self.wp.get_lexeme(raw, language="en")
            if not lemma:
                continue

            pause_hit = (w.pause_ms is not None and w.pause_ms >= self.long_pause_ms)
            low_conf_hit = (w.prob is not None and w.prob < self.low_prob)
            if not (pause_hit or low_conf_hit):
                continue

            rec = agg[lemma]
            if pause_hit:
                rec["pause_before"] += 1
                rec["score"] += 2.0
            if low_conf_hit:
                rec["low_confidence"] += 1
                rec["score"] += 2.0

        return dict(agg)

    def _calc_mastery_delta(
        self,
        *,
        confident_hits: int,
        uncertain_hits: int,
        total_hits: int,
        is_phrase: bool,
    ) -> int:
        if total_hits <= 0:
            return 0
        if is_phrase:
            return self.mastery_inc_neutral * total_hits

        neutral = max(0, total_hits - confident_hits - uncertain_hits)
        return (
            confident_hits * self.mastery_inc_confident
            - uncertain_hits * self.mastery_dec_uncertain
            + neutral * self.mastery_inc_neutral
        )

    def _is_new_candidate(self, *, item: str, used_cnt: int, before: Dict[str, Any]) -> bool:
        if used_cnt <= 0:
            return False
        if not before:
            return True
        status = str(before.get("status", "new") or "new")
        seen = int(before.get("seen_count", 0) or 0)
        return status == "new" and seen < 2

    def _is_activated(
        self,
        *,
        item: str,
        before: Dict[str, Any],
        mastery_delta: int,
        status_before: str,
        status_after: str,
        confident_hits: int,
    ) -> bool:
        if not before:
            return False
        if " " in item:
            return False  # phrases not activated in v1
        if confident_hits < self.activated_min_confident_hits:
            return False
        if status_after != status_before:
            return True
        return mastery_delta >= self.activated_min_mastery_delta

    def _make_upsert(
        self,
        *,
        lesson_id: str,
        item: str,
        item_type: str,
        cefr_level: str,
        mastery_before: int,
        mastery_after: int,
        status_before: str,
        status_after: str,
        mastery_delta: int,
        seen_inc: int,
        correct_inc: int,
        wrong_inc: int,
        contexts_to_add: List[str],
        before: Dict[str, Any],
    ) -> Dict[str, Any]:
        existing_contexts = list(before.get("contexts") or [])
        existing_source = list(before.get("source") or [])

        src_tag = f"lesson:{lesson_id}"
        if src_tag not in existing_source:
            existing_source.append(src_tag)

        for c in contexts_to_add:
            if c and c not in existing_contexts:
                existing_contexts.append(c)
        existing_contexts = existing_contexts[-20:]

        passive = bool(before.get("passive_knowledge", False)) or True
        active = bool(before.get("active_knowledge", False))
        if mastery_after >= 60:
            active = True

        return {
            "text": item,  # lemma (or phrase surface)
            "item_type": item_type,
            "difficulty_level_cefr": cefr_level,
            "mastery_before": mastery_before,
            "mastery_after": mastery_after,
            "mastery_delta": mastery_delta,
            "status_before": status_before,
            "status_after": status_after,
            "seen_inc": int(seen_inc),
            "correct_inc": int(correct_inc),
            "wrong_inc": int(wrong_inc),
            "passive_knowledge": passive,
            "active_knowledge": active,
            "contexts": existing_contexts,
            "source": existing_source,
        }

    def _examples_for_lemmas(
        self,
        student_words: List[WordToken],
        student_utts: List[Utterance],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        lemma/phrase -> [{t_start_sec, context}]
        For words we map each token to its lemma and attach utterance text.
        """
        utts = sorted(student_utts, key=lambda u: (u.start, u.end))

        def find_utt_for_time(t: float) -> Optional[Utterance]:
            for u in utts:
                if u.start <= t <= u.end:
                    return u
            return None

        examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for w in student_words:
            raw = (w.text or "").strip()
            if not raw:
                continue
            u = find_utt_for_time(w.start)
            if not u:
                continue

            lang = detect_lang(raw)
            if lang == "en":
                lemma = self.wp.get_lexeme(raw, language="en")
                if not lemma:
                    continue
                key = lemma
            elif lang == "ru":
                lemma = self.wp.get_lexeme(raw, language="ru")
                if not lemma:
                    continue
                key = lemma
            else:
                # for mixed, add both parts as separate keys
                lex = self.wp.get_lexemes(raw, language=None, as_set=False)
                for lx in lex:
                    examples[lx].append({
                        "t_start_sec": round(float(u.start), 2),
                        "context": u.text,
                    })
                continue

            examples[key].append({
                "t_start_sec": round(float(u.start), 2),
                "context": u.text,
            })

        # phrase examples (only if whitelist exists)
        if self.phrase_whitelist:
            for u in utts:
                t = (u.text or "").lower()
                for p in self.phrase_whitelist:
                    if p in t:
                        examples[p].append({
                            "t_start_sec": round(float(u.start), 2),
                            "context": u.text,
                        })

        # trim
        for k in list(examples.keys()):
            examples[k] = examples[k][:5]
        return dict(examples)

    def _format_uncertain_top(
        self,
        *,
        uncertain_by_lemma: Dict[str, Dict[str, Any]],
        examples: Dict[str, List[Dict[str, Any]]],
        existing_user_vocab: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rows = []
        for lemma, rec in uncertain_by_lemma.items():
            why = []
            if rec.get("pause_before", 0) > 0:
                why.append("pause_before")
            if rec.get("low_confidence", 0) > 0:
                why.append("low_confidence")

            cefr_level = self._cefr_level(lemma, existing_user_vocab.get(lemma, {}).get("difficulty_level_cefr"))
            ex = examples.get(lemma)
            rows.append({
                "item": lemma,
                "item_type": "WORD",
                "difficulty_level_cefr": cefr_level,
                "uncertainty_score": round(float(rec.get("score", 0.0)), 2),
                "why": why,
                "example": (ex[0] if ex else None),
            })

        rows.sort(key=lambda r: (-float(r["uncertainty_score"]), r["item"]))
        return rows[: self.uncertain_limit]

    def _format_russian_used(
        self,
        *,
        lesson_id: str,
        ru_hits: Dict[str, int],
        examples: Dict[str, List[Dict[str, Any]]],
        existing_user_vocab: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        report_rows = []
        upserts = []

        for lemma, cnt in ru_hits.items():
            ex = examples.get(lemma)
            report_rows.append({
                "item": lemma,
                "item_type": "RUSSIAN_WORD",
                "student_used_count": int(cnt),
                "needs_translation": True,
                "example": (ex[0] if ex else None),
            })

            before = existing_user_vocab.get(lemma, {})
            mastery_before = int(before.get("mastery", 0) or 0)
            status_before = str(before.get("status", "new") or "new")

            upserts.append(self._make_upsert(
                lesson_id=lesson_id,
                item=lemma,
                item_type="RUSSIAN_WORD",
                cefr_level=before.get("difficulty_level_cefr") or default_cefr_level(),
                mastery_before=mastery_before,
                mastery_after=mastery_before,
                status_before=status_before,
                status_after=status_before,
                mastery_delta=0,
                seen_inc=int(cnt),
                correct_inc=0,
                wrong_inc=0,
                contexts_to_add=[e["context"] for e in (ex or [])[:1]],
                before=before,
            ))

        report_rows.sort(key=lambda r: (-r["student_used_count"], r["item"]))
        return report_rows[:40], upserts

    def _limit_new_candidates(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def cefr_rank(lvl: str) -> int:
            order = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6, "UNKNOWN": 0}
            return order.get(lvl or "UNKNOWN", 0)

        rows.sort(key=lambda r: (-int(r["student_used_count"]), -cefr_rank(r.get("difficulty_level_cefr")), r["item"]))
        return rows[: self.new_candidates_limit]

    def _sort_activated(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows.sort(key=lambda r: (-int(r.get("mastery_delta", 0)), r["item"]))
        return rows[:20]

    def _make_action_items(self, new_candidates: List[Dict[str, Any]], uncertain_top20: List[Dict[str, Any]]) -> Dict[str, Any]:
        words = []
        for r in uncertain_top20:
            words.append(r["item"])
        for r in new_candidates:
            if r["item"] not in words and " " not in r["item"]:
                words.append(r["item"])
        return {
            "repeat_words": words[:5],
            "focus_problems": [
                "Пауза перед ответом (если встречается)",
                "Уверенность в произношении/дикции (если встречается)"
            ],
            "speaking_goal": "Начинать ответ короткой простой фразой без длинной паузы"
        }
