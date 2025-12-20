CREATE SCHEMA IF NOT EXISTS public;

CREATE TABLE IF NOT EXISTS users
(
--     Скорее всего будет телеграм id - как основная точка взаимодействия
    id            BIGINT PRIMARY KEY,
    username      TEXT        NOT NULL UNIQUE,
    tg_username   TEXT        NULL,
    email         TEXT        NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_visit_at TIMESTAMPTZ NULL
);

CREATE TABLE IF NOT EXISTS tik_tok_video
(
    video_id                   BIGINT PRIMARY KEY,
    video_link                 TEXT        NOT NULL,
    subtitle_text              TEXT,
    is_transcribed_locally     BOOLEAN     NOT NULL DEFAULT false,
    language_code              VARCHAR(10),
    metadata                   JSONB,
    reason_for_skip_processing TEXT,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_video
(
    user_id    BIGINT      NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    video_id   BIGINT      NOT NULL REFERENCES tik_tok_video (video_id) ON DELETE CASCADE,
    viewed_at  TIMESTAMPTZ NOT NULL,
    is_liked   BOOLEAN     NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, video_id, viewed_at)
);

CREATE TABLE IF NOT EXISTS lexeme
(
    id            BIGSERIAL PRIMARY KEY,
    language_code VARCHAR(10) NOT NULL, -- 'en', 'ru'
    lemma         TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (language_code, lemma)
);

CREATE TABLE IF NOT EXISTS translation
(
    id          BIGSERIAL PRIMARY KEY,
    lexeme_id   BIGINT      NOT NULL REFERENCES lexeme (id) ON DELETE CASCADE,
    provider    TEXT        NULL,
    translation TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (lexeme_id, translation)
);


CREATE TABLE IF NOT EXISTS user_vocabulary
(
    user_id           BIGINT      NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    lexeme_id         BIGINT      NULL REFERENCES lexeme (id) ON DELETE SET NULL,

    -- key: surface form; храним как lower() на уровне приложения
    text              TEXT        NOT NULL,

    lemma             TEXT        NULL,
    translation_text  TEXT        NULL,
    translation_lemma TEXT        NULL,

    -- metadata
    collection_name   TEXT        NULL,

    source            TEXT[]      NOT NULL DEFAULT '{}',
    contexts          TEXT[]      NOT NULL DEFAULT '{}',

    item_type              TEXT        NOT NULL DEFAULT 'WORD',     -- WORD|PHRASE|RUSSIAN_WORD
    difficulty_level_cefr  TEXT        NOT NULL DEFAULT 'UNKNOWN',  -- A1..C2|UNKNOWN

    importance        SMALLINT    NOT NULL DEFAULT 0, -- 0..100
    mastery           SMALLINT    NOT NULL DEFAULT 0, -- 0..100
    status            TEXT        NOT NULL DEFAULT 'new', -- new|learning|known

    passive_knowledge BOOLEAN     NOT NULL DEFAULT FALSE,
    active_knowledge  BOOLEAN     NOT NULL DEFAULT FALSE,

    -- SRS
    ease              REAL        NOT NULL DEFAULT 2.5,
    interval_days     INT         NOT NULL DEFAULT 0,
    next_review_at    TIMESTAMPTZ NULL,
    last_review_at    TIMESTAMPTZ NULL,

    -- stats
    seen_count        INT         NOT NULL DEFAULT 0,
    correct_count     INT         NOT NULL DEFAULT 0,
    wrong_count       INT         NOT NULL DEFAULT 0,
    success_streak    INT         NOT NULL DEFAULT 0,
    fail_count        INT         NOT NULL DEFAULT 0,
    last_seen_at      TIMESTAMPTZ NULL,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (user_id, text)
);

CREATE INDEX idx_user_vocab_user_status
    ON user_vocabulary (user_id, status);

CREATE INDEX idx_user_vocab_user_mastery
    ON user_vocabulary (user_id, mastery DESC);

CREATE INDEX idx_user_vocab_user_last_seen
    ON user_vocabulary (user_id, last_seen_at DESC);

CREATE TABLE lesson
(
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT      NOT NULL REFERENCES users (id) ON DELETE CASCADE,

    title         TEXT        NULL,
    source        TEXT        NULL, -- 'zoom' / 'meet' / 'file'
    source_uri    TEXT        NULL, -- путь/ссылка на файл
    language_mode TEXT        NOT NULL DEFAULT 'ru+en',

    duration_sec  INT         NULL,
    started_at    TIMESTAMPTZ NULL,

    metadata      JSONB       NULL,

    -- ASR payload
    teacher_full_text        TEXT  NULL,
    student_full_text        TEXT  NULL,
    dialog_utterances_jsonb  JSONB NULL,

    -- report & extra
    report_json              JSONB NULL,
    materials_text           TEXT  NULL,

    -- talk metrics (денормализация)
    teacher_time_sec         INT   NULL,
    student_time_sec         INT   NULL,
    student_talk_ratio       REAL  NULL,
    teacher_words_count      INT   NULL,
    student_words_count      INT   NULL,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_lesson_user_created
    ON lesson (user_id, created_at DESC);

CREATE INDEX idx_lesson_dialog_jsonb_gin
    ON lesson USING GIN (dialog_utterances_jsonb);



CREATE TABLE lesson_turn
(
    id         BIGSERIAL PRIMARY KEY,
    lesson_id  BIGINT      NOT NULL REFERENCES lesson (id) ON DELETE CASCADE,

    turn_id    INT         NOT NULL, -- порядок в таймлайне

    role       TEXT        NULL, -- teacher|student|null

    start_s    REAL        NOT NULL,
    end_s      REAL        NOT NULL,

    text       TEXT        NOT NULL,

    is_question   BOOLEAN NOT NULL DEFAULT FALSE,
    language_hint TEXT    NULL, -- en|ru|mixed|null
    stats         JSONB   NULL, -- {avg_asr_prob, words_count, ...}

    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT ck_turn_time CHECK (start_s >= 0 AND end_s >= start_s),
    CONSTRAINT uq_lesson_turn UNIQUE (lesson_id, turn_id)
);

CREATE INDEX idx_lesson_turn_lesson_start
    ON lesson_turn (lesson_id, start_s);
