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
    lexeme_id         BIGINT      NULL REFERENCES lexeme (id) ON DELETE CASCADE,

    collection_name   TEXT        NULL,
    lemma             TEXT        NULL,
    text              TEXT        NULL,
    translation_text  TEXT        NULL,
    translation_lemma TEXT        NULL,

    importance        SMALLINT    NOT NULL DEFAULT 0, -- 0..100
    mastery           SMALLINT    NOT NULL DEFAULT 0, -- 0..100
    status            TEXT        NOT NULL DEFAULT 'new',

    -- SRS
    ease              REAL        NOT NULL DEFAULT 2.5,
    interval_days     INT         NOT NULL DEFAULT 0,
    next_review_at    TIMESTAMPTZ NULL,
    last_review_at    TIMESTAMPTZ NULL,

    -- статистика
    seen_count        INT         NOT NULL DEFAULT 0,
    correct_count     INT         NOT NULL DEFAULT 0,
    wrong_count       INT         NOT NULL DEFAULT 0,
    success_streak    INT         NOT NULL DEFAULT 0,
    fail_count        INT         NOT NULL DEFAULT 0,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (user_id, text)
);
