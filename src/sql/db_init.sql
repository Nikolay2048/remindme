CREATE SCHEMA IF NOT EXISTS public;

CREATE TABLE IF NOT EXISTS tik_tok
(
    video_id                   VARCHAR(64) PRIMARY KEY,
    video_link                 TEXT,
    viewed_at                  TIMESTAMP WITHOUT TIME ZONE,
    subtitle_text              TEXT,
    is_liked                   bool,
    is_transcribed_locally     bool,
    language_code              VARCHAR(10),
    metadata                   JSONB,
    reason_for_skip_processing TEXT
);