from faster_whisper import WhisperModel

from src.services.lesson_analyzer.transcribe import LessonTranscriber


def rr():
    # transcriber = LessonTranscriber(
    #         whisper_model_size="large",
    #         device="cuda",
    #         compute_type="float16",
    #         language=None,
    #         utterance_gap_ms=1000,
    #     )"C:\Users\Admin\PycharmProjects\ReMindMe\data\processed\lessons\lesson_2026-01-21_22-50-18\_asr_chunks\student.wav.chunk_0074.wav"

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _info = model.transcribe(
        # "./data/Запись (118).m4a",
        r"C:\Users\Admin\PycharmProjects\ReMindMe\data\processed\lessons\lesson_2026-01-20_16-53-31\student.wav",
        # initial_prompt="Do NOT translate. Keep the original language (Russian and English).",
        task="transcribe",
        temperature=0.0,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        language=None,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    print(_info)
    for seg in segments:
        # print(seg)
        if seg.text:
            print(seg.text)
        else:
            print(seg)

def tt():
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segments, _info = model.transcribe(
        # "./data/Запись (118).m4a",
        r"C:\Users\Admin\PycharmProjects\ReMindMe\data\processed\lessons\lesson_2026-01-21_22-50-18\_asr_chunks\student.wav.chunk_0074.wav",
        language=None,  # авто ru/en
        task="transcribe",  # НЕ переводить
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        temperature=0.0,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=True
    )
    print(_info)
    for seg in segments:
        # print(seg)
        if seg.text:
            print(seg.text)
        else:
            print(seg)
tt()