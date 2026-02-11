from vosk import Model, KaldiRecognizer
import wave

# Скачайте модели с https://alphacephei.com/vosk/models
model_ru = Model("vosk-model-ru-0.42")
model_en = Model("vosk-model-en-us-0.42")


def transcribe(file_path, language="ru"):
    model = model_ru if language == "ru" else model_en
    wf = wave.open(file_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    transcription = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            transcription += result

    transcription += rec.FinalResult()
    return transcription


# Пример вызова
text_ru = transcribe("./data/Запись (118).m4a", language="ru")