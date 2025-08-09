import whisper


def test_whisper():
    model_whisper = whisper.load_model('large')
    text = model_whisper.transcribe(r'..\data\external\audios\share_video_7353765612553768224_.mp3', language='en').get("text", None)
    print(text)