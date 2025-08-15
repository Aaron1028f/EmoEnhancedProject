from openai import OpenAI

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(audio_file_path):
    audio_file = open(audio_file_path, "rb")
    transcript = client.audio.transcriptions.create(
    model="gpt-4o-transcribe",
    file=audio_file
    )
    return transcript.text

if __name__ == "__main__":
    audio_file_path = "ted1.wav"
    transcript = transcribe_audio(audio_file_path)
    print("Transcription Result:", transcript)
    