from transformers import *
import librosa
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
model = Wav2Vec2ForCTC.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")

def predict_emotion(audio_path):
    audio, rate = librosa.load(audio_path, sr=16000)
    inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(inputs.input_values)
        predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1), dim=-1)  # Average over sequence length
        predicted_label = torch.argmax(predictions, dim=-1)
        emotion = model.config.id2label[predicted_label.item()]
    return emotion

emotion = predict_emotion("_test/emo_wav/angry2.wav")
print(f"Predicted emotion: {emotion}")
# >> Predicted emotion: angry
