# https://huggingface.co/openai/whisper-large-v3-turbo

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# loading the model and processor
device = "cuda:1" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,  # Enable timestamps if needed
    language="zh",  # Set language to Chinese(traditional)
)

def transcribe_audio(file_path):
    """
    Transcribe audio from a given file path using the Whisper model.
    
    Args:
        file_path (str): Path to the audio file to be transcribed.
        
    Returns:
        str: Transcribed text from the audio.
    """
    result = pipe(file_path)
    return result["text"] if "text" in result else ""


if __name__ == "__main__":
    # Example usage
    # audio_file = "ted1.wav"  # Replace with your audio file path
    audio_file = "feng_ep01.wav"  # Replace with your audio file path
    transcription = transcribe_audio(audio_file)
    print(transcription)
    
# 在英国的初级保健也就是基层医疗就像我们的基层门诊一样，他们统计在2019年统计了75万人75万人里面有找到有33,367个癌症，
# 在这样的资料库里面他们有发现说维他命B12大于1000，大于1000的就是血液里面高于上限的超过上限。


#===

# 在英国的初级保健喔，也就是基层医疗，就像我们的基层门诊一样，他们统计喔，在2019年统计了75万人，75万人里面有找到有33,367个癌症，
# 在这样的资料库里面阿，他们有，有发现说维他命B12大于1000哪，大于1000的就是血液里面高于這個上限哪，超过上限哪。