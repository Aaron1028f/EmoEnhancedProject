# Install IndexTTS and use it for emotional TTS

We use IndexTTS2 as our TTS model to generate emotion-aware speech, which is then fed into EmoGene to generate emotion-aware talking head.


## 1. Install IndexTTS2 from official repo
To install IndexTTS2, follow the steps in https://github.com/index-tts/index-tts to install the latest version of IndexTTS2 in `server/models/TTS`, just like the directory in this repository.

Make sure you can run the official demo script successfully, and you can go to next step.

## 2. check the demo .wav file

Copy the content of `server/models/TTS/index-tts/examples` to your directory, details of the usage can be found in `server/models/TTS/index-tts/indextts/infer_v2_streaming_emo.py`.

## 3. Use our modified version of IndexTTS2 for context-aware emotion control with RAG 

Check `server/models/TTS/index-tts/indextts` in this repository, we provide the following scripts:
1. `server/models/TTS/index-tts/indextts/infer_v2_streaming_emo.py` (just for testing)
2. `server/models/TTS/index-tts/indextts/api_indextts.py` (for integration with the main application, with emotion control support)
