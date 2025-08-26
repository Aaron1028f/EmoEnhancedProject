from funasr import AutoModel
import time
import glob
from tqdm import tqdm

model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)

# wav_file_dir = '/home/aaron/project/server/models/SER/'
wav_file_dir = '/home/aaron/project/server/models/TTS/GPT-SoVITS/DATA/Feng_EP32/slicer'

MAPPING_TABLE = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']

emotion2wavfile_dict = {
    'angry': [],
    'disgusted': [],
    'fearful': [],
    'happy': [],
    'neutral': [],
    'other': [],
    'sad': [],
    'surprised': [],
    'unknown': []
}

print("Starting processing...")
wav_files = glob.glob(f"{wav_file_dir}/*.wav")
for wav_file in wav_files:
    # print(f'Processing file: {wav_file}')
    res = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=True)
    # print the emotion with the highest score
    max_emotion = max(zip(MAPPING_TABLE, res[0]['scores']), key=lambda x: x[1])
    # print(f"Highest emotion: {max_emotion[0]} with score {max_emotion[1]}")
    # print('-' * 60)
    
    # record the data
    emotion2wavfile_dict[max_emotion[0]].append(wav_file)

# print
print('=' * 60)
print("Analysis Results:")
for emotion, files in emotion2wavfile_dict.items():
    if emotion != 'neutral':
        print(f"{emotion}: {len(files)} files")
print('=' * 60)

# print the emotional files
for emotion, files in emotion2wavfile_dict.items():
    if emotion != 'neutral' and len(files) > 0:
        print(f"Files for emotion '{emotion}':")
        for f in files:
            print(f" - {f}")
        print('-' * 60)