# extract all the file path (.wav) in RAVDESS (find recursively) 
# and ignore files in the dir: '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/RAVDESS/audio_speech_actors_01-24
import os

RAVDESS_DIR = "/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/RAVDESS"
wav_files = []

for root, dirs, files in os.walk(RAVDESS_DIR):
    for f in files:
        if f.endswith(".wav"):
            wav_files.append(os.path.join(root, f))

# sort the file paths
wav_files.sort()

# save in a .txt file
IGNORE_DIR = "audio_speech_actors_01-24"
with open("RAVDESS_file_list.txt", "w") as f:
    for wav_file in wav_files:
        if IGNORE_DIR not in wav_file:
            f.write(f"{wav_file}\n")
