# https://huggingface.co/emotion2vec/emotion2vec_plus_large

from funasr import AutoModel

model = AutoModel(model="iic/emotion2vec_plus_large", disable_update=True)

# wav_file = f"{model.model_path}/example/test.wav"
# wav_file = "./angry2.wav"
wav_file = "./ted1.wav"

import time
print('start inference')
start = time.time()
res = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=True)
print(f"Time taken: {time.time() - start:.2f} seconds")
# print(res)
# for key, value in res[0].items():
#     print(f"{key}: {value}")

mapping_table = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']

for emotion, score in zip(mapping_table, res[0]['scores']):
    print(f"{emotion}: {score}")
    
# print the emotion with the highest score
max_emotion = max(zip(mapping_table, res[0]['scores']), key=lambda x: x[1])
print(f"Highest emotion: {max_emotion[0]} with score {max_emotion[1]}")

print('-'*100)

print(res[0]['feats'])
print(res[0]['feats'].shape) # (1024,)