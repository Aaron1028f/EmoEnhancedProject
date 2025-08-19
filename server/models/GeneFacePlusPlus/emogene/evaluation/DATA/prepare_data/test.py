import os
test_audio_path = "/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
# get audio file name
audio_filename = os.path.basename(test_audio_path)

output_base_dir = '/home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/DATA/temp'
output_mid_dir = 'May/emogene_ver'

output_base_dir = os.path.join(output_base_dir, output_mid_dir)
print(output_base_dir)

# 從test_audio_path 的倒數第三個dir開始擷取
output_mid_dir = os.path.join(output_base_dir, *test_audio_path.split(os.sep)[-3:-1])
print(output_mid_dir)

print(audio_filename)

# combine to output_path
output_path = os.path.join(output_mid_dir, audio_filename)
print(output_path)