# Using syncnet to evaluation lip sync
source: 
https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation
https://github.com/joonson/syncnet_python


### prepare the environment
```bash
conda deactivate
conda create -n syncnet python=3.10 -y
conda activate syncnet

# clone the repo
cd server/models/GeneFacePlusPlus/emogene/evaluation/LSE
git clone https://github.com/joonson/syncnet_python.git 
cd syncnet_python/

# install the requirements
cd syncnet_python/
pip install -r requirements.txt

sh download_model.sh
# (environment dependencies completed)


# prepare the evaluation scripts (from https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation)
# put the .py and .sh files from /scores_LSE to /syncnet_python


```

### Running the evaluation scripts:
```bash
# python calculate_scores_LRS.py --data_root /path/to/video/data/root --tmp_dir tmp_dir/
conda activate syncnet
cd server/models/GeneFacePlusPlus/emogene/evaluation/LSE/syncnet_python/

# sh calculate_scores_real_videos.sh /path/to/video/data/root
sh calculate_scores_real_videos.sh /home/aaron/project/server/models/GeneFacePlusPlus/emogene/evaluation/LSE/syncnet_python/EmoGene_eval_data


# evaluation for ted speech videos (ted1, ted2)
# /home/aaron/project/server/models/GeneFacePlusPlus/emogene/DATA/evaluation_usage
sh calculate_scores_real_videos.sh /home/aaron/project/server/models/GeneFacePlusPlus/emogene/DATA/evaluation_usage

```

### Handle the errors
```bash
# Use the following dependencies instead to avoid errors of numpy
# https://blog.csdn.net/m0_45267220/article/details/142910007
# torch==1.11.0
# torchvision==0.12.0
# numpy==1.22.4
# scipy==1.13.1
# scenedetect==0.6.0
# opencv-contrib-python
# python_speech_features

# My final solution (NVIDIA A40)
conda install nvidia/label/cuda-11.7.1::cuda
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html
# pip install numpy==1.23.5 scipy==1.11.1
pip install numpy==1.22.4 scipy==1.13.1

pip install opencv-contrib-python==4.11.0.86
# pip install scenedetect==0.5.1
pip install scenedetect==0.6.0
```