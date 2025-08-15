# ASR and SER and other models 
```bash
conda deactivate
conda create -n funasr python=3.10 -y
conda activate funasr

# SER model (https://github.com/ddlBoJack/emotion2vec)
# conda create -n funasr python=3.9
# conda activate funasr
# https://blog.csdn.net/qq_34717531/article/details/141159210
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -U funasr
pip install -U modelscope huggingface_hub

# for whisper
pip install transformers

```

## funasr realtime
https://github.com/modelscope/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online_zh.md