# indexTTS conda environment
```bash
conda deactivate
conda create -n indextts python=3.10 -y
conda activate indextts

# install cuda toolkit 12.8 (https://anaconda.org/nvidia/cuda-toolkit)
# conda install nvidia/label/cuda-12.8.0::cuda-toolkit (this one failed)
conda install nvidia/label/cuda-12.8.1::cuda-toolkit # (available)

# download repo
git lfs install
cd server/models/TTS/
git clone https://github.com/index-tts/index-tts.git && cd index-tts
git lfs pull  # download large repository files

# start installing uv requirements (https://github.com/index-tts/index-tts)
uv sync --all-extras

# download required models
uv tool install "huggingface_hub[cli]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# Checking PyTorch GPU Acceleration
uv run tools/gpu_check.py

```

# run demo
```bash
cd server/models/TTS/index-tts

conda activate indextts
# source .venv/bin/activate

# run webui
uv run webui.py
uv run webui.py -h
uv run webui.py --cuda_kernel --port 40000
--fp16 
--deepspeed 


CUDA_VISIBLE_DEVICES=0,1,2,3 uv run webui.py --cuda_kernel --port 40000

# best setting now
CUDA_VISIBLE_DEVICES=0 uv run webui.py --cuda_kernel --port 40000
CUDA_VISIBLE_DEVICES=0 uv run webui.py --cuda_kernel --port 40000 --fp16


# run inferernce file
uv run indextts/infer_v2.py

```