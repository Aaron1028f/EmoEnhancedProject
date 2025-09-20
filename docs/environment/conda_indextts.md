## indexTTS conda environment
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

## run demo
```bash
cd server/models/TTS/index-tts

conda activate indextts
# # for deepspeed, deepspeed need to use python in the uv environment
# source .venv/bin/activate

# run webui
uv run webui.py
uv run webui.py -h
uv run webui.py --cuda_kernel --port 40000
--fp16 
--deepspeed 


CUDA_VISIBLE_DEVICES=0,1,2,3 uv run webui.py --cuda_kernel --port 40000

# best setting now
CUDA_VISIBLE_DEVICES=0 uv run webui.py --cuda_kernel --port 40000 --fp16
CUDA_VISIBLE_DEVICES=0 uv run webui.py --cuda_kernel --port 40000 --deepspeed
#################################################################################
# RTF ~= 0.9 (This one runs the fastest in testing)
CUDA_VISIBLE_DEVICES=0 uv run webui.py --cuda_kernel --port 40000
#################################################################################


# RTF ~= 1.0 (??)
CUDA_VISIBLE_DEVICES=0 uv run webui.py --cuda_kernel --port 40000 --deepspeed


# run inferernce file
uv run indextts/infer_v2.py

```

## run TTS server
```bash
cd server/models/TTS/index-tts
conda activate indextts
CUDA_VISIBLE_DEVICES=0 uv run indextts/api_indextts.py --cuda_kernel --port 40000 --deepspeed
```



## solve problems
### deepspeed CUDA error
```bash
# edit the bashrc file
code ~/.bashrc
# add the following lines at the end of the file (CUDA_HOME can be found by using `which nvcc` command)

# handle with deepspeed CUDA error
export CUDA_HOME=/home/aaron/.conda/envs/indextts/bin/nvcc 
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# save the file and restart the terminal
source ~/.bashrc

# ====================================================
# https://github.com/index-tts/index-tts/issues/164#issuecomment-2903453206
# https://anaconda.org/conda-forge/ninja
# conda install conda-forge::ninja # (not needed)


```