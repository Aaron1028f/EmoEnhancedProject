## Reproduce the `geneface` conda environment (for just inference)

### `geneface` environment
```bash
conda create -n geneface_py310 python=3.10
conda activate geneface_py310
# conda deactivate
# conda env remove --name geneface_py310


conda install conda-forge::ffmpeg # ffmpeg with libx264 codec to turn images to video
# install cuda 11.7 in conda, this is important when installing pytorch3d
conda install nvidia/label/cuda-11.7.1::cuda


# 我们推荐安装torch2.0.1+cuda11.7. 已经发现 torch=2.1+cuda12.1 会导致 torch-ngp 错误
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html



# 从源代码安装，需要比较长的时间 (如果遇到各种time-out问题，建议使用代理)
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# MMCV安装
pip install cython
pip install openmim==0.3.9
mim install mmcv==2.1.0 # 使用mim来加速mmcv安装

# 其他依赖项
# sudo apt-get install libasound2-dev portaudio19-dev

# use conda instead for better environment management
conda install -c conda-forge alsa-lib
conda install -c conda-forge portaudio

pip install -r docs/prepare_env/requirements.txt -v

# 构建torch-ngp
bash docs/prepare_env/install_ext.sh 
```

### Handling the error messages
```bash
# ERROR: Failed to build installable wheels for some pyproject.toml based projects (dlib)
# 解決安裝dlib時的cmake問題 => 直接改用conda安裝dlib
conda install -c conda-forge dlib

# Disabling PyTorch because PyTorch >= 2.1 is required but found 2.0.1+cu117
# ImportError: 
# HubertModel requires the PyTorch library but it was not found in your environment.
pip install transformers==4.46.2


# If you got this error by calling handler(<some type>) within `__get_pydantic_core_schema__` then you likely need to call `handler.generate_schema(<some type>)` since we do not call `__get_pydantic_core_schema__` on `<some type>` otherwise to avoid infinite recursion.
# 用gradio app推論時遇到的pydantic相關問題 => 直接將fastapi和pydantic的版本固定到穩定版本

# fastapi                   0.111.0                  pypi_0    pypi
# fastapi-cli               0.0.8                    pypi_0    pypi
# pydantic                  2.10.3                   pypi_0    pypi
# pydantic-core             2.27.1                   pypi_0    pypi
# pydantic-settings         2.9.1                    pypi_0    pypi
pip install fastapi==0.111.0
pip install fastapi-cli==0.0.8
pip install pydantic==2.10.3
pip install pydantic-core==2.27.1
pip install pydantic-settings==2.9.1

```
```bash
# ModuleNotFoundError: No module named 'soxr'
pip install soxr
```