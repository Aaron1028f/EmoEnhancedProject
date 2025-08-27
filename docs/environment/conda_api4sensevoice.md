### api4sensevoice

source: https://github.com/0x5446/api4sensevoice
```bash
# download code
git clone https://github.com/0x5446/api4sensevoice.git
cd api4sensevoice
rm -rf .git
```


### installation

```bash
conda deactivate

conda create -n api4sensevoice python=3.10
conda activate api4sensevoice

conda install -c conda-forge ffmpeg

pip install -r requirements.txt

# 檢體轉繁體
pip install opencc-python-reimplemented

# 讓server_wss.py成為統一與客戶端溝通的API
pip install httpx

```

### RUN the server

```bash
python server_wss.py
```


### other sensevoice settings
```python
# 可以在這邊更改 VAD 的 max_end_silence_time 參數，預設是300ms
model_vad = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    disable_pbar = True,
    max_end_silence_time=500,
    # speech_noise_thres=0.6,
    disable_update=True,
)
```