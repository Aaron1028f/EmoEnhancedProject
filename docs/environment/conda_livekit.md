### Conda Environment for LiveKit

```bash
conda create -n livekit python=3.10
conda activate livekit

conda install uv -c conda-forge


# 開啟uv環境
source /home/aaron/project/server/lk_exp/agent-starter-python/.venv/bin/activate

# the comments below are not needed, just for record

# pip install livekit
# pip install livekit-api
# pip install livekit-agents

# uv add -U "livekit-agents[openai,turn-detector,silero,cartesia,deepgram]"
# uv add -U livekit "livekit-agents[openai,turn-detector,silero,cartesia,deepgram]" livekit-plugins-noise-cancellation

# livekit rtc from source code (IMPORTANT FOR setting the volume of TTS)
wget https://github.com/livekit/python-sdks/archive/refs/tags/rtc-v1.0.13.tar.gz
tar -xvzf rtc-v1.0.13.tar.gz
# then put the file under the directory of the project


```