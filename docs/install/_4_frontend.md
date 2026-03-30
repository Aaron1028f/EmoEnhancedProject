# Install Livekit for frontend and model communication

### 1. prepare the conda environment for livekit, and install uv
```bash
conda create -n livekit python=3.10
conda activate livekit

conda install uv -c conda-forge
```

### 2. Download the code of livekit python sdk

1. Do the following in livekit conda environment.

2. In `server/models/lk_exp`, install the livekit python sdk from https://github.com/livekit-examples/agent-starter-python and set up the uv environment as well.

3. Check if we can run the livekit successfully.

4. Then, we will have: `server/lk_exp/agent-starter-python`

5. Copy all the code in `server/lk_exp/agent-starter-python/src` (in this repo) to your local `server/lk_exp/agent-starter-python/src` to replace the original code, this is for the modified version of livekit agent for our project.

