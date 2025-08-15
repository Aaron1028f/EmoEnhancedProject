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

```