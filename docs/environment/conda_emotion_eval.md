```bash
# 創建 Python 3.10 的環境
conda create --name emotion_eval python=3.10 -y

# 啟動環境
conda activate emotion_eval

# 首先，安裝 TensorFlow, OpenCV 等
conda install -c conda-forge tensorflow opencv pandas scikit-learn notebook -y

# 接著，安裝 PyTorch 的 GPU 版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install deepface py-feat


# 驗證 OpenCV
python -c "import cv2; print(f'OpenCV 版本: {cv2.__version__}')"

# 驗證 TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow 版本: {tf.__version__}'); print('GPU 可用:', tf.config.list_physical_devices('GPU'))"

# 驗證 PyTorch
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'GPU 可用: {torch.cuda.is_available()}')"

# 驗證 DeepFace
python -c "from deepface import DeepFace; print('DeepFace 已準備就緒!')"

# 驗證 py-feat
python -c "from feat import Detector; print('py-feat 已準備就緒!')"
```