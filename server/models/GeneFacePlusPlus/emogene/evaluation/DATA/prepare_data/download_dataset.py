import kagglehub

kagglehub.login()

# Download latest version
path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")

print("Path to dataset files:", path)

# https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/
