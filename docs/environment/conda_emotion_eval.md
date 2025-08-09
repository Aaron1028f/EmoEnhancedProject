## deepface
```bash
# create this conda environment
conda create --name deepface python=3.10 -y
conda activate deepface
pip install deepface
pip install tf-keras
pip install matplotlib
```

## py-feat
```bash
conda create --name pyfeat python=3.7 -y
conda activate pyfeat

pip install py-feat

pip install scipy==1.13 # if 'scipy' error happens
```


```bash
# to delete conda environment
conda deactivate
conda env remove --name pyfeat
```