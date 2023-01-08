CONDA_ENV_NAME='dm_py38'

conda create -n $CONDA_ENV_NAME python=3.8
conda activate $CONDA_ENV_NAME

pip install -r requirements.txt

git clone https://huggingface.co/datasets/miracl/miracl