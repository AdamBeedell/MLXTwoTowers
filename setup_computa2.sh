cd ~/MLXTwoTowers
pyenv install 3.13.3
pip install --upgrade --break-system-packages -r requirements.txt
CUDA_TAG=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 | tr -d .); \
nvidia-smi && pip install --upgrade --index-url https://download.pytorch.org/whl/cu${CUDA_TAG} torch
mkdir data/
