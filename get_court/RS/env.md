pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
cd get_court/RS
pip install -r requirements.txt
pip install -e .
export HF_ENDPOINT=https://hf-mirror.com