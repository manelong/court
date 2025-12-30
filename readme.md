# 环境配置指南

## 创建虚拟环境
```bash
conda create -n court python=3.8
conda activate court
```

## 安装 PyTorch 和相关依赖
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
```

## 安装项目依赖
```bash
cd get_court/RS
pip install -r requirements.txt
pip install -e .
```

## 配置环境变量
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 下载模型文件
```bash
huggingface-cli download OAOA/InvSR noise_predictor_sd_turbo_v5.pth --local-dir weights
huggingface-cli download stabilityai/sd-turbo --local-dir stabilityai/sd-turbo
```

## 安装其他工具
```bash
pip install ultralytics
```