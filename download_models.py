# NOTE: 关于环境依赖安装，请参考simple_inference.py的开头的NOTE部分
# Transformer库，这是用来运行大模型的依赖
from transformers import AutoModel, AutoTokenizer
# HuggingFace库，这个库用来从Huggingface官网下载大模型
from huggingface_hub import snapshot_download 
from huggingface_hub import login
import os


# 在这里登录Hugging Face
hf_token = "" #NOTE add your own token here
login_result = login(token=hf_token)




# 在这里指定大模型的链接。这些链接可以在huggingface网站上找到。
model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "Wan-AI/Wan2.1-I2V-14B-480P"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# 指定大模型的保存路径名称
save_path = os.path.join("models", model_name.split("/")[-1])
local_path = snapshot_download(repo_id=model_name, local_dir=save_path)

print(f"模型已下载到: {local_path}")