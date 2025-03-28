# NOTE: 环境依赖
# 1. 首先安装anaconda——提供python环境
# 2. 安装CUDA和CUDNN——为大模型提供显卡支持（本代码使用12.4）
# 3. 在anaconda环境中安装pytorch——这是一个深度神经网络的python库，大模型是基于pytorch实现的。
#    在anaconda环境中通过“pip install transformers”指令安装transformer库——这是大模型的依赖之一
#    在anaconda环境中通过“pip install huggingface_hub”指令安装huggingface_hub——用来下载大模型的库
#    在anaconda环境中通过“pip install 'accelerate>=0.26.0'”指令安装accelerate——大模型依赖之一
#    在anaconda环境中通过“pip install -U bitsandbytes”指令安装依赖
# 4. 下载并运行大模型，查看终端输出，是否缺少某些依赖。如果缺少依赖，通过pip安装缺少的依赖。


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList

class StreamStoppingCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        print(tokenizer.decode(input_ids[0, -1]), end="", flush=True)
        return False  # 不终止，直到 generate 结束

stopping_criteria = StoppingCriteriaList([StreamStoppingCriteria()])


# Set the path of the local model
# model_path = "DeepSeek-R1-Distill-Qwen-1.5B/snapshots/6a6f4aa4197940add57724a7707d069478df56b1"

model_path = "DeepSeek-R1-Distill-Qwen-1.5B"
# Quantization_config
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,  
#     bnb_4bit_compute_dtype="float16",  # 确保计算精度为 FP16
#     bnb_4bit_use_double_quant=True,  # 启用 double quant，提高压缩效率
#     bnb_4bit_quant_type="nf4"  # 采用 NF4（比普通 INT4 更稳定）
# )
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True  # 启用 8-bit 量化
)

# Set device to cuda 
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("CUDA is not available.")

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    quantization_config = quantization_config
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


# Run the inference
# prompt = "Please help me to count how many character 'e' in this sentence: 'I am trying to running the inferecnce with the large language model.'"
prompt = "请你为我解释一下注意力的概念，以及它在LLMs中的应用"
# prompt = "我现在在玩街头霸王2,我使用的角色是春丽。我现在卡关了，我卡在和隆Ryu的对战了，他一直使用‘波升’技巧（使用‘波动拳’迫使我起跳躲避，但是在我起跳后他就会紧接着使用‘升龙拳’对空攻击），这种战术让我苦不堪言。请你为我提供一些指导，我应该怎么击败隆Ryu？"
inputs = tokenizer(prompt, padding = True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

with torch.no_grad():
    output = model.generate(
        input_ids = input_ids, 
        attention_mask = attention_mask, 
        stopping_criteria = stopping_criteria,
        max_length=1000, 
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print the output
# print(f"inputs: {prompt}")
# print(f"Output: {output_text}")