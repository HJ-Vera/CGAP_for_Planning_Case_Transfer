import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'

from huggingface_hub import snapshot_download

# 只下载核心文件，跳过ONNX和OpenVINO
model_dir = snapshot_download(
    repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    local_dir="./models/paraphrase-multilingual-MiniLM-L12-v2",
    ignore_patterns=["*.onnx", "openvino/*", "*.onnx.metadata"]
)
print(f"模型已下载至: {model_dir}")
