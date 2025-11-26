# Qwen Dual-VPS DPO Playbook

This guide explains how to fine-tune a Qwen model with DPO when you control two VPS machines: one for serving the model without a GPU and another temporary GPU VPS for training and quantization. The same workflow works for any dataset that contains prompt, chosen, and rejected fields.

## Topology and Conventions
Deploy VPS hosts the container `qwen-docker-qwen-1`. It exposes only `127.0.0.1:18080` and should not be changed outside the steps in this guide.

GPU VPS is a temporary machine that has a GPU (for example AWS `g5.xlarge` or Google Cloud `A2`). Use it only for training and converting the model.

The Ollama stack lives at `/opt/ollama-deploy/ollama-stack/`. Leave it untouched.

Set the following environment variables before running the commands so the instructions stay generic:
```
export DEPLOY_USER=root
export DEPLOY_HOST=deploy.example.com
export GPU_USER=ubuntu
export GPU_HOST=gpu.example.com
export OLLAMA_DOMAIN=api.yourcompany.com
```

## 1. Prepare the Deploy VPS (no GPU)
1. Copy the helper script to the server:
```
scp setupqwen.sh $DEPLOY_USER@$DEPLOY_HOST:/root/
ssh $DEPLOY_USER@$DEPLOY_HOST
chmod +x /root/setupqwen.sh
```
2. Edit the variables at the top of `setupqwen.sh` if you need another directory, model name, or download URL. After that run:
```
./setupqwen.sh
```
3. Confirm that the container is running and healthy:
```
docker ps | grep qwen
curl http://127.0.0.1:18080/health
docker compose -f ~/qwen-docker/docker-compose.yml logs -f qwen
```
The script sets up Docker, downloads the GGUF file, creates `~/qwen-docker/`, generates a minimal `docker-compose.yml`, launches the `llama-cpp-python` container, and performs a smoke test.

## 2. Prepare Your Dataset
DPO needs pairs of responses. Convert your raw data to JSONL with the fields `prompt`, `chosen`, and `rejected`.
```
python3 - <<'PY'
import json
from pathlib import Path
source = Path('my_raw_dataset.json').read_text()
data = json.loads(source)
with Path('my_dpo_dataset.jsonl').open('w') as f:
    for item in data:
        f.write(json.dumps({
            "prompt": item["prompt"],
            "chosen": item["preferred"],
            "rejected": item["rejected"],
        }, ensure_ascii=False) + "\n")
PY
```
Always validate the file:
```
python3 -m json.tool my_dpo_dataset.jsonl >/dev/null
```

## 3. Workflow on the GPU VPS
1. Prepare the Python environment:
```
ssh $GPU_USER@$GPU_HOST
sudo apt update && sudo apt install -y python3.10 python3.10-venv git git-lfs
python3 -m venv ~/venvs/qwen-dpo
source ~/venvs/qwen-dpo/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install trl transformers accelerate datasets peft bitsandbytes
```
2. Clone the repository and copy the dataset:
```
mkdir -p ~/qwen-dpo && cd ~/qwen-dpo
git clone https://github.com/taylordamaceno/qwen_dpo_web3.git repo
scp $DEPLOY_USER@$DEPLOY_HOST:/home/taylao/qwen_dpo_web3/my_dpo_dataset.jsonl data/
mkdir -p ~/models
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct ~/models/Qwen2.5-3B-Instruct
```
3. Create `train_dpo.py` with this example (adjust hyperparameters as needed):
```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
import torch

model_name = "/home/ubuntu/models/Qwen2.5-3B-Instruct"
dataset_path = "data/my_dpo_dataset.jsonl"

ds = load_dataset("json", data_files=dataset_path, split="train")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb,
    device_map="auto"
)

config = DPOConfig(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    args=config,
    beta=0.1,
    train_dataset=ds
)

trainer.train()
trainer.save_model("./checkpoints/final")
tokenizer.save_pretrained("./checkpoints/final")
```
4. Launch the training:
```
accelerate launch train_dpo.py
```
5. Merge LoRA weights if you used adapters:
```
python3 - <<'PY'
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

base = "/home/ubuntu/models/Qwen2.5-3B-Instruct"
adapter = "./checkpoints/final"

model = AutoPeftModelForCausalLM.from_pretrained(adapter, device_map="auto")
model = model.merge_and_unload()
model.save_pretrained("./merged_model", safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
tokenizer.save_pretrained("./merged_model")
PY
```
If you performed full fine-tuning, use the produced checkpoints directly.

## 4. Convert and Quantize to GGUF
```
cd ~/qwen-dpo
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt

python convert-hf-to-gguf.py ../merged_model --outfile qwen-dpo-fp16.gguf --model-base qwen
./quantize qwen-dpo-fp16.gguf qwen-dpo-q4_k_m.gguf q4_k_m
./llama-cli -m qwen-dpo-q4_k_m.gguf -p "Smoke test here"
```

## 5. Publish on the Deploy VPS
```
scp qwen-dpo-q4_k_m.gguf $DEPLOY_USER@$DEPLOY_HOST:/root/
ssh $DEPLOY_USER@$DEPLOY_HOST
mkdir -p /root/qwen-docker/models/backup_$(date +%Y%m%d)
mv /root/qwen-docker/models/qwen2.5-3b-instruct-q4_k_m.gguf \
   /root/qwen-docker/models/backup_$(date +%Y%m%d)/
mv /root/qwen-dpo-q4_k_m.gguf \
   /root/qwen-docker/models/qwen2.5-3b-instruct-q4_k_m.gguf
```
Keep the file name so the existing `docker-compose.yml` keeps working without changes.

## 6. Restart and Validate
```
docker compose -f /root/qwen-docker/docker-compose.yml restart qwen
sleep 5
docker logs qwen-docker-qwen-1 --since=2m

curl http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Smoke test after deploy"}'
```
If you use a proxy or the Ollama stack, run a remote test too:
```
curl -u "$OLLAMA_BASIC_USER:$OLLAMA_BASIC_PASS" https://$OLLAMA_DOMAIN/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"web3-qwen-dpo","prompt":"Explain how to evaluate flash-loan risk in DeFi.","stream":false}'
```

## 7. Optional Example: Neymar Dataset
The file `dataset_dpo_neymar.json` shows the structure you need. Convert it to JSONL and reuse the same procedure:
```
python3 - <<'PY'
import json
from pathlib import Path
content = Path('dataset_dpo_neymar.json').read_text()
data = json.loads(content)
with Path('dataset_dpo_neymar.jsonl').open('w') as f:
    for item in data:
        f.write(json.dumps({
            "prompt": item["prompt"],
            "chosen": item["preferred"],
            "rejected": item["rejected"],
        }, ensure_ascii=False) + "\n")
PY
```

## 8. Good Practices
1. Save hyperparameters, checkpoints, and metrics in files such as `logs/train-YYYYMMDD.md`.
2. Keep old GGUF backups until the new model passes all tests.
3. Automate GPU VPS creation and destruction with tools like Ansible or Terraform if you repeat this workflow often.
4. Secure external endpoints with a proxy and authentication before exposing the model to users.

Following this flow, you can train any dataset on the GPU VPS, export the GGUF file, and swap it on the Deploy VPS without touching the existing Ollama stack.

