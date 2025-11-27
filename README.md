# Qwen DPO Fine-Tuning PoC

Proof of Concept for fine-tuning Qwen 2.5 3B Instruct with DPO (Direct Preference Optimization) focused on Web3, DeFi and Smart Contracts.

This is a small-scale PoC using a tiny dataset (16 examples). The goal is to validate the full pipeline from training to deployment.

## Infrastructure

| Component | Provider | Specs | Purpose |
|-----------|----------|-------|---------|
| Deploy VPS | Contabo (or similar cheap VPS) | 24GB RAM, vCPU | Serve the model via llama.cpp |
| GPU Instance | RunPod | RTX 2000 Ada | Train DPO and convert to GGUF |

## Files

| File | Description |
|------|-------------|
| `web3_dpo_dataset.jsonl` | DPO dataset with prompt/chosen/rejected pairs |
| `train_dpo.py` | Training script using TRL + LoRA |
| `setupqwen.sh` | Setup script for the deploy VPS |

## Complete Workflow

### Part 1: Setup Deploy VPS (Contabo)

1. SSH into your VPS and run the setup script:
```bash
chmod +x setupqwen.sh
./setupqwen.sh
```

This installs Docker, downloads the base Qwen model, and starts the llama.cpp container.

2. Verify the model is running:
```bash
curl http://127.0.0.1:18080/health
```

### Part 2: Train on GPU (RunPod)

1. Create a RunPod instance with GPU (RTX 2000 Ada or similar). Access via Web Terminal or Jupyter.

2. Install dependencies:
```bash
apt update && apt install -y git-lfs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install trl transformers accelerate datasets bitsandbytes peft protobuf==3.20 huggingface_hub
```

3. Clone the repo and download the base model:
```bash
mkdir -p /workspace/models && cd /workspace
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct models/Qwen2.5-3B-Instruct
cd models/Qwen2.5-3B-Instruct && git lfs pull && cd /workspace
git clone https://github.com/taylordamaceno/slm-dpo-fine-tuning.git repo
```

4. Validate the dataset:
```bash
python3 -m json.tool --json-lines /workspace/repo/web3_dpo_dataset.jsonl >/dev/null && echo "Dataset OK"
```

5. Download and run the training script:
```bash
wget -O /workspace/train_dpo.py "https://raw.githubusercontent.com/taylordamaceno/slm-dpo-fine-tuning/refs/heads/main/train_dpo.py"
cd /workspace && accelerate launch train_dpo.py
```

Training takes about 1 minute for this small dataset. You should see:
```
trainable params: 119,734,272 || all params: 3,205,672,960 || trainable%: 3.7351
...
Training complete
```

6. Merge LoRA weights with base model:
```bash
cat > /workspace/merge_lora.py << 'EOF'
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

print("Loading adapter...")
model = AutoPeftModelForCausalLM.from_pretrained(
    "/workspace/checkpoints/final",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print("Merging LoRA weights...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("/workspace/merged_model", safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained("/workspace/models/Qwen2.5-3B-Instruct")
tokenizer.save_pretrained("/workspace/merged_model")

print("Merge complete!")
EOF

python3 /workspace/merge_lora.py
```

7. Convert to GGUF and quantize:
```bash
cd /workspace
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
python3 convert_hf_to_gguf.py /workspace/merged_model --outfile /workspace/qwen-dpo-fp16.gguf --outtype f16

cmake -B build
cmake --build build --config Release -j
./build/bin/llama-quantize /workspace/qwen-dpo-fp16.gguf /workspace/qwen-dpo-q4_k_m.gguf q4_k_m
```

8. Verify the output:
```bash
ls -lh /workspace/*.gguf
```

You should see:
```
-rw-rw-rw- 1 root root 5.8G /workspace/qwen-dpo-fp16.gguf
-rw-rw-rw- 1 root root 1.8G /workspace/qwen-dpo-q4_k_m.gguf
```

9. Download `qwen-dpo-q4_k_m.gguf` via Jupyter Lab (right-click > Download).

10. Terminate the RunPod instance to stop billing.

### Part 3: Deploy to VPS

1. From your local machine, upload the model to the VPS:
```bash
scp qwen-dpo-q4_k_m.gguf root@YOUR_VPS_IP:/root/qwen-docker/models/qwen2.5-3b-instruct-q4_k_m.gguf
```

This directly replaces the original model file. The filename must match what's configured in `docker-compose.yml`.

2. Restart the container to load the new model:
```bash
ssh root@YOUR_VPS_IP "docker compose -f /root/qwen-docker/docker-compose.yml restart qwen"
```

Wait about 5-10 seconds for the model to load into memory.

3. Verify the container is running:
```bash
ssh root@YOUR_VPS_IP "docker logs qwen-docker-qwen-1 --tail 10"
```

You should see:
```
main: server is listening on http://127.0.0.1:18080 - starting the main loop
srv  update_slots: all slots are idle
```

### Part 4: Test the Fine-Tuned Model

**Important:** The model listens only on `127.0.0.1:18080` inside the VPS for security. You must SSH into the Contabo VPS first, then run the tests.

```bash
ssh root@YOUR_VPS_IP
```

Once inside the VPS, run the tests below. All questions are in **Portuguese** (the training dataset language):

**Test 1: Smart Contracts**
```bash
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "O que é um smart contract e como ele funciona?"}], "max_tokens": 300}'
```

**Test 2: Layer 1 vs Layer 2**
```bash
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Explique a diferença entre Layer 1 e Layer 2 em blockchain"}], "max_tokens": 300}'
```

**Test 3: DeFi Components**
```bash
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "O que é DeFi e quais são seus principais componentes?"}], "max_tokens": 300}'
```

**Test 4: DAO (Decentralized Autonomous Organization)**
```bash
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "Como funciona uma DAO (Organização Autônoma Descentralizada)?"}], "max_tokens": 300}'
```

**Test 5: Impermanent Loss**
```bash
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "messages": [{"role": "user", "content": "O que é impermanent loss em liquidity providing?"}], "max_tokens": 300}'
```

Expected output should mention:
- Price manipulation
- Leverage and margin trading attacks
- Double spending risks
- Smart contract vulnerabilities

**Test 2: Smart Contract Topic**
```bash
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Explain how a DAO works and its main components"}],
    "max_tokens": 200
  }' | python3 -m json.tool
```

**Test 3: Compare with Base Model (Optional)**

If you kept a backup of the original model, you can compare responses to see the improvement from DPO training. The fine-tuned model should provide more detailed, technically accurate answers about Web3/DeFi topics.

Expected output should mention:
- Price manipulation
- Leverage and margin trading attacks
- Double spending risks
- Smart contract vulnerabilities

**Test 2: Smart Contract Topic**
```bash
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Explain how a DAO works and its main components"}],
    "max_tokens": 200
  }' | python3 -m json.tool
```

**Test 3: Compare with Base Model (Optional)**

If you kept a backup of the original model, you can compare responses to see the improvement from DPO training. The fine-tuned model should provide more detailed, technically accurate answers about Web3/DeFi topics.

### Troubleshooting

If the model doesn't respond or gives errors:

1. **SSH into the VPS first** (all commands below run inside the Contabo VPS):
```bash
ssh root@YOUR_VPS_IP
```

2. Check container logs:
```bash
docker logs qwen-docker-qwen-1 --tail 50
```

3. Verify the model file size (should be ~1.8G):
```bash
ls -lh /root/qwen-docker/models/
```

4. Restart if needed:
```bash
docker compose -f /root/qwen-docker/docker-compose.yml restart qwen
```

5. Test basic health endpoint:
```bash
curl http://127.0.0.1:18080/health
```

## Cost Estimate

| Item | Cost |
|------|------|
| RunPod RTX 2000 Ada (~30 min) | ~$0.50 |
| Contabo VPS (monthly) | ~$10-15 |

Total PoC cost: less than $1 for GPU training.

## Notes

This is a PoC with a tiny dataset (16 examples). For production:
1. Use a larger, high-quality dataset (1000+ examples)
2. Train for more epochs
3. Evaluate model quality before deploying
4. Keep backups of the original model

## License
Taylor Damaceno
