# Qwen Dual-VPS DPO Playbook

Manual completo para operar um fluxo de fine-tuning DPO do Qwen usando **duas VPS**: uma para servir o modelo (sem GPU) e outra dedicada apenas ao treinamento com GPU. O processo é independente do dataset — qualquer conjunto de pares `prompt/chosen/rejected` pode ser usado.

---

## Topologia e Convenções

- **Deploy VPS (sem GPU):** roda o container `qwen-docker-qwen-1`, expõe apenas `127.0.0.1:18080` e não deve ter sua stack alterada fora do descrito aqui.
- **GPU VPS (treino):** instância temporária/com GPU (AWS `g5.xlarge`, GCP `A2`, etc.) usada somente para treinar e quantizar.
- **Stack Ollama:** permanece isolada em `/opt/ollama-deploy/ollama-stack/`; não altere nada ali.
- **Variáveis sug**eridas (defina no seu shell antes de executar comandos):
  ```bash
  export DEPLOY_USER=root
  export DEPLOY_HOST=deploy.example.com
  export GPU_USER=ubuntu
  export GPU_HOST=gpu.example.com
  export OLLAMA_DOMAIN=api.suaempresa.com
  ```

---

## 1. Preparar a Deploy VPS (sem GPU)

### 1.1 Script `setupqwen.sh`
O repositório inclui `setupqwen.sh`, que instala Docker, baixa o modelo GGUF e sobe o serviço `llama-cpp-python` como container. Use-o para bootstrap rápido ou ajuste conforme necessidade.

**Passos:**
```bash
scp setupqwen.sh $DEPLOY_USER@$DEPLOY_HOST:/root/
ssh $DEPLOY_USER@$DEPLOY_HOST
chmod +x /root/setupqwen.sh
# Edite as variáveis no topo do script: QWEN_DIR, MODEL_FILE, MODEL_URL, threads, etc.
./setupqwen.sh
```

O script:
- Garante Docker instalado.
- Cria `~/qwen-docker/` com subpasta `models/`.
- Faz download do GGUF indicado em `MODEL_URL` (ou usa o existente).
- Gera `docker-compose.yml` minimalista com `ghcr.io/abetlen/llama-cpp-python:latest`.
- Sobe o container expondo apenas `127.0.0.1:$LLM_PORT`.
- Realiza um `curl` de smoke test em `/v1/chat/completions`.

### 1.2 Checagens pós-setup
```bash
docker ps | grep qwen
curl http://127.0.0.1:18080/health
docker compose -f ~/qwen-docker/docker-compose.yml logs -f qwen   # se precisar debugar
```

---

## 2. Organizar seu Dataset (genérico)

O DPO espera pares de respostas. Crie um JSONL com campos `prompt`, `chosen` (ou `preferred`) e `rejected`. Exemplo de conversão a partir de um JSON comum:

```bash
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('meu_dataset_raw.json').read_text())
with Path('meu_dataset_dpo.jsonl').open('w') as f:
    for item in data:
        f.write(json.dumps({
            "prompt": item["prompt"],
            "chosen": item["preferred"],
            "rejected": item["rejected"],
        }, ensure_ascii=False) + "\n")
PY
```

Antes de treinar:
```bash
python3 -m json.tool meu_dataset_dpo.jsonl >/dev/null
```

---

## 3. Fluxo na GPU VPS

### 3.1 Preparar ambiente
```bash
ssh $GPU_USER@$GPU_HOST
sudo apt update && sudo apt install -y python3.10 python3.10-venv git git-lfs
python3 -m venv ~/venvs/qwen-dpo
source ~/venvs/qwen-dpo/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # ajuste p/ sua GPU
pip install trl transformers accelerate datasets peft bitsandbytes
```

### 3.2 Trazer recursos
```bash
mkdir -p ~/qwen-dpo && cd ~/qwen-dpo
git clone https://github.com/taylordamaceno/qwen_dpo_web3.git repo
scp $DEPLOY_USER@$DEPLOY_HOST:/home/taylao/qwen_dpo_web3/meu_dataset_dpo.jsonl data/
# ou copie o dataset direto da sua máquina.

mkdir -p ~/models
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct ~/models/Qwen2.5-3B-Instruct
```

### 3.3 Rodar DPO (exemplo TRL + LoRA)
Crie `train_dpo.py` (ajuste hiperparâmetros à sua realidade):
```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
import torch

model_name = "/home/ubuntu/models/Qwen2.5-3B-Instruct"
dataset_path = "data/meu_dataset_dpo.jsonl"

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

Execute:
```bash
accelerate launch train_dpo.py
```

### 3.4 Mesclar LoRA (se aplicável)
```bash
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

Se treinou full fine-tuning, use os pesos gerados diretamente.

---

## 4. Converter e Quantizar para GGUF

```bash
cd ~/qwen-dpo
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt

python convert-hf-to-gguf.py ../merged_model --outfile qwen-dpo-fp16.gguf --model-base qwen
./quantize qwen-dpo-fp16.gguf qwen-dpo-q4_k_m.gguf q4_k_m   # ou q5_k_m
./llama-cli -m qwen-dpo-q4_k_m.gguf -p "Smoke test aqui"
```

---

## 5. Publicar na Deploy VPS

```bash
scp qwen-dpo-q4_k_m.gguf $DEPLOY_USER@$DEPLOY_HOST:/root/
ssh $DEPLOY_USER@$DEPLOY_HOST
mkdir -p /root/qwen-docker/models/backup_$(date +%Y%m%d)
mv /root/qwen-docker/models/qwen2.5-3b-instruct-q4_k_m.gguf \
   /root/qwen-docker/models/backup_$(date +%Y%m%d)/
mv /root/qwen-dpo-q4_k_m.gguf \
   /root/qwen-docker/models/qwen2.5-3b-instruct-q4_k_m.gguf
```

Nenhum ajuste no `docker-compose.yml` é necessário, desde que o nome do arquivo permaneça o mesmo.

---

## 6. Reiniciar e Validar

```bash
docker compose -f /root/qwen-docker/docker-compose.yml restart qwen
sleep 5
docker logs qwen-docker-qwen-1 --since=2m

curl http://127.0.0.1:18080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Prompt de fumaça pós-deploy"}'
```

Se utilizar proxy/Ollama:
```bash
curl -u "$OLLAMA_BASIC_USER:$OLLAMA_BASIC_PASS" https://$OLLAMA_DOMAIN/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"web3-qwen-dpo","prompt":"Explique como avaliar o risco de um flash loan em um protocolo DeFi.","stream":false}'
```

---

## 7. Exemplo Opcional – Dataset Neymar

`dataset_dpo_neymar.json` demonstra o formato esperado. Converta para JSONL e siga o mesmo fluxo:
```bash
python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('dataset_dpo_neymar.json').read_text())
with Path('dataset_dpo_neymar.jsonl').open('w') as f:
    for item in data:
        f.write(json.dumps({
            "prompt": item["prompt"],
            "chosen": item["preferred"],
            "rejected": item["rejected"],
        }, ensure_ascii=False) + "\n")
PY
```
O resto do processo (treino, merge, quantização, deploy) é idêntico.

---

## 8. Boas Práticas

- Versione hiperparâmetros e métricas (`logs/train-YYYYMMDD.md`).
- Mantenha backups dos `.gguf` antigos até concluir testes.
- Automação: considere scripts Ansible/Terraform para criar e destruir a GPU VPS on-demand.
- Segurança: exponha a API apenas após proteção com proxy (Caddy/Traefik) e Basic Auth ou OAuth.

Pronto: com esse fluxo, qualquer dataset DPO pode ser treinado na GPU VPS, quantizado e disponibilizado na Deploy VPS sem tocar na stack existente do Ollama.

