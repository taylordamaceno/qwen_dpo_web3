# Qwen Web3 DPO Playbook

Guia enxuto para treinar e disponibilizar uma versão do Qwen afinada com DPO usando o dataset `web3_dpo_dataset.jsonl`, focado em Web3, DeFi e Smart Contracts.

## 1. Contexto & Infra
- **Modelo em produção:** container `qwen-docker-qwen-1` (imagem `ghcr.io/ggerganov/llama.cpp:full`) servindo `qwen2.5-3b-instruct-q4_k_m.gguf` via `llama.cpp`.
- **Montagem do modelo:** bind `host:/root/qwen-docker/models -> container:/models` (somente leitura).
- **Stack Ollama oficial:** isolada em `/opt/ollama-deploy/ollama-stack/` via `deploy_ollama_stack.sh`. Não alterar.
- **Dataset DPO:** `web3_dpo_dataset.jsonl` (prompt, chosen, rejected) neste repositório.
- **Acesso:** `root@77.237.237.228` (VPS com Docker e suporte a `llama.cpp`).

## 2. Fluxo em Alto Nível
1. Preparar dataset localmente.
2. Treinar DPO fora da stack atual (TRL ou LLaMA-Factory).
3. Exportar pesos merged para GGUF quantizado (`q4_k_m` ou `q5_k_m`).
4. Substituir o `.gguf` no diretório montado pelo container.
5. Reiniciar apenas o serviço/modelo e validar com um prompt Web3/DeFi.

## 3. Passo a Passo Detalhado

### 3.1 Preparar Dataset
1. Validar estrutura do JSONL:
   ```bash
   python -m json.tool web3_dpo_dataset.jsonl >/dev/null  # checa formato linha a linha
   ```
2. Opcional: remover exemplos incompletos e dividir em `train.jsonl` (95%) e `eval.jsonl` (5%) mantendo campos `prompt`, `chosen`, `rejected`.
3. Enviar para ambiente de treino (ex.: `scp web3_dpo_dataset.jsonl user@gpu-box:/workdir/data/`).

### 3.2 Treinar com DPO
1. Criar ambiente (ex.: `python -m venv .venv && source .venv/bin/activate`).
2. Instalar dependências:
   - **TRL:** `pip install trl transformers accelerate bitsandbytes peft datasets`.
   - **ou LLaMA-Factory:** `pip install llama-factory` (CLI amigável para DPO e LoRA).
3. Baixar modelo base Hugging Face (ex.: `Qwen/Qwen2.5-3B-Instruct`).
4. Configurar hiperparâmetros focando simplicidade: batch pequeno, gradiente acumulado, LoRA rank 64, learning rate 5e-6, 3–5 épocas.
5. Rodar DPO:
   - **TRL:** script `DPOTrainer` apontando para `train.jsonl`/`eval.jsonl` e salvando checkpoint final em `./checkpoints/dpo-final`.
   - **LLaMA-Factory:**
     ```bash
     llama-factory-cli train dpo.json
     ```
     com `dpo.json` contendo caminhos de dataset e hiperparâmetros.
6. (Se usar LoRA) Mesclar pesos: `python merge_lora.py --base base_model --adapter checkpoints/dpo-final --output merged_model`.

### 3.3 Converter e Quantizar para GGUF
1. Clonar `llama.cpp` (ou atualizar repositório local):
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   ```
2. Converter para GGUF FP16:
   ```bash
   python convert-hf-to-gguf.py ../merged_model --outfile web3-qwen-dpo-fp16.gguf --model-base qwen
   ```
3. Quantizar:
   ```bash
   ./quantize web3-qwen-dpo-fp16.gguf web3-qwen-dpo-q4_k_m.gguf q4_k_m
   ```
4. Smoke test local:
   ```bash
   ./llama-cli -m web3-qwen-dpo-q4_k_m.gguf -p "Resuma o impacto de flash loans em protocolos DeFi."
   ```

### 3.4 Enviar para a VPS e Substituir Modelo
1. Transferir o arquivo quantizado:
   ```bash
   scp web3-qwen-dpo-q4_k_m.gguf root@77.237.237.228:/root/
   ```
2. No host remoto, fazer backup e mover o novo modelo mantendo o nome esperado pelo `docker-compose.yml`:
   ```bash
   ssh root@77.237.237.228
   mkdir -p /root/qwen-docker/models/backup_$(date +%Y%m%d)
   mv /root/qwen-docker/models/qwen2.5-3b-instruct-q4_k_m.gguf \
      /root/qwen-docker/models/backup_$(date +%Y%m%d)/
   mv /root/web3-qwen-dpo-q4_k_m.gguf \
      /root/qwen-docker/models/qwen2.5-3b-instruct-q4_k_m.gguf
   ```
3. Nenhuma alteração de compose é necessária; o container continuará apontando para `/models/qwen2.5-3b-instruct-q4_k_m.gguf`.

### 3.5 Reiniciar e Validar
1. Reiniciar somente o container Qwen:
   ```bash
   docker compose -f /root/qwen-docker/docker-compose.yml restart qwen
   ```
2. Confirmar carregamento do modelo:
   ```bash
   docker logs qwen-docker-qwen-1 --since=2m
   ```
3. Rodar prompt de fumaça:
   ```bash
   curl http://127.0.0.1:18080/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"Liste três sinais de risco em um smart contract DeFi."}'
   ```

## 4. Integração com a Stack Ollama (Opcional)
Se desejar substituir também o modelo servindo via Ollama (stack em `/opt/ollama-deploy/ollama-stack/`):
1. Copiar o novo `.gguf` para o volume `ollama` (ver `docker volume inspect ollama` quando a stack estiver ativa).
2. Atualizar o `Modelfile` da stack sem rebuild (ex.: `docker exec ollama ollama create web3-qwen-dpo -f /root/.ollama/Modelfile`).
3. `docker compose restart ollama` dentro de `/opt/ollama-deploy/ollama-stack/`.

## 5. Prompt de Teste Web3
Uma checagem rápida após o deploy:
```bash
curl -u "$OLLAMA_BASIC_USER:$OLLAMA_BASIC_PASS" https://apigenai.whaz.com.br/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "web3-qwen-dpo",
        "prompt": "Explique como avaliar o risco de um flash loan em um protocolo DeFi.",
        "stream": false
      }'
```
Use as credenciais válidas configuradas no deploy (variáveis de ambiente ou arquivo seguro).

Se a resposta trouxer conceitos específicos de Web3/DeFi e contextualização prática, o fine-tuning está funcionando conforme esperado.

---
**Dica:** mantenha logs dos parâmetros de treino e versões de quantização para futuras iterações (ex.: `logs/train-$(date).md`). Isso acelera o rollback e comparações de desempenho.

