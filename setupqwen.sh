#!/usr/bin/env bash
set -euo pipefail

##########################
# >>> edite estas vars <<<
# Modelo: Qwen2.5-7B-Instruct (mais novo e free; ~4.4GB em q4_k_m)
##########################
QWEN_DIR="$HOME/qwen-docker"
# Q4_K_M do 7B é split em 2 shards; o llama.cpp carrega pelo primeiro.
MODEL_FILE="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
MODEL_URLS=(
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
)
LLM_THREADS="8"
LLM_CTX="4096"
LLM_PORT="18080"   # porta interna (somente localhost)
##########################

ok(){ echo -e "\033[1;32m[OK]\033[0m $*"; }
warn(){ echo -e "\033[1;33m[WARN]\033[0m $*"; }
err(){ echo -e "\033[1;31m[ERR]\033[0m $*" >&2; }

# 1) Docker (se faltar, instala)
if ! command -v docker >/dev/null 2>&1; then
  ok "Instalando Docker..."
  sudo apt-get update -y
  sudo apt-get install -y ca-certificates curl gnupg lsb-release
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo systemctl enable --now docker
else
  ok "Docker já instalado"
fi

# 2) Diretórios e modelo
mkdir -p "$QWEN_DIR/models"
missing=0
for url in "${MODEL_URLS[@]}"; do
  file_name="$(basename "$url")"
  if [ ! -f "$QWEN_DIR/models/$file_name" ]; then
    missing=1
    ok "Baixando shard GGUF: $file_name"
    curl -fL "$url" -o "$QWEN_DIR/models/$file_name"
  else
    ok "Shard já existe: $QWEN_DIR/models/$file_name"
  fi
done
if [ "$missing" -eq 0 ]; then
  ok "Todos os shards do modelo já estavam presentes"
fi

# 3) docker-compose para o servidor do LLM (exposto só em 127.0.0.1)
cat > "$QWEN_DIR/docker-compose.yml" <<'YAML'
version: "3.9"
services:
  qwen:
    image: ghcr.io/ggml-org/llama.cpp:server
    command: >
      -m /models/${MODEL_FILE}
      -c ${LLM_CTX}
      --host 127.0.0.1
      --port ${LLM_PORT}
      -t ${LLM_THREADS}
      -ngl 0
      --jinja
    environment:
      - MODEL_FILE=${MODEL_FILE}
      - LLM_CTX=${LLM_CTX}
      - LLM_THREADS=${LLM_THREADS}
      - LLM_PORT=${LLM_PORT}
    volumes:
      - ./models:/models:ro
    network_mode: "host"     # publica somente em 127.0.0.1:${LLM_PORT}
    restart: unless-stopped
YAML

# substitui variáveis no YAML
sed -i "s|\${MODEL_FILE}|$MODEL_FILE|g" "$QWEN_DIR/docker-compose.yml"
sed -i "s|\${LLM_CTX}|$LLM_CTX|g" "$QWEN_DIR/docker-compose.yml"
sed -i "s|\${LLM_THREADS}|$LLM_THREADS|g" "$QWEN_DIR/docker-compose.yml"
sed -i "s|\${LLM_PORT}|$LLM_PORT|g" "$QWEN_DIR/docker-compose.yml"

# 4) subir e testar
(cd "$QWEN_DIR" && sudo docker compose up -d)
sleep 2
ok "Servidor LLM subindo em 127.0.0.1:$LLM_PORT"

if curl -fsS "http://127.0.0.1:$LLM_PORT/health" >/dev/null; then
  ok "Health OK"
else
  warn "Health falhou. Veja logs: sudo docker compose -f $QWEN_DIR/docker-compose.yml logs -f"
fi

# teste rápido (sem Nginx, local)
curl -s "http://127.0.0.1:$LLM_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_FILE\",\"messages\":[{\"role\":\"user\",\"content\":\"Explique RAG em 3 bullets.\"}],\"max_tokens\":120}" \
  | head -c 400; echo
ok "Qwen rodando localmente. Próximo passo: colocar Nginx na frente."

