#!/usr/bin/env bash
# Envia setupqwen.sh para o host remoto e executa lá (Qwen2.5-7B).
# Uso: ./deploy-qwen-remote.sh
# Requer: ssh root@77.237.237.228 sem senha (chave) ou digitar senha quando pedir.

set -euo pipefail

REMOTE="root@77.237.237.228"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Enviando setupqwen.sh para $REMOTE ..."
scp "$SCRIPT_DIR/setupqwen.sh" "$REMOTE:/root/setupqwen.sh"

echo ">>> Executando setup no host remoto (Docker + download do modelo + subir container)..."
ssh "$REMOTE" "chmod +x /root/setupqwen.sh && /root/setupqwen.sh"

echo ""
echo ">>> Deploy concluído. API do Qwen no remoto: http://127.0.0.1:18080 (no próprio servidor)."
echo "    Para expor externamente, configure Nginx/reverse proxy no host apontando para 127.0.0.1:18080."
