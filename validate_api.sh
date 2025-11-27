#!/bin/bash
# Script de Validação via API (para modelo rodando na VPS)
# Execute este script dentro da VPS onde o modelo está rodando

echo "=========================================="
echo "🔬 VALIDAÇÃO DO MODELO VIA API"
echo "=========================================="
echo ""

# URL da API (ajuste se necessário)
API_URL="http://127.0.0.1:18080/v1/chat/completions"

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para fazer pergunta
ask_question() {
    local question="$1"
    local test_name="$2"
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}📝 ${test_name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "❓ Pergunta: ${question}"
    echo ""
    echo -e "${GREEN}🤖 Resposta do modelo:${NC}"
    echo ""
    
    # Faz request e extrai apenas o conteúdo da resposta
    curl -s "$API_URL" \
        -H 'Content-Type: application/json' \
        -d "{
            \"model\": \"qwen\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$question\"}],
            \"max_tokens\": 400,
            \"temperature\": 0.7
        }" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    content = data['choices'][0]['message']['content']
    print(content)
    print()
    print(f'📏 Tamanho: {len(content.split())} palavras | {len(content)} caracteres')
except Exception as e:
    print(f'Erro ao processar resposta: {e}')
"
    echo ""
    echo ""
}

echo "Este script testa o modelo fine-tunado via API."
echo "Ele faz perguntas do dataset e perguntas novas para validar o aprendizado."
echo ""
echo "Pressione ENTER para começar..."
read

# ==================== TESTE 1: PERGUNTAS DO DATASET ====================
echo ""
echo "=========================================="
echo "🧠 TESTE 1: PERGUNTAS DO DATASET"
echo "=========================================="
echo "Estas perguntas estão no dataset de treino."
echo "O modelo deve dar respostas DETALHADAS e TÉCNICAS."
echo ""

ask_question "O que é um smart contract e como ele funciona?" "Teste 1.1 - Smart Contracts"

ask_question "Explique a diferença entre Layer 1 e Layer 2 em blockchain" "Teste 1.2 - Layer 1 vs Layer 2"

ask_question "O que é DeFi e quais são seus principais componentes?" "Teste 1.3 - DeFi"

# ==================== TESTE 2: PERGUNTAS NOVAS (GENERALIZAÇÃO) ====================
echo ""
echo "=========================================="
echo "🔄 TESTE 2: GENERALIZAÇÃO"
echo "=========================================="
echo "Estas perguntas NÃO estão no dataset, mas são relacionadas."
echo "O modelo deve usar o conhecimento aprendido."
echo ""

ask_question "Por que smart contracts são importantes para aplicações Web3?" "Teste 2.1 - Importância de Smart Contracts"

ask_question "Quais são as vantagens de usar Layer 2 ao invés de Layer 1?" "Teste 2.2 - Vantagens de L2"

ask_question "Como DEXs funcionam sem uma empresa central?" "Teste 2.3 - DEXs Descentralizados"

# ==================== TESTE 3: CONTROLE NEGATIVO ====================
echo ""
echo "=========================================="
echo "🌍 TESTE 3: CONTROLE NEGATIVO"
echo "=========================================="
echo "Pergunta fora do domínio de Web3."
echo "O modelo deve responder normalmente (conhecimento geral preservado)."
echo ""

ask_question "Como funciona a fotossíntese nas plantas?" "Teste 3.1 - Fora do Domínio"

# ==================== ANÁLISE ====================
echo ""
echo "=========================================="
echo "📊 ANÁLISE DOS RESULTADOS"
echo "=========================================="
echo ""
echo "✅ O modelo APRENDEU se você observou:"
echo ""
echo "  1. TESTE 1 (Dataset):"
echo "     ✓ Respostas com 70-100+ palavras"
echo "     ✓ Menciona tecnologias específicas (Solidity, Ethereum, Solana)"
echo "     ✓ Usa jargão técnico (ERC-20, PoS, AMM, DEX, TVL)"
echo "     ✓ Dá exemplos concretos (Uniswap, Aave, Compound)"
echo ""
echo "  2. TESTE 2 (Generalização):"
echo "     ✓ Respostas detalhadas (50+ palavras)"
echo "     ✓ Mantém o estilo técnico"
echo "     ✓ Usa vocabulário aprendido"
echo ""
echo "  3. TESTE 3 (Controle):"
echo "     ✓ Responde normalmente sobre outros assuntos"
echo "     ✓ Não tenta forçar Web3 em tudo"
echo ""
echo "⚠️  O modelo NÃO aprendeu bem se:"
echo ""
echo "  ✗ Respostas muito curtas (<30 palavras)"
echo "  ✗ Não menciona tecnologias específicas"
echo "  ✗ Respostas genéricas sem exemplos"
echo ""
echo "=========================================="
echo "✅ VALIDAÇÃO COMPLETA!"
echo "=========================================="
echo ""

