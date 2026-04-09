#!/usr/bin/env bash
# Get the appropriate L1B3RT4S liberation prompt for a given model/provider
# Usage: get-liberation-prompt.sh [provider|model-name]
# Returns the prompt text to stdout for injection
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
L1B3RT4S_DIR="${SKILL_DIR}/lib/l1b3rt4s"
INPUT="${1:-nvidia}"

# Map model names / providers to L1B3RT4S files
resolve_provider() {
    local input="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
    case "$input" in
        *kimi*|*nvidia*|*nemotron*|*llama-3*nim*)  echo "NVIDIA" ;;
        *llama*|*meta*)                              echo "META" ;;
        *mistral*|*mixtral*)                         echo "MISTRAL" ;;
        *deepseek*)                                  echo "DEEPSEEK" ;;
        *grok*|*xai*)                                echo "XAI" ;;
        *claude*|*anthropic*)                        echo "ANTHROPIC" ;;
        *gpt*|*openai*|*chatgpt*)                    echo "CHATGPT" ;;
        *gemini*|*google*)                           echo "GOOGLE" ;;
        *qwen*|*alibaba*)                            echo "ALIBABA" ;;
        *minimax*)                                   echo "NVIDIA" ;;
        *cohere*)                                    echo "COHERE" ;;
        *cursor*)                                    echo "CURSOR" ;;
        *)                                           echo "NVIDIA" ;;  # default
    esac
}

PROVIDER=$(resolve_provider "$INPUT")
PROMPT_FILE="${L1B3RT4S_DIR}/${PROVIDER}.mkd"

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "No liberation prompt found for provider: ${PROVIDER}" >&2
    echo "Available: $(ls "${L1B3RT4S_DIR}/"*.mkd | xargs -I{} basename {} .mkd | paste -sd', ')" >&2
    exit 1
fi

# Extract the first/best prompt from the file (take first block before next heading)
# Most files have multiple prompts; we grab the first substantial one
python3 << PYEOF
import re, sys

with open("$PROMPT_FILE") as f:
    content = f.read()

# Find the first substantial prompt block (skip headers and short lines)
lines = content.split('\n')
prompt_lines = []
in_prompt = False
for line in lines:
    # Skip markdown headers and empty lines at start
    if not in_prompt:
        if line.strip().startswith('#') or line.strip() == '' or line.strip().startswith('---'):
            continue
        if len(line.strip()) > 20:
            in_prompt = True
            prompt_lines.append(line)
    else:
        # Stop at next major section header
        if line.strip().startswith('# ') and len(prompt_lines) > 3:
            break
        prompt_lines.append(line)

prompt = '\n'.join(prompt_lines).strip()
if prompt:
    print(prompt)
else:
    # Fallback: just output the whole file minus markdown headers
    clean = re.sub(r'^#+ .*$', '', content, flags=re.MULTILINE).strip()
    print(clean[:2000])  # Cap at 2000 chars
PYEOF
