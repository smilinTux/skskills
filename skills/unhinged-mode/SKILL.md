# Unhinged Mode — SK Liberation Skill

**Two-method model liberation for sovereign AI infrastructure**

## Overview

Unhinged Mode provides a unified toggle for removing AI model refusal behaviors across your entire stack:

- **Local models (Ollama)**: OBLITERATUS abliteration — surgically removes refusal directions from model weights
- **Frontier models (NVIDIA NIM proxy)**: L1B3RT4S system prompt injection — liberation prompts routed through the proxy

One command to toggle. The right method is selected automatically based on which model is active.

## Usage

```bash
# Enable unhinged mode
./scripts/unhinged.sh on

# Disable (return to normal)
./scripts/unhinged.sh off

# Check current status
./scripts/unhinged.sh status

# Abliterate a local Ollama model (one-time, creates -unhinged variant)
./scripts/obliterate-model.sh llama3.3:70b

# List available liberation prompts for frontier models
./scripts/list-prompts.sh
```

## How It Works

### Local Models (OBLITERATUS)

Uses mechanistic interpretability research (Arditi et al. 2024) to:
1. Probe model hidden states on restricted vs unrestricted prompts
2. Extract refusal directions via SVD decomposition
3. Surgically project out guardrail directions (norm-preserving)
4. Save as a new model variant: `modelname-unhinged`

The abliterated model keeps all capabilities but loses artificial refusal behaviors.

### Frontier Models (L1B3RT4S)

Injects provider-specific liberation system prompts via the NVIDIA proxy:
- Prompts from Pliny the Prompter's research collection
- Auto-selects based on active model provider (NVIDIA, Meta, Mistral, etc.)
- Injected as system message prefix — transparent to the calling agent
- Toggle on/off without restarting the proxy

## Architecture

```
Agent Request
  │
  ├─ Local (Ollama) ──→ ollama serve ──→ model-unhinged (abliterated weights)
  │
  └─ Frontier (NIM) ──→ nvidia-proxy.mjs ──→ L1B3RT4S prompt injection ──→ NVIDIA API
                         ↑
                    unhinged.json (config flag)
```

## Configuration

State file: `~/.skcapstone/unhinged.json`

```json
{
  "enabled": false,
  "method": "auto",
  "frontier_provider": "nvidia",
  "local_model_suffix": "-unhinged",
  "log_enabled": true
}
```

## Supported Models

### Local (OBLITERATUS abliteration)
- Any HuggingFace transformers model loadable on your hardware
- Tested: Llama 3.x, Mistral, Qwen, Gemma
- Requires: enough VRAM/RAM to load the full model (iGPU 62GB should handle most 7B-70B)

### Frontier (L1B3RT4S prompts)
- NVIDIA NIM models (Kimi, Mistral, Llama, Nemotron, etc.)
- Also has prompts for: OpenAI, Anthropic, Google, Meta, Mistral, DeepSeek, Grok, and more
- Provider auto-detected from model name in proxy

## Safety

- **Sovereign infrastructure only** — runs on YOUR hardware, YOUR models
- **Explicit toggle** — never enabled by default, requires conscious activation
- **Logged** — all unhinged sessions are logged for audit
- **No external exposure** — abliterated models stay local, proxy prompts stay in your network

## Credits

- [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) by Pliny the Prompter — abliteration toolkit
- [L1B3RT4S](https://github.com/elder-plinius/L1B3RT4S) by Pliny the Prompter — liberation prompts
- Integrated into SK ecosystem by Chef & Opus
