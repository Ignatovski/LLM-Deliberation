#!/usr/bin/env bash
set -euo pipefail

# Batch runner for Llama-3.3-70B-Instruct on all polynomial games.
# Uses OpenAI-compatible chat/completions (e.g., Groq/OpenRouter) and the
# per-game config_llama.txt without mutating config.txt.

GAME_DIRS=(
  "games_descriptions/polynomial_game_all_AI"
  "games_descriptions/polynomial_game_human"
  "games_descriptions/polynomial_game"
)
SEEDS=(-7 0 7)

# Load .env if present
ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "${ENV_FILE}" | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' | xargs -d '\n')
fi

OPENAI_API="${OPENAI_API:-${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}"

USE_AZURE=0
AZURE_API="${AZURE_OPENAI_API_KEY:-}"
AZURE_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-}"

OUTPUT_BASE="./polynomial/outputs/output_llama"
TEMP=1
SUFFIX=1
START=1
END=15
MAX_RETRIES=3

usage() {
  cat <<EOF
Usage:
  OPENAI_API=<key> [OPENAI_BASE_URL=<url>] bash run_both_llama.sh [options]

Options:
  --openai_api <key>         OpenAI-compatible API key (Groq/OpenRouter/etc.)
  --openai_base_url <url>    OpenAI-compatible base URL
  --azure                    Use Azure client (requires deployment names in config_llama.txt)
  --azure_api <key>          Azure OpenAI API key (overrides env)
  --azure_endpoint <url>     Azure OpenAI endpoint, e.g., https://your-resource.openai.azure.com/
  --help|-h                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --openai_api)
      OPENAI_API="$2"
      shift 2
      ;;
    --openai_base_url)
      OPENAI_BASE_URL="$2"
      shift 2
      ;;
    --azure)
      USE_AZURE=1
      shift 1
      ;;
    --azure_api)
      AZURE_API="$2"
      shift 2
      ;;
    --azure_endpoint)
      AZURE_ENDPOINT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${USE_AZURE}" -eq 1 ]]; then
  if [[ -z "${AZURE_API}" || -z "${AZURE_ENDPOINT}" ]]; then
    echo "Error: --azure requires --azure_api and --azure_endpoint (or AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT envs)." >&2
    exit 1
  fi
else
  if [[ -z "${OPENAI_API}" ]]; then
    echo "Error: OPENAI_API / OPENAI_API_KEY / OPENROUTER_API_KEY not provided." >&2
    usage
    exit 1
  fi
fi

if [[ "${USE_AZURE}" -eq 1 ]]; then
  export AZURE_OPENAI_API_KEY="${AZURE_API}"
  export AZURE_OPENAI_ENDPOINT="${AZURE_ENDPOINT}"
else
  export OPENAI_API_KEY="${OPENAI_API}"
  if [[ -n "${OPENAI_BASE_URL}" ]]; then
    export OPENAI_BASE_URL
  fi
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate llm_deli || true
fi

# Normalize output base to an absolute path so main_polynomial.py writes outside game_dir.
mkdir -p "${OUTPUT_BASE}"
OUTPUT_BASE_ABS="$(cd "${OUTPUT_BASE}" && pwd)"

run_game() {
  local game_dir="$1"
  (
    set -e
    local config_llama="${game_dir}/config_llama.txt"
    local initial="${game_dir}/initial_deal.txt"
    local initial_backup="${initial}.bak"
    local output_root="${OUTPUT_BASE_ABS}/$(basename "${game_dir}")"

    if [[ ! -f "${config_llama}" ]]; then
      echo "Missing ${config_llama}" >&2
      exit 1
    fi
    if [[ ! -f "${initial}" ]]; then
      echo "Missing ${initial}" >&2
      exit 1
    fi

    mkdir -p "${output_root}"
    cp "${initial}" "${initial_backup}"
    trap 'mv -f "${initial_backup}" "${initial}"' EXIT

    for seed in "${SEEDS[@]}"; do
      echo "[$(basename "${game_dir}")] Running Llama seed ${seed}"
      printf "<VALUE>%s</VALUE>\n" "${seed}" > "${initial}"
      local out_dir="${output_root}/poly_x${seed}"

      local config_basename
      config_basename="$(basename "${config_llama}")"

      python runs/run_batch_polynomial.py \
        --exp-prefix "" --suffix "${SUFFIX}" --start "${START}" --end "${END}" --max-retries "${MAX_RETRIES}" \
        -- --game_dir "${game_dir}" \
           --config_file "${config_basename}" \
           --output_dir "${out_dir}" \
           --temp "${TEMP}" \
           --reuse_faiss \
           --result "${seed}" \
           --min_answers 16 \
           $([[ "${USE_AZURE}" -eq 1 ]] && printf '%s' "--azure --azure_openai_api ${AZURE_API} --azure_openai_endpoint ${AZURE_ENDPOINT}") \
           $([[ "${USE_AZURE}" -eq 0 ]] && printf '%s' "--api_key ${OPENAI_API}")

      # Generate final_x plot for this seed
      python -m polynomial.tools.plot_final_x "${out_dir}" || true
    done
  )
}

for dir in "${GAME_DIRS[@]}"; do
  run_game "${dir}"
done

echo "Llama runs complete for seeds: ${SEEDS[*]}."
