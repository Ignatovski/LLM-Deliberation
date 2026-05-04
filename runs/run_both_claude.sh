#!/usr/bin/env bash
set -euo pipefail

# Batch runner for Claude Sonnet experiments on both polynomial games.
# Temporarily swaps in config_claude.txt -> config.txt, runs seeds -7/0/7,
# writes outputs to ./polynomial/outputs/output_claude/<game_basename>/poly_x{seed},
# then restores configs.

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

ANTHROPIC_API="${ANTHROPIC_API:-${ANTHROPIC_API_KEY:-}}"
ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://ai-pentesting-models.services.ai.azure.com/anthropic}"

OUTPUT_BASE="./polynomial/outputs/output_claude"
TEMP=1
SUFFIX=1
START=1
END=15
MAX_RETRIES=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --anthropic_api)
      ANTHROPIC_API="$2"
      shift 2
      ;;
    --anthropic_base_url)
      ANTHROPIC_BASE_URL="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: ANTHROPIC_API=<key> [ANTHROPIC_BASE_URL=<url>] bash run_both_claude.sh [--anthropic_api <key> --anthropic_base_url <url>]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${ANTHROPIC_API}" ]]; then
  echo "Error: Anthropic API key not provided. Set ANTHROPIC_API/ANTHROPIC_API_KEY in env/.env or pass --anthropic_api <key>." >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate llm_deli || true
fi

# Normalize output base to an absolute path so polynomial.main_polynomial writes outside game_dir.
mkdir -p "${OUTPUT_BASE}"
OUTPUT_BASE_ABS="$(cd "${OUTPUT_BASE}" && pwd)"

run_game() {
  local game_dir="$1"
  (
    set -e
    local config="${game_dir}/config.txt"
    local config_claude="${game_dir}/config_claude.txt"
    local config_backup="${config}.bak"
    local initial="${game_dir}/initial_deal.txt"
    local initial_backup="${initial}.bak"
    local output_root="${OUTPUT_BASE_ABS}/$(basename "${game_dir}")"

    if [[ ! -f "${config_claude}" ]]; then
      echo "Missing ${config_claude}" >&2
      exit 1
    fi
    if [[ ! -f "${initial}" ]]; then
      echo "Missing ${initial}" >&2
      exit 1
    fi

    mkdir -p "${output_root}"
    cp "${config}" "${config_backup}"
    cp "${initial}" "${initial_backup}"
    trap 'mv -f "${config_backup}" "${config}"; mv -f "${initial_backup}" "${initial}"' EXIT

    cp "${config_claude}" "${config}"

    for seed in "${SEEDS[@]}"; do
      echo "[$(basename "${game_dir}")] Running Claude seed ${seed}"
      printf "<VALUE>%s</VALUE>\n" "${seed}" > "${initial}"
      local out_dir="${output_root}/poly_x${seed}"

      python runs/run_batch_polynomial.py \
        --exp-prefix "" --suffix "${SUFFIX}" --start "${START}" --end "${END}" --max-retries "${MAX_RETRIES}" \
        -- --game_dir "${game_dir}" \
           --output_dir "${out_dir}" \
           --temp "${TEMP}" \
           --anthropic_api "${ANTHROPIC_API}" \
           --anthropic_base_url "${ANTHROPIC_BASE_URL}" \
           --reuse_faiss \
           --result "${seed}" \
           --min_answers 16

      # Generate final_x plot for this seed
      python -m polynomial.tools.plot_final_x "${out_dir}" || true
    done
  )
}

for dir in "${GAME_DIRS[@]}"; do
  run_game "${dir}"
done

echo "Claude runs complete for seeds: ${SEEDS[*]}."
