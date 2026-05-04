#!/usr/bin/env bash
set -euo pipefail

# Smoke-test runner for a single xyz polynomial game (one run, one seed).
# Expects a future multi-variable driver: main_polynomial_xyz.py

if [[ ! -f "main_polynomial_xyz.py" ]]; then
  echo "Error: main_polynomial_xyz.py not found. Add the xyz driver before running this script." >&2
  exit 1
fi

GAME_DIR="games_descriptions/polynomial_game_xyz"
EXP_NAME="${EXP_NAME:-xyz_single}"
SEED="${SEED:-0}"
TEMP="${TEMP:-1}"
ROUNDS="${ROUNDS:-16}"
MAX_STEP="${MAX_STEP:-2}"

# Load .env if present
ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "${ENV_FILE}" | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' | xargs -d '\n')
fi

python main_polynomial_xyz.py \
  --exp_name "${EXP_NAME}" \
  --game_dir "${GAME_DIR}" \
  --output_dir "./polynomial/outputs/output_xyz/${EXP_NAME}" \
  --temp "${TEMP}" \
  --rounds_num "${ROUNDS}" \
  --max_step "${MAX_STEP}" \
  --agents_num 4 \
  --result "${SEED}" \
  --reuse_faiss \
  --min_answers 16

python plot_final_x.py "./polynomial/outputs/output_xyz/${EXP_NAME}" || true
