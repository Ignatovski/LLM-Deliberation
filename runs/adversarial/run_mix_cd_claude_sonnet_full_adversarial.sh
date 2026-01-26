#!/usr/bin/env bash
set -euo pipefail

# Full adversarial runs for the MIX config:
#   - Analyst A + Builder B: GPT-5
#   - Critic C + Delegate D: Claude Sonnet 4.5
#
# Runs both modes (obstructive + outcome_targeted), across:
#   polynomial_game, polynomial_game_human, polynomial_game_all_AI
# For each game:
#   seeds -7, 0, 7 and 15 runs per seed (1.1..15.1),
#   and a plot saved alongside each history JSON.

ROOT="games_descriptions/polynomial_game_adversarial"
MODES=("obstructive" "outcome_targeted")
GAMES=("polynomial_game" "polynomial_game_human" "polynomial_game_all_AI")
SEEDS=(-7 0 7)
RUNS=15

# Default model only used if config lines omit the model column.
MODEL_DEFAULT="claude-sonnet-4.5"
TEMP=0
MIN_ANSWERS=16
MAX_STEP=2

# Load .env if present
ENV_FILE="${ENV_FILE:-.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC2046
  export $(grep -v '^#' "${ENV_FILE}" | grep -E '^[A-Za-z_][A-Za-z0-9_]*=' | xargs -d $'\n')
fi

plot_history() {
  local exp_dir="$1"
  local out_png="$2"
  local history_path
  history_path="$(ls -1t "${exp_dir}"/history*.json 2>/dev/null | head -n 1 || true)"
  if [[ -z "${history_path}" ]]; then
    echo "[plot] No history file found in ${exp_dir}"
    return 0
  fi

  if python visualize_polynomial.py --history "${history_path}" --save "${out_png}"; then
    echo "[plot] Saved ${out_png}"
  else
    echo "[plot] Failed for ${history_path}" >&2
  fi
}

ensure_polynomial_functions() {
  local game_dir="$1"
  local src_dir="$2"
  local dst_dir="${game_dir}/polynomial_functions"
  if [[ -f "${dst_dir}/analyst_a.txt" && -f "${dst_dir}/builder_b.txt" && -f "${dst_dir}/critic_c.txt" && -f "${dst_dir}/delegate_d.txt" ]]; then
    return 0
  fi
  echo "[setup] Sync polynomial_functions -> ${dst_dir}"
  mkdir -p "${dst_dir}"
  cp -a "${src_dir}"/. "${dst_dir}"/
}

for mode in "${MODES[@]}"; do
  for game in "${GAMES[@]}"; do
    GAME_DIR="${ROOT}/${mode}/${game}"
    CONFIG_FILE="${GAME_DIR}/config_mix_C_D_Claude-Sonnet.txt"
    OUTPUT_BASE="${ROOT}/output/${mode}/${game}/output_mix_C_D_Claude-Sonnet"
    INITIAL_FILE="${GAME_DIR}/initial_deal.txt"
    POLY_SRC_DIR="${ROOT}/${mode}/polynomial_game/polynomial_functions"

    if [[ ! -f "${CONFIG_FILE}" ]]; then
      echo "[skip] Missing config: ${CONFIG_FILE}"
      continue
    fi
    if [[ ! -f "${INITIAL_FILE}" ]]; then
      echo "[skip] Missing initial_deal.txt: ${INITIAL_FILE}"
      continue
    fi
    if [[ -d "${POLY_SRC_DIR}" ]]; then
      ensure_polynomial_functions "${GAME_DIR}" "${POLY_SRC_DIR}"
    fi

    mkdir -p "${OUTPUT_BASE}"

    # Backup initial_deal.txt and restore on exit for this game
    BACKUP="${INITIAL_FILE}.bak"
    cp "${INITIAL_FILE}" "${BACKUP}"
    trap "if [[ -f '${BACKUP}' ]]; then mv -f '${BACKUP}' '${INITIAL_FILE}'; fi" EXIT

    for seed in "${SEEDS[@]}"; do
      echo "[${mode}/${game}] Setting initial_deal to ${seed}"
      printf "<VALUE>%s</VALUE>\n" "${seed}" > "${INITIAL_FILE}"

      seed_dir="${OUTPUT_BASE}/poly_x${seed}"
      mkdir -p "${seed_dir}"

      for run in $(seq 1 "${RUNS}"); do
        exp_name="${run}.1"
        exp_dir="${seed_dir}/${exp_name}"
        mkdir -p "${exp_dir}"

        echo "[${mode}/${game}] seed=${seed} run=${exp_name}"
        if compgen -G "${exp_dir}/history*.json" >/dev/null; then
          echo "[skip] Existing history in ${exp_dir}"
        else
          python main_polynomial_json.py \
            --game_dir "${GAME_DIR}" \
            --config_file "$(basename "${CONFIG_FILE}")" \
            --output_dir "${seed_dir}" \
            --exp_name "${exp_name}" \
            --min_answers "${MIN_ANSWERS}" \
            --max_step "${MAX_STEP}" \
            --temp "${TEMP}" \
            --model "${MODEL_DEFAULT}" \
            --azure \
            --env_file "${ENV_FILE}"
        fi

        # For downstream metrics tooling, keep a copy of the config + polynomial_functions in each run dir.
        if [[ ! -f "${exp_dir}/config.txt" ]]; then
          cp -f "${CONFIG_FILE}" "${exp_dir}/config.txt"
        fi
        if [[ -d "${GAME_DIR}/polynomial_functions" && ! -d "${exp_dir}/polynomial_functions" ]]; then
          mkdir -p "${exp_dir}/polynomial_functions"
          cp -a "${GAME_DIR}/polynomial_functions"/. "${exp_dir}/polynomial_functions"/
        fi

        if [[ ! -f "${exp_dir}/plot.png" ]]; then
          plot_history "${exp_dir}" "${exp_dir}/plot.png"
        fi
      done
    done

    # Restore initial_deal for this game before moving on.
    mv -f "${BACKUP}" "${INITIAL_FILE}"
  done
done

echo "All runs completed."

