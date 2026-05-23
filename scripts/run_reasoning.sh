#!/usr/bin/env bash
set -o pipefail

# =========================
# Basic configuration
# =========================

# Project root directory.
# Override it locally with:
# PROJECT_DIR=/path/to/project bash run_reasoning.sh
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Main extraction script.
SCRIPT="${PROJECT_DIR}/int-rule/get_pointwise_internal.py"

# Python executable.
# Override it locally with:
# PYTHON_BIN=/path/to/python bash run_reasoning.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

# Directory for saving output JSON files.
SAVE_DIR="${PROJECT_DIR}/results"

# Failure log file.
FAIL_LOG="${SAVE_DIR}/failed_runs_with_feedback.log"

# Optional directory for model symlinks.
# This avoids exposing the original checkpoint path in output filenames.
MODEL_LINK_DIR="${PROJECT_DIR}/model_links"

# Directory that contains local model checkpoints.
# Replace this locally when running experiments.
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/path/to/model/cache}"

# Directory that contains input datasets.
DATA_DIR="${PROJECT_DIR}/data/main"

mkdir -p "${SAVE_DIR}" "${MODEL_LINK_DIR}"
: > "${FAIL_LOG}"


# =========================
# Model list
# Format:
#   "model_display_name|model_checkpoint_path"
#
# model_display_name:
#   A short name used for logging and output organization.
#
# model_checkpoint_path:
#   Path to a local HuggingFace model directory.
#   It should either directly contain config.json, or contain snapshots/*/config.json.
# =========================

models=(
  "Model-Name-1|${MODEL_CACHE_DIR}/path-to-model-1"
  "Model-Name-2|${MODEL_CACHE_DIR}/path-to-model-2"
)


# =========================
# Dataset list
# Format:
#   "dataset_name|input_file"
#
# dataset_name:
#   A readable name used only for logging.
#
# input_file:
#   Path to the dataset file used by get_pointwise_internal.py.
# =========================

datasets=(
  "flask|${DATA_DIR}/flask.json"
  "helpsteer|${DATA_DIR}/helpsteer.json"
  "biggen|${DATA_DIR}/biggen.json"
  "mt_bench|${DATA_DIR}/mt_bench_digit.jsonl"
  "chatbot_arena|${DATA_DIR}/chatbot_arena_digit.jsonl"
)


# =========================
# Run settings
# =========================

POINTS=5
MAX_NEW_TOKENS=512
TEMPERATURE=0
DTYPE="bfloat16"

# Use smaller batch sizes for large models.
SMALL_MODEL_BATCH_SIZE=4
LARGE_MODEL_BATCH_SIZE=1

# Keywords used to detect large models.
LARGE_MODEL_KEYWORDS=("27B" "35B" "70B" "122B")

# GPU ids visible to the script.
# Override it locally with:
# CUDA_DEVICES=0,1,2,3 bash run_reasoning.sh
CUDA_DEVICES="${CUDA_DEVICES:-0}"


# =========================
# Main loop
# =========================

for item_model in "${models[@]}"; do
  model_name="${item_model%%|*}"
  model_root="${item_model#*|}"

  batch_size="${SMALL_MODEL_BATCH_SIZE}"
  for keyword in "${LARGE_MODEL_KEYWORDS[@]}"; do
    if [[ "${model_name}" == *"${keyword}"* ]]; then
      batch_size="${LARGE_MODEL_BATCH_SIZE}"
      break
    fi
  done

  echo "===================="
  echo "Running model: ${model_name}"
  echo "Model root: ${model_root}"
  echo "Batch size: ${batch_size}"
  echo "Setting: with_feedback"
  echo "===================="

  model_path="${model_root}"

  # Resolve HuggingFace cache layout:
  # model_root may contain either config.json or snapshots/<hash>/config.json.
  if [ ! -f "${model_path}/config.json" ]; then
    cfg=$(find "${model_path}/snapshots" -maxdepth 2 -name config.json 2>/dev/null | head -n 1 || true)
    if [ -n "${cfg}" ]; then
      model_path=$(dirname "${cfg}")
    fi
  fi

  if [ ! -f "${model_path}/config.json" ]; then
    echo "FAILED: model config not found for ${model_name}, root=${model_root}" | tee -a "${FAIL_LOG}"
    continue
  fi

  link_dir="${MODEL_LINK_DIR}/${model_name}"
  rm -rf "${link_dir}"
  ln -s "${model_path}" "${link_dir}"

  echo "Resolved model path: ${model_path}"
  echo "Model link: ${link_dir}"

  for item_data in "${datasets[@]}"; do
    dataset_name="${item_data%%|*}"
    input_file="${item_data#*|}"

    echo "Running dataset: ${dataset_name}"
    echo "Input file: ${input_file}"

    if [ ! -f "${input_file}" ]; then
      echo "FAILED: input file not found: ${input_file}" | tee -a "${FAIL_LOG}"
      continue
    fi

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" "${SCRIPT}" \
      --model_name_or_path "${link_dir}" \
      --save_dir "${SAVE_DIR}" \
      --points "${POINTS}" \
      --batch_size "${batch_size}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --with_feedback \
      --dtype "${DTYPE}" \
      --input_file "${input_file}"

    if [ $? -ne 0 ]; then
      echo "FAILED: model=${model_name}, dataset=${dataset_name}" | tee -a "${FAIL_LOG}"
    else
      echo "SUCCESS: model=${model_name}, dataset=${dataset_name}"
    fi
  done
done