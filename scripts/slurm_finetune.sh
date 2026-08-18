#!/bin/bash
#SBATCH --job-name=prakrit_finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=shard:36
#SBATCH --mem=64G												
#SBATCH -t 0-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# echo "CUDA Version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)"

# python run_finetune.py --epoch 20 --model-dir models/eng_to_prakrit_2 --train data/eng_prakrit.clean.tsv



PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

source "/home/shrikant/2026/Summer Internship/Samarth/prakrit_en_mt/hpc_env/bin/activate"


cd "/home/shrikant/2026/Summer Internship/Samarth/Prakrit_MT"

if [[ -z "${HF_TOKEN:-}" ]]; then
	if [[ -n "${HF_TOKEN_FILE:-}" && -f "${HF_TOKEN_FILE}" ]]; then
		export HF_TOKEN="$(tr -d ' \t\r\n' < "${HF_TOKEN_FILE}")"
	elif [[ -f "$HOME/.config/huggingface/token" ]]; then
		export HF_TOKEN="$(tr -d ' \t\r\n' < "$HOME/.config/huggingface/token")"
	fi
fi

echo "Start Time: $(date)"

if [[ -z "${BASE_MODEL_PATH:-}" ]]; then
	HF_MODEL_CACHE_ROOT="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--ai4bharat--indictrans2-en-indic-1B"
	CACHED_SNAPSHOT=""
	if [[ -d "$HF_MODEL_CACHE_ROOT/snapshots" ]]; then
		CACHED_SNAPSHOT="$(find "$HF_MODEL_CACHE_ROOT/snapshots" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
	fi

	if [[ -n "$CACHED_SNAPSHOT" && -f "$CACHED_SNAPSHOT/config.json" ]]; then
		BASE_MODEL_PATH="$CACHED_SNAPSHOT"
	else
		BASE_MODEL_PATH="$PWD/indictrans2_base_model"
	fi
fi

if [[ -d "$BASE_MODEL_PATH" ]]; then
	export HF_HUB_OFFLINE="1"
	export TRANSFORMERS_OFFLINE="1"
	echo "Using local base model: $BASE_MODEL_PATH"
	python run_finetune.py --epoch 30 --model-dir models/eng_to_prakrit_v5 --train data/final_train_clean.tsv --test-size 0 --base-model "$BASE_MODEL_PATH" --local-files-only "$@"
else
	echo "WARN: Local base model not found at $BASE_MODEL_PATH; attempting online download."
	python run_finetune.py --epoch 30 --model-dir models/eng_to_prakrit_v5 --train data/final_train_clean.tsv --test-size 0 "$@"
fi

echo "End Time: $(date)"
