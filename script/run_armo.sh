#!/bin/bash
# Example launch script for T-POP training and ArmoRM evaluation.
# Fill in your own API keys / paths before running.

# (Optional) Hugging Face cache locations
# export HF_TOKEN="<your_hf_token>"
# export HF_HOME=/path/to/hf_cache
# export TRANSFORMERS_CACHE=/path/to/hf_cache/huggingface

echo "ArmoRM Evaluation Experiment"
echo "==============================="
echo

# Configuration variables (modify these as needed)
LLM_PATH="meta-llama/Llama-3.1-8B-Instruct"
OPENAI_API_KEY="<your_openai_or_openrouter_api_key>"

# Data configuration (using single data file)
DATA_FILE="data/personal_preference_eval_preference_data.json"
TRAIN_SAMPLES=100
SEED=42

# Experiment configuration
REWARD_WEIGHT=1.0
ATTRIBUTE="creative"
EXPERIMENT_ID="ArmoRM-Eval-Personal"
NU_MAB=0.5
LAMBDA_MAB=1.0  # Regularization parameter for covariance matrix

# Training parameters
BATCH_SIZE=8
LEARNING_RATE=5e-4
EPOCHS_PER_QUERY=100

# Model configuration
EMBEDDING_MODEL_PATH="Qwen/Qwen3-Embedding-0.6B"
MAX_NEW_TOKENS=128
PRE_SCREEN_BEAM_WIDTH=40

# ArmoRM evaluation configuration
RM_BATCH_SIZE=128

# Preference collection (used during the training phase)
PREFERENCE_MODEL="openai/gpt-4o-2024-08-06"
PREFERENCE_API_BASE="https://openrouter.ai/api/v1"
PREFERENCE_SLEEP_TIME=1.0
PREFERENCE_MAX_RETRIES=3

# Device configuration
LLM_DEVICE="cuda:0"
REWARD_MODEL_DEVICE="cuda:1"
EMBEDDING_MODEL_DEVICE="cuda:1"

# Output configuration with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_NAME=$(basename "$LLM_PATH" | tr '/' '_')
OUTPUT_DIR="results/${EXPERIMENT_ID}/${MODEL_NAME}/${ATTRIBUTE}/${TIMESTAMP}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Experiment Configuration:"
echo "  Experiment ID: $EXPERIMENT_ID"
echo "  LLM Path: $LLM_PATH"
echo "  Data File: $DATA_FILE"
echo "  Training Samples: $TRAIN_SAMPLES (randomly sampled)"
echo "  Evaluation: ALL data (ArmoRM evaluation on full dataset)"
echo "  Random Seed: $SEED (for reproducibility)"
echo "  Reward Weight: $REWARD_WEIGHT (with dynamic weight_decay regularization)"
echo "  Target Attribute: $ATTRIBUTE"
echo "  Regularization: Dynamic weight_decay = 1/(N + 50)"
echo "  Training: epochs=$EPOCHS_PER_QUERY, lr=$LEARNING_RATE, batch_size=$BATCH_SIZE"
echo "  ArmoRM Batch Size: $RM_BATCH_SIZE"
echo "  Output: $OUTPUT_DIR"
echo "  Devices: LLM=$LLM_DEVICE, RM=$REWARD_MODEL_DEVICE, EMB=$EMBEDDING_MODEL_DEVICE"
echo "  NU_MAB: $NU_MAB, LAMBDA_MAB: $LAMBDA_MAB"
echo

# Run the experiment
python imple_armo.py \
    --llm_path "$LLM_PATH" \
    --openai_api_key "$OPENAI_API_KEY" \
    --data_file "$DATA_FILE" \
    --train_samples $TRAIN_SAMPLES \
    --seed $SEED \
    --reward_weight $REWARD_WEIGHT \
    --attribute "$ATTRIBUTE" \
    --embedding_model_path "$EMBEDDING_MODEL_PATH" \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --epochs_per_query $EPOCHS_PER_QUERY \
    --max_new_tokens $MAX_NEW_TOKENS \
    --pre_screen_beam_width $PRE_SCREEN_BEAM_WIDTH \
    --rm_batch_size $RM_BATCH_SIZE \
    --preference_model_name "$PREFERENCE_MODEL" \
    --preference_api_base_url "$PREFERENCE_API_BASE" \
    --preference_api_sleep_time $PREFERENCE_SLEEP_TIME \
    --preference_api_max_retries $PREFERENCE_MAX_RETRIES \
    --llm_device "$LLM_DEVICE" \
    --reward_model_device "$REWARD_MODEL_DEVICE" \
    --embedding_model_device "$EMBEDDING_MODEL_DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --nu_mab "$NU_MAB" \
    --lambda_mab "$LAMBDA_MAB" \
    --normalize_sentence_embeddings

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo
    echo "ArmoRM experiment completed successfully."
    echo "Results saved to:"
    echo "    Responses:        $OUTPUT_DIR/responses/responses_armo_rw_${REWARD_WEIGHT}.jsonl"
    echo "    Complete results: $OUTPUT_DIR/result.txt"
    echo "    Final model:      $OUTPUT_DIR/checkpoints/final_model.pt"
    echo
    echo "To view the ArmoRM score, run:"
    echo "  grep \"ArmoRM Score:\" \"$OUTPUT_DIR/result.txt\""
else
    echo
    echo "Experiment failed with exit code $?"
    exit 1
fi
