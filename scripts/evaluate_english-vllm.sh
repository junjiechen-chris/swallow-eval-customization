#!/bin/bash

# This script is used to evaluate
# triviaqa,gsm8k,openbookqa,hellaswag,xwinograd_en,squad2
# to evaluate with all testcases, set NUM_TESTCASE=None

MODEL_NAME_PATH=$1
GPU_MEM_PROPORTION=$2
OUTPUT_DIR=${3:-results/${MODEL_NAME_PATH}}
TARGETED_TP_SIZE=$4
TARGETED_DP_SIZE=$5
#echo "GPU CONFIG:" $GPU_MEM_PROPORTION, $TARGETED_DP_SIZE, $TARGETED_TP_SIZE
#exit

# Convert OUTPUT_DIR to an absolute path
OUTPUT_DIR=$(realpath $OUTPUT_DIR)

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# TODO: force NUM_GPU to 1 since it's a shared machine 
NUM_GPUS=1
echo "ASSIGNED GPU: COUNT $NUM_GPUS; PROPORTION $GPU_MEM_PROPORTION" 

MODEL_ARGS="--model_args pretrained=$MODEL_NAME_PATH,tensor_parallel_size=$NUM_GPUS,dtype=auto,gpu_memory_utilization=$GPU_MEM_PROPORTION,data_parallel_size=$TARGETED_DP_SIZE"

GENERAL_LABEL="General"
GENERAL_TASK_NAME="triviaqa,gsm8k,openbookqa,hellaswag,xwinograd_en,squadv2"
GENERAL_NUM_FEWSHOT=4
GENERAL_NUM_TESTCASE="all"
GENERAL_BATCH_SIZE=16
GENERAL_OUTDIR="${OUTPUT_DIR}/en/harness_en/alltasks_${GENERAL_NUM_FEWSHOT}shot_${GENERAL_NUM_TESTCASE}cases/general"

MMLU_LABEL="MMLU"
MMLU_TASK_NAME="mmlu"
MMLU_NUM_FEWSHOT=5
MMLU_NUM_TESTCASE="all"
MMLU_BATCH_SIZE=16
MMLU_OUTDIR="${OUTPUT_DIR}/en/harness_en/alltasks_${MMLU_NUM_FEWSHOT}shot_${MMLU_NUM_TESTCASE}cases/mmlu"

BBH_LABEL="BBH"
BBH_TASK_NAME="bbh_cot_fewshot"
BBH_NUM_FEWSHOT=3
BBH_NUM_TESTCASE="all"
BBH_BATCH_SIZE=16
BBH_OUTDIR="${OUTPUT_DIR}/en/harness_en/alltasks_${BBH_NUM_FEWSHOT}shot_${BBH_NUM_TESTCASE}cases/bbh_cot"

GPQA_LABEL="GPQA"
GPQA_NUM_FEWSHOT=0
GPQA_NUM_TESTCASE="all"
GPQA_BATCH_SIZE=16
GPQA_TASK_NAME="gpqa_main_cot_zeroshot_meta_llama3_wo_chat"
GPQA_OUTDIR="${OUTPUT_DIR}/en/harness_en/alltasks_${GPQA_NUM_FEWSHOT}shot_${GPQA_NUM_TESTCASE}cases/gpqa_main_cot_zeroshot_meta_llama3_wo_chat"

MATH_LABEL="MATH"
MATH_NUM_FEWSHOT=4
MATH_NUM_TESTCASE="all"
MATH_BATCH_SIZE=16
MATH_TASK_NAME="math_500"
MATH_OUTDIR="${OUTPUT_DIR}/en/harness_en/alltasks_${MATH_NUM_FEWSHOT}shot_${MATH_NUM_TESTCASE}cases/${MATH_TASK_NAME}"


mkdir -p $GENERAL_OUTDIR
mkdir -p $MMLU_OUTDIR
mkdir -p $BBH_OUTDIR
mkdir -p $GPQA_OUTDIR
mkdir -p $MATH_OUTDIR

# TASK_NAME=($GENERAL_TASK_NAME $MMLU_TASK_NAME $BBH_TASK_NAME)
LABELS=($GENERAL_LABEL $MMLU_LABEL $BBH_LABEL $GPQA_LABEL $MATH_LABEL)
TASK_NAME=($GENERAL_TASK_NAME $MMLU_TASK_NAME $BBH_TASK_NAME $GPQA_TASK_NAME $MATH_TASK_NAME)
NUM_FEWSHOT=($GENERAL_NUM_FEWSHOT $MMLU_NUM_FEWSHOT $BBH_NUM_FEWSHOT $GPQA_NUM_FEWSHOT $MATH_NUM_FEWSHOT)
BATCH_SIZE=($GENERAL_BATCH_SIZE $MMLU_BATCH_SIZE $BBH_BATCH_SIZE $GPQA_BATCH_SIZE $MATH_BATCH_SIZE)
NUM_TESTCASE=($GENERAL_NUM_TESTCASE $MMLU_NUM_TESTCASE $BBH_NUM_TESTCASE $GPQA_NUM_TESTCASE $MATH_NUM_TESTCASE)
OUTDIRS=($GENERAL_OUTDIR $MMLU_OUTDIR $BBH_OUTDIR $GPQA_OUTDIR $MATH_OUTDIR)

pushd lm-evaluation-harness-en

# for i in "${!TASK_NAME[@]}"; do
for i in $(seq 1 2); do
    echo "Starting evaluation for: ${LABELS[$i]}"
    echo "Tasks: ${TASK_NAME[$i]}"
    echo "Output directory: ${OUTDIRS[$i]}"
    echo "Few-shot: ${NUM_FEWSHOT[$i]}, Testcase: ${NUM_TESTCASE[$i]}, Batch size: ${BATCH_SIZE[$i]}"
    
    lm_eval --model vllm \
        --model_args pretrained=$MODEL_NAME_PATH,tensor_parallel_size=$TARGETED_TP_SIZE,dtype=auto,gpu_memory_utilization=$GPU_MEM_PROPORTION,data_parallel_size=$TARGETED_DP_SIZE \
        --tasks ${TASK_NAME[$i]} \
        --num_fewshot ${NUM_FEWSHOT[$i]} \
        --batch_size ${BATCH_SIZE[$i]} \
        --device cuda \
        --write_out \
        --output_path "${OUTDIRS[$i]}" \
        --use_cache "${OUTDIRS[$i]}" \
        --log_samples \
        --seed 42
done
echo "All evaluations are done."
exit 0


echo $MMLU_TASK_NAME
lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,tensor_parallel_size=$TARGETED_TP_SIZE,dtype=auto,gpu_memory_utilization=$GPU_MEM_PROPORTION,data_parallel_size=$TARGETED_DP_SIZE \
    --tasks $MMLU_TASK_NAME \
    --num_fewshot $MMLU_NUM_FEWSHOT \
    --batch_size 16 \
    --device cuda \
    --write_out \
    --output_path "$MMLU_OUTDIR" \
    --use_cache "$MMLU_OUTDIR" \
    --seed 42 \

lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,tensor_parallel_size=$TARGETED_TP_SIZE,dtype=auto,gpu_memory_utilization=$GPU_MEM_PROPORTION,data_parallel_size=$TARGETED_DP_SIZE \
    --tasks $BBH_TASK_NAME \
    --num_fewshot $BBH_NUM_FEWSHOT \
    --batch_size 16 \
    --device cuda \
    --write_out \
    --output_path "$BBH_OUTDIR" \
    --use_cache "$BBH_OUTDIR" \
    --log_samples \
    --seed 42 \

lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME_PATH,tensor_parallel_size=$TARGETED_TP_SIZE,dtype=auto,gpu_memory_utilization=$GPU_MEM_PROPORTION,data_parallel_size=$TARGETED_DP_SIZE \
    --tasks $GENERAL_TASK_NAME \
    --num_fewshot $GENERAL_NUM_FEWSHOT \
    --batch_size 16 \
    --device cuda \
    --write_out \
    --output_path "$GENERAL_OUTDIR" \
    --use_cache "$GENERAL_OUTDIR" \
    --log_samples \
    --seed 42 \

# aggregate results
# cd ../
popd
python scripts/aggregate_result.py --model $MODEL_NAME_PATH --result-dir $OUTPUT_DIR
