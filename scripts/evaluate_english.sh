#!/bin/bash

source .venv_harness_en/bin/activate

# This script is used to evaluate
# triviaqa,gsm8k,openbookqa,hellaswag,xwinograd_en,squad2
# to evaluate with all testcases, set NUM_TESTCASE=None

MODEL_NAME_PATH=$1

GENERAL_TASK_NAME="triviaqa,gsm8k,openbookqa,hellaswag,xwinograd_en,squadv2"
GENERAL_NUM_FEWSHOT=4
GENERAL_NUM_TESTCASE="all"
GENERAL_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${GENERAL_NUM_FEWSHOT}shot_${GENERAL_NUM_TESTCASE}cases/general"

MMLU_TASK_NAME="mmlu"
MMLU_NUM_FEWSHOT=5
MMLU_NUM_TESTCASE="all"
MMLU_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${MMLU_NUM_FEWSHOT}shot_${MMLU_NUM_TESTCASE}cases/mmlu"

BBH_TASK_NAME="bbh_fewshot"
BBH_NUM_FEWSHOT=3
BBH_NUM_TESTCASE="all"
BBH_OUTDIR="results/${MODEL_NAME_PATH}/en/harness_en/alltasks_${BBH_NUM_FEWSHOT}shot_${BBH_NUM_TESTCASE}cases/bbh"

mkdir -p $GENERAL_OUTDIR
mkdir -p $MMLU_OUTDIR
mkdir -p $BBH_OUTDIR

cd lm-evaluation-harness-en

echo $MMLU_TASK_NAME
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME_PATH \
    --tasks $MMLU_TASK_NAME \
    --num_fewshot $MMLU_NUM_FEWSHOT \
    --batch_size auto \
    --max_batch_size 32 \
    --device cuda \
    --write_out \
    --output_path "../$MMLU_OUTDIR" \
    --use_cache "../$MMLU_OUTDIR" \
    --seed 42 \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME_PATH \
    --tasks $BBH_TASK_NAME \
    --num_fewshot $BBH_NUM_FEWSHOT \
    --batch_size auto \
    --max_batch_size 32 \
    --device cuda \
    --write_out \
    --output_path "../$BBH_OUTDIR" \
    --use_cache "../$BBH_OUTDIR" \
    --seed 42 \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME_PATH \
    --tasks $GENERAL_TASK_NAME \
    --num_fewshot $GENERAL_NUM_FEWSHOT \
    --batch_size auto \
    --max_batch_size 32 \
    --device cuda \
    --write_out \
    --output_path "../$GENERAL_OUTDIR" \
    --use_cache "../$GENERAL_OUTDIR" \
    --seed 42 \
