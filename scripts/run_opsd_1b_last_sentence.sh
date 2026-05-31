PER_DEVICE_BATCH=${1:-4}
GRAD_ACCUM=${2:-4}
MAX_STEPS=${3:-500}
SAVE_STEPS=${4:-25}

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 2 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --max_steps "$MAX_STEPS" \
    --output_dir  data0/siyanz/opsd/ \
    --run_config qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005_lastsent_topk1 \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps "$SAVE_STEPS" \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --token_selection_mode last_sentence \
    --token_selection_top_k 1 \
    --wandb_project OPSD
