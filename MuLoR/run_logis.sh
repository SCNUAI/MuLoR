export RECLOR_DIR=logiqas
export TASK_NAME=logiqas
export MODEL_NAME=roberta-model

CUDA_VISIBLE_DEVICES=3
export OUTPUT_NAME=logi-large-len384-bs2-acc1-lr4e6
python train.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 384 \
    --per_gpu_eval_batch_size 16   \
    --per_gpu_train_batch_size 16   \
    --gradient_accumulation_steps 3 \
    --learning_rate 5e-06 \
    --num_train_epochs 10.0 \
    --output_dir Checkpoints/$TASK_NAME/${OUTPUT_NAME} \
    --logging_steps 800 \
    --save_steps 200000 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --weight_decay 0.01
