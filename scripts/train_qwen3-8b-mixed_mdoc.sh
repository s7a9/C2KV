export PYTHONPATH=`pwd`/python:$PYTHONPATH
export OUTPUT_DIR=./checkpoints/qwen3-8b-mixed/
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=64 torchrun --nproc_per_node 8 -m train.train_mdoc \
    --num_train_epochs 1 \
    --warmup_steps 500 \
    --model_name_or_path Qwen/Qwen3-8B \
    --attn_impl flash_attention_2 \
    --padding_side right \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.1 \
    --enable_pic True \
    --pic_param qkv \
    --output_dir $OUTPUT_DIR/residual-qkv-pic \
    --logging_steps 1 \
    --deepspeed ./configs/ds_config.json \
    --do_train True \
    --eval_strategy steps \
    --eval_steps 100 \
    --only_train_pic True \
    --dataloader_num_workers 8 \
    --dataloader_prefetch_factor 32 \
    --train_data /mnt/nas1/duchuheng/datasets \
    --bf16 True \
    --save_strategy steps \
    --save_steps 2000 \
    --dataset_shuffle_seed 2948
    # --gradient_checkpointing True
    # --device_map auto \
