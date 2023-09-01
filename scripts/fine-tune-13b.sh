export WANDB_API_KEY=[redacted]
torchrun --nnodes=1 --nproc_per_node=8 llama_finetuning.py \
	--pure_bf16 \
	--enable_fsdp \
	--model_name /workspace/llama-2-13b-hf \
	--batch_size_training 6 \
	--micro_batch_size 1 \
	--num_epochs 3 \
	--dist_checkpoint_root_folder /workspace/ft-13b-checkpoints \
	--dist_checkpoint_folder /workspace/ft-13b \
	--dataset airoboros_dataset \
	--data_path ../instructions-fewer-trivia.jsonl \
	--output_dir /workspace/airoboros-l2-13b-2.1-ft \
	--deepspeed deepspeed.json \
	--report_to wandb \
	--gradient_checkpointing 1 \
	--use_fast_kernels True
