export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

python train_text_to_image_lora.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --dataset_name $DATASET_NAME \
  --caption_column text \
  --resolution 512 \
  --random_flip \
  --train_batch_size 1 \
  --num_train_epochs 100 \
  --checkpointing_steps 5000 \
  --learning_rate 1e-04 \
  --lr_scheduler constant \
  --seed 42 \
  --output_dir "sd-pokemon-model-lora" \
  --validation_prompt "cute dragon creature" \
  --gaudi_config_name Habana/stable-diffusion \
  --throughput_warmup_steps 3

  #--mixed_precision fp16 \
  #--lr_warmup_steps 0 \
  #--gaudi_config_name Habana/stable-diffusion \
  #--bf16
