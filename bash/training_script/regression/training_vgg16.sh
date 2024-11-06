#!/bin/bash

# Define arrays for batch sizes, learning rates, and weight decays
batch_sizes=(256)
learning_rates=(0.01)
weight_decays=(0.0001)

# Loop over each combination of batch size, learning rate, and weight decay
for batch_size in "${batch_sizes[@]}"; do
  for learning_rate in "${learning_rates[@]}"; do
    for weight_decay in "${weight_decays[@]}"; do
      echo "Running with batch_size=${batch_size}, learning_rate=${learning_rate}, weight_decay=${weight_decay}"
      python training.py \
        --data.train_dataset=/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/imagenet_train_256.ffcv \
        --data.val_dataset=/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/imagenet_validation_256.ffcv \
        --data.in_memory=1 \
        --data.num_workers=10 \
        --dist.world_size=4 \
        --logging.folder=./vgg16_logs \
        --logging.model_ckpt_path=./vgg16_weights \
        --logging.log_level=1 \
        --lr.lr_schedule_type=steplr \
        --lr.lr_step_size=30 \
        --lr.lr_gamma=0.1 \
        --lr.lr_warmup_epochs=0 \
        --lr.lr_warmup_method=linear \
        --lr.lr_warmup_decay=0.01 \
        --lr.lr=${learning_rate} \
        --model.arch=vgg16 \
        --resolution.min_res=160 \
        --resolution.max_res=192 \
        --resolution.end_ramp=13 \
        --resolution.start_ramp=11 \
        --resolution.fix_res=0 \
        --training.task=clf \
        --training.eval_only=0 \
        --training.batch_size=${batch_size} \
        --training.optimizer=sgd \
        --training.momentum=0.9 \
        --training.weight_decay=${weight_decay} \
        --training.epochs=91 \
        --training.label_smoothing=0.1 \
        --training.distributed=1 \
        --training.use_blurpool=1 \
        --validation.batch_size=256 \
        --validation.resolution=256 \
        --validation.lr_tta=1
    done
  done
done