#!/bin/bash
#SBATCH --job-name=LaMem_training_resnet50
#SBATCH --output=LaMem_training_resnet50.out
#SBATCH --error=LaMem_training_resnet50.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:t4:4
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job BEGIN, END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

# Define arrays for batch sizes, learning rates, and weight decays
batch_sizes=(128 256 512)
learning_rates=(0.01 0.001 0.0001 0.00001)
weight_decays=(0.0001 0.00001)

# Loop over each combination of batch size, learning rate, and weight decay
for batch_size in "${batch_sizes[@]}"; do
  for learning_rate in "${learning_rates[@]}"; do
    for weight_decay in "${weight_decays[@]}"; do
      echo "Running with batch_size=${batch_size}, learning_rate=${learning_rate}, weight_decay=${weight_decay}"
      python regression.py \
        --data.train_dataset=/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/transform_lamem_256_train.ffcv \
        --data.val_dataset=/home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/data/transform_lamem_256_val.ffcv \
        --data.in_memory=1 \
        --data.num_workers=30 \
        --dist.world_size=4 \
        --logging.folder=./resnet50_logs \
        --logging.log_level=1 \
        --lr.lr_schedule_type=steplr \
        --lr.lr_step_size=30 \
        --lr.lr_gamma=0.1 \
        --lr.lr_warmup_epochs=0 \
        --lr.lr_warmup_method=linear \
        --lr.lr_warmup_decay=0.01 \
        --lr.lr=${learning_rate} \
        --model.arch=resnet50 \
        --resolution.min_res=160 \
        --resolution.max_res=192 \
        --resolution.end_ramp=13 \
        --resolution.start_ramp=11 \
        --resolution.fix_res=0 \
        --training.task=reg \
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
