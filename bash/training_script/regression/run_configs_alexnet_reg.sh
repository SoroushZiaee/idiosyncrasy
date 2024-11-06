#!/bin/bash
#SBATCH --job-name=training_alexnet
#SBATCH --output=training_alexnet.out
#SBATCH --error=training_alexnet.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job BEGIN, END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

srun /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/regression_alexnet.sh
