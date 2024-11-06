#!/bin/bash
#SBATCH --job-name=training_vgg19
#SBATCH --output=training_vgg19.out
#SBATCH --error=training_vgg19.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:30:00
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job BEGIN, END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

echo "Start Installing and setup env"
source /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/prepare_env/setup_env_node.sh
echo "Env has been set up"

module list

pip freeze

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

echo "Installing requirements"
pip install --no-index -r requirements.txt

echo "Env has been set up"

pip freeze

echo "Running vgg19"

srun /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/bash/training_script/classification/training_vgg19.sh
