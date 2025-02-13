#!/bin/bash
#SBATCH --job-name=Imagenet_resnet50
#SBATCH --output=Imagenet_resnet50.out
#SBATCH --error=Imagenet_resnet50.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
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

echo "Running ResNet18"
python /home/soroush1/projects/def-kohitij/soroush1/training_fast_publish_faster/training_scripts_resnets.py