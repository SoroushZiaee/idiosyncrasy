#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter_t4.out
#SBATCH --error=jupyter_t4.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=10G
#SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
#SBATCH --mail-user=soroush1@yorku.ca

echo "Start Installing and setup env"
source /home/soroush1/projects/def-kohitij/soroush1/idiosyncrasy/scripts/prepare_env/setup_env.sh

module list

pip freeze

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip

echo "Installing requirements"
pip install --no-index -r /home/soroush1/projects/def-kohitij/soroush1/idiosyncrasy/requirements.txt

echo "Env has been set up"

pip freeze

srun /home/soroush1/projects/def-kohitij/soroush1/idiosyncrasy/scripts/notebook/lab.sh