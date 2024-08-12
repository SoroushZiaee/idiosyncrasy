#!/bin/bash

#SBATCH --job-name=test_idio
#SBATCH --output=test_idio.out
#SBATCH --error=test_idio.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=cascade,v100
#SBATCH --mem=30G
    ##SBATCH --mail-type=BEGIN,END,FAIL # Send email on job END and FAIL
    ##SBATCH --mail-user=soroush1@yorku.ca

module --ignore_cache load gcc/12.3 scipy-stack
module list
source /home/soroush1/projects/def-kohitij/soroush1/idiosyncrasy/venv/bin/activate
# Store the output of the pip freeze | grep colorama command in a variable
colorama_version=$(pip freeze | grep colorama)

# Print the stored variable
echo "Colorama version: $colorama_version"

echo "env is activated."
# if pip freeze | grep -q termcolor; then
#     echo "termcolor is installed."
# else
#     echo "termcolor is not installed. Installing now..."
#     pip install termcolor
# fi


export CUDA_VISIBLE_DEVICES=0
arch=resnet50
loss=logCKA
hvm_classify=0
config="bento 94 1 0"

IFS=" " read -ra values <<< "${config}"


animals+=("${values[0]}")
neurons_animal+=("${values[1]}")
neurons_animal=${neurons_animal//+/" "}
mix_rate+=("${values[2]}")
seed+=("${values[3]}")

# Print all values
# for value in "${values[@]}"; do
#   echo "${value}"
# done

# Alternatively, print values using printf
# printf "%s\n" "${values[@]}"

IFS="_" read -r -a values <<< "${animals}"
for animal in "${values[@]}"; do
    fit_animals+=" ${animal}.right"
done
fit_animals="${fit_animals:1}"  # Remove leading space

echo "Animals [$animals] mix_rate [$mix_rate] fit_animals [$fit_animals] neurons_animal [$neurons_animal]"

srun python -m cka_reg.main -v \
--seed $seed \
--seed_select_neurons $seed \
--neural_loss $loss \
--arch $arch \
--epochs 10000 \
--step_size 6000 \
--save_path ${animals}_mix${mix_rate}_seed${seed}_v4 \
-nd manymonkeys \
-s All \
-na $neurons_animal \
--trials 36 \
-aei \
--loss_weights 1 1 ${hvm_classify} -mix_rate $mix_rate -causal 1 --val_every 30 \
--num_workers 7 \
--fit_animals $fit_animals