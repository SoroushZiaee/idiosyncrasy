#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
arch=resnet50
loss=logCKA
hvm_classify=0
config="bento 94 1 0" # animal neurons mix_rate seed

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

python -m cka_reg.main -v \
--seed $seed \
--seed_select_neurons $seed \
--neural_loss $loss \
--arch $arch \
--epochs 2 \
--step_size 6000 \
--save_path ${animals}_mix${mix_rate}_seed${seed}_v4 \
-nd manymonkeys \
-s All \
-na $neurons_animal \
--trials 36 \
-aei \
--loss_weights 1 1 ${hvm_classify} -mix_rate $mix_rate -causal 0 --val_every 30 \
--num_workers 7 \
--fit_animals $fit_animals