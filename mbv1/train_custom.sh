#!/bin/bash     

dataset="rastrigin"
expname="custom"

in_dim=1
hid_dim=10
out_dim=1
total_layers=$1

epochs=1500
train_batch=32
test_batch=64

total_seed=$2

for (( seed=0; seed<$total_seed; seed++))
do
python3 main_regression.py --custom_model --regression --seed $seed --dataset ${dataset} --save "split/saved_models/${expname}/finetune/" --lr 0.001  --epochs ${epochs} --input-dim ${in_dim} --hidden-dim ${hid_dim} --output-dim ${out_dim} --total-layers ${total_layers} --batch-size ${train_batch} --test-batch-size ${test_batch} 

    for i in {0..30}
    do
    loaddir="split/saved_models/${expname}/finetune/${dataset}_$i.pth.tar"
    splitdir="split/saved_models/${expname}/${dataset}_$i.pth.tar"
    bash compute_eigen_custom.sh $i ${dataset} ${loaddir} $seed $in_dim $hid_dim $out_dim $total_layers $train_batch $test_batch
    python3 index_eigen_custom.py $i 1 ${dataset} 
    python3 main_split_custom.py --regression --seed $seed --dataset ${dataset} --exp-name ${expname} --split-index $i --rd $i --load ${loaddir} --total-layers $total_layers --input-dim ${in_dim} --hidden-dim ${hid_dim} --output-dim ${out_dim} --batch-size ${train_batch} --test-batch-size ${test_batch} 
    python3 main_finetune_regression.py --custom_model --regression --seed $seed --rd $i --dataset ${dataset} --load ${splitdir} --total-layers $total_layers --input-dim ${in_dim} --hidden-dim ${hid_dim} --output-dim ${out_dim} --layer -1 --epochs 1500 --lr 0.001 --warm 0 --batch-size ${train_batch} --test-batch-size ${test_batch} 
    done
done