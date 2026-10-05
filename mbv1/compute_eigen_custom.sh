#!/bin/bash  

dataset=$2
loaddir=$3
seed=$4
train_batch=$9
test_batch=${10}


for ((i=0; i<$8; i++))
do
# if [ $i -ge 4 ];then
#     batch=48
# fi
python3 main_finetune_regression.py --custom_model --regression --seed $seed --batch-size ${train_batch} --test-batch-size ${test_batch} --dataset ${dataset} --load $3 --total-layers $8 --input-dim $5 --hidden-dim $6 --output-dim $7 --layer $i --rd $1 --custom_model --epoch 1500
done
