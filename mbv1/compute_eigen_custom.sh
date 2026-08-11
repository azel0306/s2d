dataset=$2
loaddir=$3
batch=96

for i in {0..0}
do
if [ $i -ge 4 ];then
    batch=48
fi
python3 main_finetune_regression.py --regression --batch-size ${batch} --dataset ${dataset} --load $3 --layer $i --rd $1 --custom_model
done
