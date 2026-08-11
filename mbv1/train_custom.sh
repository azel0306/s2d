dataset="rosenbrock"
expname="custom"


python3 main_regression.py --custom_model --regression --dataset ${dataset} --save "split/saved_models/${expname}/finetune/" --no-cuda --epochs 300


for i in {0..9}
do
loaddir="split/saved_models/${expname}/finetune/${dataset}_$i.pth.tar"
splitdir="split/saved_models/${expname}/${dataset}_$i.pth.tar"
bash compute_eigen_custom.sh $i ${dataset} ${loaddir}
python3 index_eigen_custom.py $i 1 ${dataset}
python3 main_split_custom.py --dataset ${dataset} --exp-name ${expname} --split-index $i --rd $i --load ${loaddir} 
python3 main_finetune_regression.py --custom_model --regression --rd $i --dataset ${dataset} --load ${splitdir} --layer -1 --epoch 160 --lr 0.1 --warm 0
done
