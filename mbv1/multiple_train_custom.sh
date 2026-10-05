#!/bin/bash     

total_layers=3
total_seeds=10

bash train_custom.sh $total_layers $total_seeds

total_layers=4
total_seeds=10

bash train_custom.sh $total_layers $total_seeds

total_layers=5
total_seeds=10

bash train_custom.sh $total_layers $total_seeds