#!/bin/bash

algos=("FedAvg")
datasets=("UNDIR10")
jr="1"
client_activity_rate="0"
batch_size="64"
num_clients=100
num_classes=10
for algo in "${algos[@]}"; do
    for dataset in "${datasets[@]}"; do
        go_value="_nc${num_clients}"
        python main.py -algo "$algo" -jr "$jr" -data "$dataset" -go "$go_value" -nc "$num_clients" \
                       -nb "$num_classes" -car "$client_activity_rate" -lbs "$batch_size"
    done
done