#!/bin/bash


declare -a datasetnames=("sciq" "truthfulqa" "triviaqa")

declare -a models=("qwen3-4b-instruct" "llama3.1-8b-instruct" "mistral-nemo-instruct")


for dataset in "${datasetnames[@]}"
do
    for model in "${models[@]}"
    do
        
#   
        config_path="config/${dataset}_config.yaml"
        data_path="data/datasets/$dataset/test.jsonl"
        output_path="output/cached_results/$dataset"

        python run_generation.py -c $config_path -dp $data_path -o $output_path -m $model -b 10 -eb 512 -n 5
        
    done
done

