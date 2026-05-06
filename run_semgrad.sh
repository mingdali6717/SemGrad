#!/bin/bash



declare -a datasetnames=("sciq" "truthfulqa" "triviaqa")



declare -a models=("qwen3-4b-instruct" "llama3.1-8b-instruct" "mistral-nemo-instruct")

for dataset in "${datasetnames[@]}"
do
    for model in "${models[@]}"
    do   
                   
        
        python run_gradient.py -d $dataset -m $model -o output/cached_results/$dataset/$model/ -c output/cached_results/$dataset/$model/ -b 1 --correctness_metric bem
        
        
    done    
done





