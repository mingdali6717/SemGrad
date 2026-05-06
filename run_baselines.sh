#!/bin/bash



declare -a datasetnames=("sciq" "truthfulqa" "triviaqa")


declare -a models=("qwen3-4b-instruct" "llama3.1-8b-instruct" "mistral-nemo-instruct")


for dataset in "${datasetnames[@]}"
do
    for model in "${models[@]}"
    do   
                   
        python run_estimation.py \
            -c output/cached_results/$dataset/$model/results.json \
            -o output/cached_results/$dataset/$model/ \
            --correctness_metric bem \
            --sd_batch_size 1 \
            --sd_cached_path output/cached_results/$dataset/$model/ \
            --se_batch_size 3 \
            --se_cached_path output/cached_results/$dataset/$model/ \
            --sc_batch_size 3 \
            --sc_cached_path output/cached_results/$dataset/$model/ \
            --sar_batch_size 3 \
            --sar_cached_path output/cached_results/$dataset/$model/ \
            --inside_batch_size 3 \
            --inside_cached_path output/cached_results/$dataset/$model/ \
            --mi_batch_size 1 \
            --mi_cached_path output/cached_results/$dataset/$model/ \
            --mi_cached_mu2_path output/cached_results/$dataset/$model/ \
            --pt_cached_path output/cached_results/$dataset/$model/ \
        
        python run_gradient.py -d $dataset -m $model -o output/cached_results/$dataset/$model/ -c output/cached_results/$dataset/$model/ -b 1 --correctness_metric bem -ne
    done    
done




