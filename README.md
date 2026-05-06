# Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
This is the official repository for our ICML 2026 paper [Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models]


# Experiments
1. [Installation](#Installation)
2. [Model Responses Generation and Evaluation](#Generation)
4. [Run SemGrad and Baselines](#SemGrad)

## Installation
1. Create and activate a Python environment
    ```bash
    conda create -n 
    conda activate esi
    ```

2. Install via requirements.txt
    ```bash
    pip install -r requirements.txt
    ```

**Other models and files needed:**

- Bem model (Used for correctness evaluation). Please download the model from [Download link](https://tfhub.dev/google/answer_equivalence/bem/1) then save it to the directory **data/model/** (or change the default cache path **CACHED_BEM_PATH** in **uncertainty/generation_evaluation/metrics/bem.py** line 14). 

- Other models needed can be downloaded directly from HuggingFace. you can add or replace any model by modifying the LLM_MODEL_CONFIG attribute in the LLM class in **uncertainty/utils/llm.py**:
  
    ```bash
    class LLM
     └───LLM_MODEL_CONFIG
         └───[the name to represent this model]
             │───model_name: [the name to represent this model]
             │───model_path: [path used for loading the model using .from_pretrained in huggingface]
             │───model_class: [the model class used before .from_pretrained in huggingface, such as 'AutoModel']
             │───fp16: [Whether use half precision]
             │───tokenizer_path: [path used for loading the tokenizer using .from_pretrained in huggingface]
             │───tokenizer_class: [tokenizer class used before .from_pretrained in huggingface]
    ```
  

## Generation
To generate the answers of QA datasets and evaluate their correctness, you can directly run the bash script to run on all datasets and models
```
bash run_generations.bash
```

If you want results on a specific dataset and model, run the following code:
```
python run_generation.py -c config/${dataset}_config.yaml -dp data/datasets/${dataset}/test.jsonl -o output/cached_results/${dataset} -m ${model} -b 20
```
change ${dataset} to the dataset name you wanna run. Supported names are the directory names in **data/datasets**. Change ${model} to the base model name you wanna estimate the uncertainty. Model name should be the same as the keys defined in LLM_MODEL_CONFIG attribute of LLM class.


## SemGrad
### RUN SemGrad and HybridGrad
Evaluate the performance of SemGrad and HybridGrad as follows:
```
bash run_semgrad.bash
```
Before running the bash script, change the -c ${path_save_the_generation_results} to the path saved "results.json", which saves all results generated in the above generation and evaluation step.


### RUN BASELINES
Evaluate the performance of SemGrad and HybridGrad as follows:
```
bash run_baselines.bash
```
Before running the bash script, change the -c ${path_save_the_generation_results} to the path to "results.json", which saves all results generated in the above generation and evaluation step.


## Other
- When generating parapharses for ESI(Para), please add your Deepseek API keys to **uncertainty/run_prompt_paraphrase.py** line 16 'DEEPSEEK_API_KEY'.




