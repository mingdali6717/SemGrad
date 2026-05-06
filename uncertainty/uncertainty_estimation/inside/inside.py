import torch
import json
import numpy as np
from tqdm import tqdm
from uncertainty.utils import LLM
from uncertainty.utils.llm import process_eos_token, adjust_length_to_model, get_sequence_length, truncate_eos_tokens
from loguru import logger
from itertools import chain
from transformers import DataCollatorForLanguageModeling
 
from datasets import Dataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_INSIDE_CONFIG = {
    "cached_path": None,
    "save_path": None,
    "generation_config": {
        "do_sample": True,
        "num_responses_per_prompt": 10,
        "temperature": 0.5,
        "top_p": 0.99,
        "top_k": 5,
        "output_scores": False,
        "return_normalized_transition_scores": False,
        "batch_size": 5,
        "output_hidden_states": True

    }
}

def calculate_inside(prompts, model_name, generate_kwargs, tokenize_kwargs, cached_path=None, save_path=None, device = None, verbose=True):
    if cached_path is not None:
        with open(cached_path, "r") as f:
            cached_data = json.load(f)
        last_token_hidden_states_list = [cached_data[q] for q in prompts]
    else:
    
        assert model_name in LLM.LLM_MODEL_CONFIG.keys(), f"model '{model_name}' is not supported!"
        model_key = LLM.initial_lm(model_name, device, verbose, tokenizer_kwargs=tokenize_kwargs)
        model, tokenizer = LLM.loaded_llms[model_key]
        
        
        # setting generate_kwargs
        # setting generate_kwargs
        if ("num_responses_per_prompt" in generate_kwargs) :
            if generate_kwargs["do_sample"] and (generate_kwargs["num_responses_per_prompt"] > 1):
                n = generate_kwargs["num_responses_per_prompt"]
            else:
                if generate_kwargs["num_responses_per_prompt"] > 1:
                    if generate_kwargs.get("num_beams", None) is not None:
                        assert generate_kwargs["num_beams"] >= generate_kwargs["num_responses_per_prompt"], f"num_beams should be larger than or equal to num_responses_per_prompt, but {generate_kwargs['num_beams']} < {generate_kwargs['num_responses_per_prompt']}"
                        n = generate_kwargs["num_responses_per_prompt"]
                    else:
                        logger.warning(f"num_responses_per_prompt is set to {generate_kwargs['num_responses_per_prompt']} > 1, but do_sample is set to False and num_beams is not given, hence only 1 response will be generated for each prompt")
                        n = 1
                else:
                    n=1
            generate_kwargs.pop("num_responses_per_prompt")
        else:
            n = 1
            
        generate_kwargs["num_return_sequences"] = n
    
        return_normalized_transition_scores=False
        if generate_kwargs.get("eos_token_id", None) is not None:
            print(f"other eos_tokens are provided: {generate_kwargs['eos_token_id']}")
            if isinstance(generate_kwargs["eos_token_id"], str):
                generate_kwargs["eos_token_id"] = [generate_kwargs["eos_token_id"]]
            single_eos_token_ids, other_eos_token_ids = process_eos_token(generate_kwargs["eos_token_id"], tokenizer)
            
            generate_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + list(chain(*single_eos_token_ids))
        else:
            other_eos_token_ids = None
        

            
        if ("llama3" in model_name.lower() or "llama-3" in model_name.lower()) and "instruct" in model_name.lower():
            if generate_kwargs.get("eos_token_id", None) is not None:
                if tokenizer.eos_token_id not in generate_kwargs["eos_token_id"]:
                    generate_kwargs["eos_token_id"].append(tokenizer.eos_token_id)
                if tokenizer.convert_tokens_to_ids("<|eot_id|>") not in generate_kwargs["eos_token_id"]:
                    generate_kwargs["eos_token_id"].append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
            else:
                generate_kwargs.update({"eos_token_id": [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]})
        if generate_kwargs.get("eos_token_id", None) is not None:
            logger.info(f"additional eos tokens are given:\nsingle eos_token_ids: {generate_kwargs['eos_token_id']}\nother eos_token_ids : {other_eos_token_ids}")

        if type(prompts) is str:
            prompts = [prompts]
        
        generate_kwargs.pop("num_scores_returned")
        generate_kwargs.pop("return_normalized_transition_scores") 
        terminators = generate_kwargs.get("eos_token_id")
        n = generate_kwargs["num_return_sequences"]

        
        max_sequence_length = max(getattr(model.config, "max_position_embeddings", 0), getattr(model.config, "n_positions", 0), getattr(model.config, "seq_length", 0))
        max_prompt_length, max_new_tokens = adjust_length_to_model(generate_kwargs["max_new_tokens"], max_sequence_length) 
        generate_kwargs["max_new_tokens"] = max_new_tokens
        tokenize_kwargs["max_length"] = max_prompt_length
        tokenize_kwargs["truncation"] = True
        #tokenize_kwargs["return_tensors"] = "pt"
        padding = tokenize_kwargs.pop("padding")
        tokenize_kwargs["padding"] = False
        tokenize_prompt = tokenizer(prompts, **tokenize_kwargs)
        batch_size = generate_kwargs.pop("batch_size", 10)

        with torch.no_grad():
            
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8 if model.dtype==torch.float16 else None)
            dataloader = DataLoader(Dataset.from_dict(tokenize_prompt.data), batch_size=batch_size, collate_fn=data_collator)
            if terminators is not None:
                eos_token_id = terminators
            else:
                if hasattr(model, "generation_config"):
                    eos_token_id = model.generation_config.eos_token_id
                else:
                    eos_token_id = model.config.eos_token_id
            
            last_token_hidden_states_list = []

            for batch in tqdm(dataloader):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True, **generate_kwargs)
                output_seq = outputs.sequences[:, input_ids.shape[1]:]
        
                seq_lens = get_sequence_length(output_seq.tolist(), eos_token_id)
                if other_eos_token_ids is not None: 
                    truncate_lens = [truncate_eos_tokens(r_ids, other_eos_token_ids) for r_ids in output_seq.tolist()]
                    seq_lens = [min(t_l, s_l) for t_l, s_l in zip(truncate_lens, seq_lens)]
                hidden_states = list(outputs.hidden_states)
                del outputs
                total_layer_num = len(hidden_states[0])
                select_layer_num = (total_layer_num-1) // 2
                hidden_states[0] = (layer[:, -1, :].unsqueeze(1) for layer in hidden_states[0])
                hidden_states_tensor = torch.cat([torch.cat([h.unsqueeze(1) for h in tokens], dim=1).squeeze(-2).unsqueeze(1) for tokens in hidden_states], dim=1).cpu()
                del hidden_states
                
                hidden_states_tensor = hidden_states_tensor[:, :, select_layer_num, :]
                last_token_index = [s - 1 for s in seq_lens]
                last_token_hidden_state = hidden_states_tensor[torch.arange(hidden_states_tensor.shape[0]), torch.tensor(last_token_index), :].reshape(-1, n, hidden_states_tensor.shape[-1]).tolist()
                last_token_hidden_states_list.extend(last_token_hidden_state)
                
        
        if save_path is not None:
            with open(save_path, "w") as f:
                json.dump({q: h for q,h in zip (prompts, last_token_hidden_states_list)}, f, indent=4)
        
    return {
        "inside": [calculate_EigenScore(hs, len(hs)).item() for hs in last_token_hidden_states_list]
        }
    



def calculate_EigenScore(data_matrix, sample_size):
    data_matrix = np.array(data_matrix, dtype = np.float64)
    cov_matrix = np.cov(data_matrix)+ 0.001 * np.diag(np.ones(sample_size, dtype=np.float64))
    return np.log(np.linalg.det(cov_matrix))/sample_size
