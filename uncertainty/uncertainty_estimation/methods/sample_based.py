from itertools import chain
from ...utils import LLM, reshape_sequences

class BASE_UNCERTAINTY_ESTIMATOR:

    def __init__(self, model_name, white_box=True, sample_num=1):
        self.white_box = True
        self.sample_num = sample_num

    def _prompts_preprocess(self, prompts):
        return prompts
    
    def _prompts_spawn(self, prompts):
        """
        the function to spawn n prompts for each query for sampled responses generation.
        input:
        prompts: str or List[str] or List[List[str]]. the prompts to evaluate uncertainty

        output:
        spawned_prompts: List[List[str]]. num_prompts x sample_num. 

        """
        if isinstance(prompts, str):
            spawned_prompts = [[prompts] * self.sample_num]
        elif isinstance(prompts, list) and isinstance(prompts[0], str):
            spawned_prompts = [[s]*self.sample_num for s in prompts]
        elif isinstance(prompts, list) and isinstance(prompts[0], list) and isinstance(prompts[0][0], str):
            spawned_prompts = [s*self.sample_num for s in prompts]
        else:
            raise ValueError("input prompts should be a string or a list of string or a list of list of string, but the given is not")
        
        return spawned_prompts

    def _generate_sample_responses(self, prompts, model_name_or_model, generation_kwargs, tokenization_kwargs, device_name, verbose=True, tokenizer=None):
        """
        input:
        model_or_model_name: Str or torch.nn.Module. if an OpenAI model name is given, wil call the openai API. otherwise will initialize based on LLM_MODEL_CONFIG.
        prompts: List[str]
        generate_kwargs: Dict
            {
            batch_size: int,
            temperature: float,
            top_p: float,
            top_k: int,
            do_sample: bool,
            max_new_tokens: int,
            output_scores: bool - if True, will return all output logits for each token position.
            num_score_returned: int, default 100. - the top-k scores will be returned for each token position. if less or equal to zero, all scores will be returned
            num_responses_per_prompt: int, - the number of reponses generated for each prompt, only activated when do_sample=True.
            return_normalized_transition_scores: bool, - if true, will return the log softmax of the logits of the generated tokens.
            }
        tokenizer_kwargs: Dict
            {
            padding: str,
            truncation: bool,
            padding_side: str,
            truncation_side: str
            }
        device_name: 'cpu' or 'gpu{gpu_id}'
        verbose: bool
        tokenizer: if model_or_model_name is a nn.module model, the correponding tokenizer is required

        return:

        outputs - dict:
            keys: - prompts: (List[str] or List[List[str] if self.sample_num > 1]
                  - responses: (List[str] or List[List[str]] if self.sample_num>1), generated responses
                  - response_ids: (List[List[int]] or List[List[List[int]]]) shape(prompts_num, self.sample_num, max_seq_length) if self.sample_num > 1 else (prompts_num, max_seq_length))
                  - scores [optional]: (List[List[List[float]]] or List[List[List[List[float]]]]) shape (prompts_num, n, max_seq_length, vocab_size) if n > 1 else (prompts_num, max_seq_length, vocab_size). logits of all position
                  - transition_scores [optional]: (List[List[int]] or List[List[List[int]]]) shape (prompts_num, n, max_seq_length) if n > 1 else (prompts_num, max_seq_length). log prob of each generated tokens
        """
        spawned_prompts = list(chain(*self._prompts_spawn(prompts)))
        processed_prompts = self._prompts_preprocess(spawned_prompts)
        generation_kwargs["num_responses_per_prompt"] = 1
        outputs = LLM.lm_generate(model_name_or_model, processed_prompts, generation_kwargs, tokenization_kwargs, device_name=device_name, verbose=verbose, tokenizer=tokenizer)

        if self.sample_num > 1:
            for k, v in outputs.items():
                if isinstance(v, list):
                    outputs[k] = reshape_sequences(v, self.sample_num)
                else:
                    new_logits = dict()
                    for logits_k, logits_v in v.items():
                        new_logits[logits_k] = reshape_sequences(logits_v, self.sample_num)
                    outputs[k] = new_logits
        
        return outputs

    


