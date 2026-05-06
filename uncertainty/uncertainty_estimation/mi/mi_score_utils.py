import numpy as np
from uncertainty.generation_evaluation import f1_score
from uncertainty.utils.prompt_template import PromptTemplate
from uncertainty.utils import LLM, get_logits, reshape_sequences
def read_mu_1(responses, probs_list, f1_threshold=0.25):
    
    """
    This function reads the mu_1 values from the responses and probabilities list.
    It creates a mapping of distinct answers to their probabilities, and then normalizes the probabilities.
    
    Inputs:
    responses: List[List[str]] - A list of lists containing the responses for each query.
    probs_list: List[List[float]] - A list of lists containing the probabilities for each response.
    f1_threshold: float - The threshold for the F1 score to consider two answers as distinct.

    Outputs:
    mu_1_kv_mapping: List[Dict[str, float]] - A list of dictionaries where each dictionary contains the distinct answers and their normalized probabilities.
    """
    deduplicate_dict = [ {r: p for r, p in zip(rs, probs)} for rs, probs in zip(responses, probs_list)]

    mu_1_kv_mapping = []
    for ex in deduplicate_dict:
        all_answers = list(ex.keys())
        distinct_answer_prob_mapping = dict()
        total_probs = sum(list(ex.values()))
        for ans in all_answers:
            if distinct_answer_prob_mapping == dict():
                distinct_answer_prob_mapping[ans] = ex[ans]
            else:
                is_distinct = True
                for dis_a in distinct_answer_prob_mapping.keys():

                    if f1_score(ans, dis_a) > f1_threshold:

                        distinct_answer_prob_mapping[dis_a] += ex[ans]
                        is_distinct = False
                        break
                if is_distinct:
                    distinct_answer_prob_mapping[ans] = ex[ans]
        mu_1_kv_mapping.append({k: v/total_probs for k,v in distinct_answer_prob_mapping.items()})
    return mu_1_kv_mapping

def construct_mu_2_prompt(query, original_answer, builder):

    
    if original_answer.endswith("."):
        original_answer = original_answer[:-1]
    
    return builder.build_prompt({"query": query, "response_1": original_answer})

def read_probs_of_conditional_mu(query_ans1_ans2_triplet_list, model_name, device_name=None, batch_size=5):
    """
    This function reads the probabilities of conditional mu values from the query-answer triplet list.
    
    Inputs:
    query_ans1_ans2_triplet_list: List[Tuple[str, str, str]] - A list of tuples containing the query, answer 1, and answer 2.

    Outputs:
    
    """
    template = "Consider the following question: Q: {query}\nOne answer to question Q is {response_1}.\nPlease directly provide an answer to the following question with one or few words:\n{query}"
    builder = PromptTemplate(model_name=model_name, template=template, system_message=None, use_system_message=False)

    prompts = []
    outputs = []
    for query, ans1, ans2 in query_ans1_ans2_triplet_list:
        prompt = construct_mu_2_prompt(query, ans1, builder)
        prompts.append(prompt)
        outputs.append(ans2)
    
    tokenizer = LLM.initial_tokenizer(model_name)
    prompt_ids = tokenizer(prompts, padding=False, add_special_tokens=True)["input_ids"]
    prompt_lens = [len(p) for p in prompt_ids]
    response_ids = tokenizer(outputs, padding=False, add_special_tokens=False)["input_ids"]
    response_ids = [r[:-1] if r[-1] == tokenizer.eos_token_id else r for r in response_ids]
    input_ids = [p + r for p, r in zip(prompt_ids, response_ids)]
    lm_name = LLM.initial_lm(model_name, device_name)

    model, _ = LLM.loaded_llms[lm_name]
    logprobs = get_logits(input_ids, model, tokenizer, batch_size=batch_size, return_transition_scores=True, prompt_lens=prompt_lens, num_scores_returned=1)["transition_scores"]
    probs = [np.exp(np.sum(lp)) for lp in logprobs]
    
    LLM.release_all()

    return {mu_cached_key(*k): p for k, p in zip(query_ans1_ans2_triplet_list, probs)}
    

def read_mu_2(mu_1_dict_list, query_list, model_name, cached_mu_2_result=None, device_name=None, batch_size=5):
    """
    This function reads the mu_2 values from the mu_1 dictionary list and query list.
    Input:
    mu_1_dict_list: List[Dict[str, float]] - A list of dictionaries where each dictionary contains the distinct answers and their probabilities.
    query_list: List[str] - A list of queries.
    model_name: str - The name of the model to be used.
    
    Outputs:
    cached_mu_2_result: Dict[str, float] - A dictionary containing the cached mu_2 results.
    """

    key_triplet = []
    for query, mu_1_dict in zip(query_list, mu_1_dict_list):
        distinct_responses = list(mu_1_dict.keys())
        if len(distinct_responses) == 1:
            continue

        for as1 in distinct_responses:
            for as2 in distinct_responses:
                
                key_triplet.append((query, as1, as2))
    
    if cached_mu_2_result is None:
        print("No cached mu_2 result found, reading from model...")
        cached_mu_2_result = read_probs_of_conditional_mu(key_triplet, model_name, device_name=device_name, batch_size=batch_size)
    else:
        uncached_triplet = []
        for triplet in key_triplet:
            if mu_cached_key(*triplet) not in cached_mu_2_result:
                uncached_triplet.append(triplet)
        if len(uncached_triplet) > 0:
            print(f"Found {len(uncached_triplet)} uncached mu_2 results, reading from model...")
            addition_result = read_probs_of_conditional_mu(uncached_triplet, model_name, device_name=device_name, batch_size=batch_size)
            cached_mu_2_result.update(addition_result)
        else:
            print("All mu_2 results are cached, no need to read from model.")
        
        
    return cached_mu_2_result


def mu_cached_key(query, ans1, ans2):
    return f"{query}[mu_2_sep]{ans1}[mu_2_sep]{ans2}"






                