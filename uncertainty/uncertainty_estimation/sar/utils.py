from ...generation_evaluation import SemSimCalculator
from itertools import chain


def get_tokenwise_importance(questions, response_ids, tokenizer, sim_models,device_name="gpu0"):

    """
    question: List[str] or str
    responses: List[str] or str
    tokenier:  transformers.tokenizer
    sim_models: str a semantic similarity model supported by the class SemSimCakculator. all supported models can be found in uncertainty.generation_evaluation.metrics.semantic_similarity.ALL_SUPPORTED_MODELS
    device_name: str - "cpu" or "gpu{rank}", rank is the int number of current gpu device.

    return: List[List[int]] response_num x token_num, token num is the number of tokens after tokenizing the response with the tokenizer.
    """
    if isinstance(response_ids, list) and (not isinstance(response_ids[0]), list):
        response_ids = [response_ids]
    if isinstance(questions, str):
        questions = [questions]
    measure_model = SemSimCalculator(model_name=sim_models, device_name=device_name)
    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    token_nums = [len(r) for r in response_ids]
    rm_token_responses = list(chain(*[rm_tokens(r, tokenizer) for r in response_ids]))
    expand_questions = list(chain(*[[q]*n for q,n in zip(questions, token_nums)]))
    expand_responses = list(chain(*[[r]*n for r,n in zip(responses, token_nums)]))
    original_inputs = [q.strip() + " " + r.strip() for q,r in zip(expand_questions, expand_responses)]
    rmtoken_inputs = [q.strip() + " " + r.strip() for q,r in zip(expand_questions, rm_token_responses)]

    flat_scores = (1 - measure_model(original_inputs, rmtoken_inputs, batch_size = 256)).tolist()
    scores = []
    pos = 0
    for l in token_nums:
        scores.append(flat_scores[pos:pos+l])
        pos+=l
    return scores
    

    

    

                                                                
def rm_tokens(sentence_id, tokenizer):
    rmed_sentences = []
    for i in range(len(sentence_id)):
        pre_part = sentence_id[:i]
        latter_part = sentence_id[i+1:]
        rm_tok = tokenizer.decode(sentence_id[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)     
        if rm_tok.startswith(" "):
            rmed_sentences.append(tokenizer.decode(pre_part, skip_special_tokens=True, clean_up_tokenization_spaces=True) + " " + tokenizer.decode(latter_part, skip_special_tokens=True, clean_up_tokenization_spaces=True))
        else:
            rmed_sentences.append(tokenizer.decode(pre_part + latter_part, skip_special_tokens=True, clean_up_tokenization_spaces=True))
    return rmed_sentences    
               