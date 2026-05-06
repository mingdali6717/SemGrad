from transformers import DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
MAX_LENGTH = int(10000)




def get_logits(texts_or_ids, model, tokenizer,  batch_size=10, return_transition_scores=False, num_scores_returned=-1, temperature=None, prompt_lens=None):
    """
    input:
    texts_or_ids - str or List[Str] or List[int] or List[List[int]], a seqence of texts or a sequence of ids tokenized to view logits
    model - nn.module
    tokenizer
    batch_size: int
    return_transition_scores: bool - if True, will return the log softmax of the logits of the generated tokens. 
    num_scores_returned: int, default -1 - top-k scores will be returned for each token position, k equal the 'num_scores_returned'. if less or equal to zero, all scores will be returned
    prompt_lens: None or List[int] - the length of the prompt ids in each texts which will be substracted from the output logits. the output logits will be in length of (full_length - prompt_length). if set to None, the default length will be 1.

    return:

        result - dict:
            keys: 
                - input_ids: List[List[int]] batch_size x sequence_length - the exact tokenized token id input to the model.
            
                - logits: dict
                    -scores: List[List[List[float]]] batch_size x (sequence_len-prompt_len) x num_scores_returned - the corrresponding logits of the input ids output by the model.
                    -ids: List[List[List[float]]] batch_size x (sequence_len-prompt_len) x num_scores_returned - the corresponding token id of the logit score.
                    
                - transition_scores [optional]: (List[List[float]] batch_size x (sequence_length - prompt_len) - the logprob of generating the token in the input_ids from the second position.
    """
    # get the probabilities

    vocab_size = len(tokenizer.get_vocab())
    if num_scores_returned >= vocab_size:
        print(f"the number of scores returned per token is set to {num_scores_returned} which is larger than the vocab size {vocab_size}, ALL SCORES WILL BE RETURNED")
        num_scores_returned = vocab_size
    elif num_scores_returned <= 0:
        print(f"the number of scores returned per token is set to {num_scores_returned} which is less than 0, ALL SCORES WILL BE RETURNED")
        num_scores_returned = vocab_size

    if isinstance(texts_or_ids, str) or (isinstance(texts_or_ids, list) and isinstance(texts_or_ids[0], int)):
        texts_or_ids = [texts_or_ids]
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    max_sequence_length = max(getattr(model.config, "max_position_embeddings", 0), getattr(model.config, "n_positions", 0), getattr(model.config, "seq_length", 0))
    if max_sequence_length == 0:
        print(f"model max input length is not detected, set max_prompt_length to {MAX_LENGTH}")
        max_length = MAX_LENGTH
    if isinstance(texts_or_ids[0], str):
        inputs = tokenizer(texts_or_ids, padding=False, truncation=True, max_length=max_length).data
    elif isinstance(texts_or_ids[0][0], int):

        attention_mask = [[1 if _id != tokenizer.pad_token else 0 for _id in ids] for ids in texts_or_ids]
        
        inputs = {"input_ids": texts_or_ids, "attention_mask": attention_mask}
    else:
        raise ValueError("given texts_or_ids should be a str;List[str];List[int] or List[List[int]]")
    
    if prompt_lens is None:
        prompt_lens = [1] * len(inputs["input_ids"])
    assert len(prompt_lens) == len(inputs["input_ids"]), f"prompt_len should be with the same length of num of prompts '{len(inputs['input_ids'])}', but {len(prompt_lens)} prompt_len are given"
    inputs["prompt_lens"] = prompt_lens
    
    scores = []
    ids = []
    transition_scores = []
    
    with torch.no_grad():
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8 if model.dtype==torch.float16 else None)
        dataloader = DataLoader(Dataset.from_dict(inputs), batch_size=batch_size, collate_fn=data_collator)
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            seq_lens = attention_mask.sum(dim=1).tolist()
            outputs = model(input_ids = input_ids, attention_mask = attention_mask).logits
            trun_scores, trun_ids = torch.topk(outputs, num_scores_returned, dim=-1, sorted=True) # shape(batch_size, sequence_length, num_scores_returend_per_token)
            trun_scores = remove_pad(trun_scores.tolist(), seq_lens)
            trun_ids = remove_pad(trun_ids.tolist(), seq_lens)

            scores.extend([s[l-1:-1] for s, l in zip(trun_scores, batch["prompt_lens"].tolist())])
            ids.extend([s[l-1:-1] for s, l in zip(trun_ids, batch["prompt_lens"].tolist())])

            if return_transition_scores:
                if temperature is None:
                    generate_scores = tuple(outputs[:,i,:] for i in range(outputs.shape[1]))[:-1]  
                else:
                    generate_scores = tuple(outputs[:,i,:]/temperature for i in range(outputs.shape[1]))[:-1]

                sequence_ids = input_ids[:, 1:]
                transition_score = model.compute_transition_scores(sequence_ids, generate_scores, normalize_logits=True)#shape batch_size x (max_seq_length -1)
                transition_score = remove_pad(transition_score.tolist(), [l-1 for l in seq_lens])
        
                transition_scores.extend([s[l-1:] for s, l in zip(transition_score, batch["prompt_lens"].tolist())])

        
    # list of every tokenized word's distribution from the second one

    tokenizer.padding_side = padding_side
    result = {
        "input_ids": inputs["input_ids"],
        "logits": {"scores": scores, "ids": ids}
    }
    if return_transition_scores:
        result["transition_scores"] = transition_scores
    return result

def check_model_generated(logits, token_ids, seq_length, start_position=0):
    """
    check whether the token ids are greadily generated decoded from the logits
    input:
    logits: torch_tensor[batchsize, max_sequence_length, vocab_size]
    token_ids: torch_tensor[batch_size, max_sequence_length]
    seq_length: List[batch_size]
    start_position: int, the start position of decoding
    """
    
    id_decoded = torch.argmax(logits, dim=-1)[:, start_position:-1]
    token_ids = token_ids[:, start_position + 1:]
    seq_length = [s - (start_position + 1) for s in seq_length]

    
    max_len = max(seq_length)

    # Create an empty mask with all elements zero
    attention_mask = torch.zeros((len(seq_length), max_len), dtype=torch.long)

    # Fill the mask
    for idx, seq_len in enumerate(seq_length):
        attention_mask[idx, :seq_len] = 1

    # Apply the mask by element-wise multiplication
    masked_token_ids = token_ids * attention_mask.int() 
    masked_decoded_ids = id_decoded * attention_mask.int()
    return torch.eq(masked_decoded_ids, masked_token_ids)

def remove_pad(data_list, seq_lengths):
    """
    remove the padding position in the second dimenson of data_list following the seqence length given in seq_lengths
    input: 
        data_list: List[List[****]] batch_size x padded_sequence_length x *****
        seq_length: List, batchsize.

    """
    assert len(data_list) == len(seq_lengths), f"data_list (length {len(data_list)}) and seq_lengths {len(seq_lengths)} are in different lens."
    for i in range(len(data_list)):
        data_list[i] = data_list[i][:seq_lengths[i]]
    return data_list