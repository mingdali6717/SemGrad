from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer
from ...utils import LLM
from bert_score import bert_cos_score_idf, get_idf_dict, sent_encode
from collections import Counter, defaultdict
from multiprocessing import Pool, get_context
from transformers import DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader
from functools import partial
from itertools import chain
import torch
from tqdm import tqdm
from math import log
import json
import gc
import os

CROSS_ENCODER_MODELS = ["cross-encoder/nli-deberta-v3-base","cross-encoder/stsb-roberta-large","cross-encoder/stsb-distilroberta-base"]
SENTENCE_TRANSFORMER_MODELS = ["all-mpnet-base-v2"]
BERT_SCORE_MODELS = ["bertscore/deberta"]
NLI_MODELS = ["nli/deberta_nli"]

SENTENCE_TRANSFORMER_MODELS_CACHED_PATH = {
    "all-mpnet-base-v2": ""
}
ALL_SUPPORTED_MODELS = CROSS_ENCODER_MODELS + SENTENCE_TRANSFORMER_MODELS + BERT_SCORE_MODELS + NLI_MODELS

class SemSimCalculator:
    
    def __init__(self, model_name, device_name = None, nli_return_type="score", nli_score_type="entailment"):
        """
        parameters:
        model_name: str - one of the modol name in "ALL_SUPPORTED_MODELS"
        device_name: str - "cpu" or "gpu{rank}", rank is the int number of current gpu device.
        """
        if model_name in CROSS_ENCODER_MODELS:
            self.scorer = CrossEnocoderEvaluator(model_name, device_name=device_name)
        elif model_name in SENTENCE_TRANSFORMER_MODELS:
            self.scorer = SentenceTransformerEvaluator(model_name, device_name=device_name)
        elif model_name in BERT_SCORE_MODELS:
            self.scorer = BertScoreEvaluator(model_name, device_name=device_name)
        elif model_name in NLI_MODELS:
            self.scorer = NLIEvaluator(model_name, device_name=device_name, return_type=nli_return_type, score_type=nli_score_type)
        else:
            raise ValueError(f"given model_name is not supported, the supported model name is as follows\n{json.dumps(ALL_SUPPORTED_MODELS, indent=4)}") 

    def __call__(self, hyps, refs, prepend_text = None, batch_size = 32):
        """
        parameters:
        hyps: str or List[str]. - the fist sequence to compute similarity score
        refs: str or List[str]. - the second sequence to compute similarity score
        prepend_text: None or str or List[str] - if given, will be prepended to the corrsponding hyp and ref sentence before computing the similarity score.
        batch_size: int - batch size for computing, default to 32

        return: torch.tensor shape[hyp_len]
        """
        
        if isinstance(hyps, str):
            hyps = [hyps]
        if isinstance(refs, str):
            refs = [refs]
        if isinstance(prepend_text, str):
            prepend_text = [prepend_text]
        assert len(hyps) == len(refs), "given two sequences of sentences are in different lenght"
        if prepend_text is not None:
            assert len(prepend_text) == len(hyps), f"text to be prepended should be in the same length as hyps and refs"
            hyps = [q.strip()+" "+h.strip() for q,h in zip(prepend_text, hyps)]
            refs = [q.strip()+" "+r.strip() for q,r in zip(prepend_text, refs)]
        
        def dedup_and_sort(l):
            return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)
        
        concated_sents = [hyp + "[$usedforsep$]" + ref for hyp, ref in zip(hyps, refs)]
        sents = dedup_and_sort(concated_sents)
        compact_hyps, compact_refs = list(zip(*[s.split("[$usedforsep$]") for s in sents]))
        
        compact_scores =  self.scorer(compact_hyps, compact_refs, batch_size=batch_size)
        stat_dict = {k:v for k,v in zip(sents, compact_scores.tolist())}
        scores = [stat_dict[k] for k in concated_sents]
        return torch.tensor(scores, dtype=compact_scores.dtype, device=compact_scores.device)

    def release_model(self):
        self.scorer.release_model()
        
      
    
    




class SentenceTransformerEvaluator():

    def __init__(self, model_name, device_name=None):
        device = LLM.to_torch_device(device_name)

        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            print(f"Model '{model_name}' is not available. Error: {e}")
            if model_name in SENTENCE_TRANSFORMER_MODELS_CACHED_PATH:
                model_path = SENTENCE_TRANSFORMER_MODELS_CACHED_PATH[model_name]
                if not os.path.exists(model_path):
                    print(f"given cached file for model '{model_name}' is not found, start to download model from internet")
                else:
                    model_name = model_path
        
        
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
    
    def __call__(self, hyps, refs, batch_size=64, normalize_embeddings=False):
        hyps_embedding = self.model.encode(hyps, batch_size=batch_size, normalize_embeddings=normalize_embeddings, show_progress_bar=True)

        refs_embedding = self.model.encode(refs, batch_size=batch_size, normalize_embeddings=normalize_embeddings, show_progress_bar=True)

        sim_scores = self.model.similarity_pairwise(hyps_embedding, refs_embedding)

        return sim_scores.cpu()
    
    def release_model(self):
        del self.model
        torch.cuda.empty_cache()
    

class CrossEnocoderEvaluator():

    def __init__(self, model_name, device_name=None):
        device = LLM.to_torch_device(device_name)
        self.model = CrossEncoder(model_name, device=device, local_files_only=True)
        self.model_name = model_name
    
    def __call__(self, hyps, refs, batch_size=64):
        sentence_pairs = list(zip(hyps, refs))
        sim_scores = self.model.predict(sentence_pairs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        if "nli" in self.model_name:
            sim_scores = torch.softmax(sim_scores, dim=-1)[:, 1]
        return sim_scores.cpu()

    def release_model(self):
        del self.model
        torch.cuda.empty_cache()
        

class BertScoreEvaluator():
    def __init__(self, model_name, device_name=None):
        llm_name = LLM.initial_lm(model_name, device_name)
        self.model, self.tokenizer = LLM.loaded_llms[llm_name]
        self.llm_name = llm_name
        self.model_name = model_name
        
        if device_name is None:
            if not torch.cuda.is_available():
                device_name = "cpu"
            else:
                device_name = [f"gpu{i}" for i in range(torch.cuda.device_count())][0]
        self.device_name = device_name


    def __call__(self, hyps, refs, batch_size=64):
        """
        Calculate bert score with idf weighted
        args:
        model_name: str -  base model for embedding caculation, should be one of supported model name in LLMConfig
        hyps: List[str] - candidate sentences
        refs: List[str] - reference sentences
        device: str - device to use, should be in the format of f"gpu{device_id}" or 'cpu'

        return: torch.tensor
        """
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model = self.model.module

    

        num_layers = 40 if self.model_name == "deberta" else 8

        if len(self.model.encoder.layer) > num_layers:
            self.model.encoder.layer = torch.nn.ModuleList([layer for layer in self.model.encoder.layer[:num_layers]])
        elif len(self.model.encoder.layer) < num_layers:
            assert False, "Model layer num error"


        idf_dict = get_idf_dict(hyps, self.tokenizer)
        idf_dict[self.tokenizer.sep_token_id] = 0
        idf_dict[self.tokenizer.cls_token_id] = 0  


        bert_scores = bert_cos_score_idf(self.model, refs, hyps, self.tokenizer, idf_dict, device=self.device_name.replace("gpu","cuda:"), batch_size=batch_size, verbose=True).cpu()


        return bert_scores[:, 2]
    
    def release_model(self):
        del self.model
        LLM.release_one(self.llm_name)
        

class NLIEvaluator():

    def __init__(self, model_name, return_type="score", device_name=None, score_type="entailment"):
        """
        the class to calculate the natural language entailment of text pairs. if return type is score, the probability of score_type will be returned. If 'class', the classified class will be returned, 0 represent contradiction, 1 represent entail and 2 represents neutral. if "bool", will return 0 or 1, 1 represents entailments. if soft_bool, will return 0 or 1, 0 represents contracdition while 1 represent entailment or neutral. If distribution, will return the distribution of contradiction, entailment and neutral.
        """
        assert return_type in ["score", "class", "bool", "soft_bool", "distribution"], f"the return type should one of 'score', 'class' or 'bool', 'soft_bool' or 'distribution', but got '{return_type}'"
        if model_name == "potsawee/deberta-v3-large-mnli":
            from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
            device = LLM.to_torch_device(device_name)
            self.tokenizer = DebertaV2Tokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
            self.model = DebertaV2ForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli", device_map=device, torch_dtype=torch.float32)
            self.tokenizer.padding_side = "right"
            self.tokenizer.truncation_side = "right"
        else:
            llm_name = LLM.initial_lm(model_name, device_name)
            self.model, self.tokenizer = LLM.loaded_llms[llm_name]
            self.llm_name = llm_name
        self.model_name = model_name
        self.return_type = return_type
        self.score_type = score_type
        if device_name is None:
            if not torch.cuda.is_available():
                device_name = "cpu"
            else:
                device_name = [f"gpu{i}" for i in range(torch.cuda.device_count())][0]
        self.device_name = device_name
    
    def __call__(self, hyps, refs, batch_size=64):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model = self.model.module
        scores = calculate_nli_score(self.model, self.tokenizer, hyps, refs, batch_size=batch_size)
        if self.return_type == "score":
            return_score = torch.tensor([s[self.score_type] for s in scores], dtype=torch.double)
            return return_score.cpu()
        else:
            scores_ts = torch.tensor([[s["contradiction"], s["entailment"], s["neutral"]] for s in scores], dtype=torch.double)
            class_label = torch.topk(scores_ts, 1, dim=1)[1].squeeze(1)
            if self.return_type == "class":
                return class_label.cpu()
            elif self.return_type == "bool":
                return (class_label == 1).int().cpu()
            elif self.return_type == "soft_bool":
                return (1 -  (class_label == 0).int()).cpu()
            elif self.return_type == "distribution":
                return scores_ts.cpu()

       
    def release_model(self):
        del self.model
        if self.model_name != "potsawee/deberta-v3-large-mnli":
            LLM.release_one(self.llm_name)

def process(a, tokenizer=None):
    if tokenizer is not None:
        a = sent_encode(tokenizer, a)
    return set(a)

def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    if nthreads > 0:
        with get_context("spawn").Pool(nthreads) as p:
            idf_count.update(chain.from_iterable(p.map(process_partial, arr)))
            p.close()
            p.join()
    else:
        idf_count.update(chain.from_iterable(map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update(
        {idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()}
    )
    return idf_dict




def calculate_nli_score(model, tokenizer, hyps, refs, batch_size=64):
    """
    calculate the element-wise entailment score of hyps and refs
    Parameters:
    model: nn.Module - nli model used for calculatiton
    tokenizer:
    hyps; List[str]
    refs: List[str]
    batch_size: int
    
    return:
    scores: List[dict()] - the key of each dict is "entailment", "neutral" and "contradiction",  value is the correponding normalized predicted probability.
    """
    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)
    
    concated_sents = [hyp + "[$usedforsep$]" + ref for hyp, ref in zip(hyps, refs)]

    sents = dedup_and_sort(concated_sents)
    compact_hyps, compact_refs = list(zip(*[s.split("[$usedforsep$]") for s in sents]))


    
    if "deberta-v2-xlarge-mnli" in  model.name_or_path.lower() or "deberta-xlarge-mnli" in model.name_or_path.lower():
        label_names = ["contradiction", "neutral", "entailment"]
    else:
        label_names = ["entailment", "neutral", "contradiction"]
    tokenized_input = tokenizer(compact_hyps, compact_refs, truncation=True)
    with torch.no_grad():
        data_collator = DataCollatorWithPadding(tokenizer,  pad_to_multiple_of=8 if model.dtype==torch.float16 else None)
        dataloader = DataLoader(Dataset.from_dict(tokenized_input.data), batch_size=batch_size, collate_fn=data_collator)
        compact_scores = []
        for batch in tqdm(dataloader):
            output = model(**batch.to(model.device))
            score = torch.softmax(output["logits"], dim=-1).tolist()
            if len(score[0]) == 2:
                label_names = [label_names[0], label_names[-1]]
            prediction = [{name: pred for pred, name in zip(s, label_names)} for s in score]
            compact_scores.extend(prediction)
    
    stat_dict = {k:v for k,v in zip(sents, compact_scores)}
    scores = [stat_dict[k] for k in concated_sents]
    
        
    return scores


