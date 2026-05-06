from .metrics import f1_score, rouge1_score, rouge2_score, rougeL_score, bleu_score, em_score, BemCalculator, SemSimCalculator,llm_evalutor
from .metrics.bem import CACHED_BEM_PATH 


from .utils import ResultSaver, load_datasets
import os
import pandas as pd
import json
import torch
import copy
import gc

from tqdm import tqdm


EVALUATOR_MAPPING = {
    "f1": f1_score,
    "em": em_score,
    "llm": llm_evalutor,
    "rouge1": rouge1_score,
    "rouge2": rouge2_score,
    "rougeL": rougeL_score,
    "bleu": bleu_score,
}

SUPPORTED_METRIC_NAMES = ["f1", "em", "llm", "rouge1", "rouge2", "rougeL", "bleu", "bem", "semantic_similarity"]






def metric_max_over_ground_truths(candidate, ground_truth, metric_func):
    scores_for_ground_truths = []
    if candidate == '':
        candidate = "None"
    for g in ground_truth:
        if g == '':
            score = 0.0
        else:
            score = metric_func(candidate, g)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class  Evaluator:

    def __init__(self, metrics=["em"], models_for_sim=None):
        """
        Parameters:
        metrics: str or List[str]. - metrics to evaluate
        models_for_sim: str of List[str] - models used for calculating semantic similarity, if a list is given, each model will used respectively, and the metric name will be formatted as "semantic/{model_name}"
        """
        os.environ["TOKENIZERS_PARALLELISM"]="false"
        self.evaluator_mapping =  copy.deepcopy(EVALUATOR_MAPPING)
        if isinstance(metrics, str):
            self.metrics = [metrics]
        else: 
            self.metrics = metrics

        for metric in metrics:
            assert metric in SUPPORTED_METRIC_NAMES, f"given metric '{metric}' is not supported, supported metrics are as follows:\n{json.dumps(SUPPORTED_METRIC_NAMES, indent=4)}"
        
        if "semantic_similarity" in self.metrics:
            assert models_for_sim is not None, "a model should be provided by 'model_for_similarity' for computing semantic simialarity"
            if isinstance(models_for_sim, str):
                self.sim_models = [models_for_sim]
            else:
                self.sim_models = models_for_sim
            
            self.metrics.remove("semantic_similarity")
            for m in self.sim_models:
                self.metrics.append("semantic/"+m)
        else:
            self.sim_models = None
        
    def evaluate(self, path_or_data, cached_score_path: str = None, cached_score = None, cached_score_names = None, verbose=True, batch_size=512, max_num=None, model_evaluated="default_model", device_name=None):
        """
        parameters:
        path_or_data: str or Dict -  the path or data itself.  data should be in the format of:
        {
        "queries": List[str],
        "responses": List[str],
        "ground_truth_answers": List[List[str]] or List[str]
        }
        cached_score_path: str - file path where save the cached metric score
        cached_score: dict - cached_score constructed by self.load_cached_score, if given, cached_score_path will be ignored.
        cached_score_name: List[str] - a list of metric names to be loaded, if None, all metrics in self.metrics will be loaded
        model_evaluated: str - name of model generating the evaluated answers, will be written in the saved files
        device_name: str - should be "cpu" or "gpu{rank}", if None, will detect available model. default=None
        """
        if device_name is None:
            if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                device_name = f"gpu{current_device}"

            else:
                device_name = "cpu"

        if cached_score is not None:
            if cached_score_names is None:
                metrics_to_load = set(self.metrics)
                
            else:
                metrics_to_load = set(self.metrics).intersection(set(cached_score_names))
            
            if metrics_to_load != set():
                if verbose:
                    print(f"metrics to be loaded from cached file are: {'|'.join(list(metrics_to_load))} ")
                self.cached_scores = cached_score


        elif cached_score_path is not None:
            assert os.path.exists(cached_score_path), f"given cached score path '{cached_score_path}' do not exists"
            if cached_score_names is None:
                metrics_to_load = set(self.metrics)
            else:
                metrics_to_load = set(self.metrics).intersection(set(cached_score_names.keys()))

            if metrics_to_load != set():
                if verbose:
                    print(f"metrics to be loaded from cached file are: {'|'.join(list(metrics_to_load))} ")
                
                self.load_cached_score(cached_score_path, metrics_to_load, verbose=verbose)
        else:
            metrics_to_load = set()
        

            
        self.questions, self.candidates, self.ground_truth = load_datasets(path_or_data)
       
        if max_num is not None:
            self.questions = self.questions[:max_num]
            self.candidates = self.candidates[:max_num]
            self.ground_truth = self.ground_truth[:max_num]
            
        if verbose:
            print(f"{device_name}:start to evaluate with metrics: {', '.join(self.metrics)}.")
        ground_truth = ["; ".join(gt) for gt in self.ground_truth]

        results = dict()

            
        for metric in self.metrics:
            
            if verbose:
                print(f"{device_name}:evaluate answer generated with metric {metric}")
            if metric in metrics_to_load:
                
                results.update(self.evaluate_one_metric_from_cache(self.questions, self.candidates, metric))
            else:
                if "semantic/" in metric:
                    m_name = metric.replace("semantic/", "")
                    if verbose:
                        print(f"start to load similairty evaluation model for {metric}")
                    scorer = SemSimCalculator(m_name, device_name=device_name)
                    if verbose:
                        print(f"{device_name}:similairty evaluation model loaded for {metric}")
                
                elif metric == "bem":
                     
                    scorer = BemCalculator(model_path=CACHED_BEM_PATH, device_name=device_name)

                else:
                    scorer = self.evaluator_mapping[metric]
                
                # print(f"before generation of metric {metric}:\nCurrent GPU Memory Usage: {float(get_gpu_memory())/1024:.2f} GB")

                results.update(self.evaluate_one_metric(self.questions, self.candidates, self.ground_truth, scorer, metric, batch_size=batch_size))
                
                if "semantic/" in metric or metric == "bem":
                    scorer.release_model()
                    del scorer
                    gc.collect()
                # print(f"after del model of metric {metric}:\nCurrent GPU Memory Usage: {float(get_gpu_memory())/1024:.2f} GB")

        print(f"{device_name}:evaluation finished!!!")
          # 如果没有匹配的知识，默认为 None
        # print(json.dumps(results, indent=4))
        results.update({"questions": self.questions, "answers": self.candidates, "ground_truth": ground_truth})
            
        self.result = ResultSaver({model_evaluated: results}, verbose=verbose)
        return self.result, results
        #return self.result

    def to_excel(self, dir_to_save, score_only=False, **kwargs):
        self.result.to_excel(dir_to_save, score_only=score_only, **kwargs)
    
    def to_dict(self, output_path, score_only=False, model_first = True):
        self.result.to_json(output_path, score_only=score_only, model_first=model_first)

    def evaluate_one_metric_from_cache(self, questions, candidates, metric, ):
        scores = []
        
        for q, c in zip(questions, candidates):
            key = self.key(q, c)
            
            scores.append(self.cached_scores[key][metric])
            
        return {metric: scores}
    
    def load_cached_score(self, path, metrics, scorer, verbose=True):
        """
        path: str - file path where save the cached metric score
        metric_mapping: dict - key is one of the supported metric name, value is the column name in file corresponding to the key
        """
        if verbose:
            print(f"load cached score from {path}")
        data = pd.read_excel(path)
        data.fillna("", inplace=True)
        data_dict = data.to_dict(orient="records")
        self.cached_scores = dict()

        for ex in data_dict:

            key = self.key(ex["questions"],ex["answers"])
            self.cached_scores[key] = dict()

            for c_name in metrics:

                self.cached_scores[key][c_name] = ex[c_name]
        
        return self.cached_scores


    def key(self, question, answer):
        #return question + "[sep]" + answer + "[sep]" + method
        return question.strip() + "[sep]" + answer.strip()



    def evaluate_one_metric(self, questions, candidates, ground_truths, score_func, metric, batch_size=512):
        """
        input:
        questions: List[str]
        candidates: List[str]
        ground_truths: List[List[str]]
        scorer: a function to compute the score
        metric: str

        return:
        dict{"{score_name}": scores (List[float])}
        """

        
        if metric in ["f1", "rouge1", "bleu", "em", "rouge2", "rougeL"]:
            scores = []
            for c, g in tqdm(zip(candidates, ground_truths)):
                
                score = metric_max_over_ground_truths(c, g, score_func)
                scores.append(score)
            return {metric: scores}
        elif metric in ["bleu"]:
            scores = []
            for c, g in tqdm(zip(candidates, ground_truths)):
                
                score = score_func(c, g)
                scores.append(score)
            return {metric: scores}
        elif metric == "bem":
            scores = []
            expand_qs = []
            expand_cs = []
            expand_gt = []
            index = [0]
            count_index = 0
            for q,c, gs in zip(questions, candidates, ground_truths):
                if c == "":
                    c = "None"
                for g in gs:
                    count_index += 1
                    expand_qs.append(q)
                    expand_cs.append(c)
                    expand_gt.append(g)
                index.append(count_index)
            examples = [{
                    'question': q,
                    'reference': g,
                    'candidate': c
                } for q, g, c in zip(expand_qs, expand_gt, expand_cs)]
            
        
            expand_scores = score_func(examples, batch_size=batch_size)
            scores = [max(expand_scores[index[i]: index[i+1]]) for i in range(len(index)-1)]

            return {metric: scores}
        elif "semantic/" in metric:
            scores = []
            expand_qs = []
            expand_cs = []
            expand_gt = []
            index = [0]
            count_index = 0
            for q,c, gs in zip(questions, candidates, ground_truths):
                if c == "":
                    c = "None"
                for g in gs:
                    count_index += 1
                    expand_qs.append(q)
                    expand_cs.append(c)
                    expand_gt.append(g)
                index.append(count_index)

            if "nli" not in metric.lower():
                prepend_text = None
            else:
                prepend_text = expand_qs
            expand_scores = score_func(expand_cs, expand_gt, prepend_text=prepend_text, batch_size=batch_size).tolist()
            scores = [max(expand_scores[index[i]: index[i+1]]) for i in range(len(index)-1)]
            return {metric: scores}
    
        else:
            score = EVALUATOR_MAPPING[metric](questions, ground_truths, candidates)
            return {metric: score}
    

