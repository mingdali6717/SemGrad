from .metrics import acc_score, auroc_score, p_r_f1, coverage_scores
from collections import defaultdict
from tabulate import tabulate
import os
import pandas as pd
from loguru import logger

UNCERTAINTY_METRICS_MAPPING = {
    "auroc": auroc_score,
    "accuracy": acc_score,
    "precision": p_r_f1,
    "recall": p_r_f1,
    "aucpr": p_r_f1,
    "f1": p_r_f1,
    "coverage": coverage_scores,
}

class Uncertainty_Evaluator:
    def __init__(self, metrics=["auroc"]):
        if metrics is None:
            self.metrics_to_evaluate = list(UNCERTAINTY_METRICS_MAPPING.keys())
        else:
            if isinstance(metrics, str):
                metrics = [metrics]
            self.metrics_to_evaluate = metrics
        
        self.result = None
    
    def evaluate(self, prediction_scores, labels, verbose=True, coverages=(0.9, 0.8, 0.7)):
        if isinstance(prediction_scores, list):
            prediction_scores = {"method": prediction_scores}
        methods = []
        scores = defaultdict(list)
        for m, pred_scores in prediction_scores.items():
            if verbose:
                logger.info(f"start to evaluate the performance of method '{m}'")
            methods.append(m)
            if set(self.metrics_to_evaluate).intersection({"precision", "recall", "f1", "aucpr"}):
                p_r_scores = p_r_f1(labels, pred_scores)
            for metric in self.metrics_to_evaluate:
                if metric in ["precision", "recall", "f1", "aucpr"]:
                    scores[metric].append(p_r_scores[metric])
                elif metric == "coverage":
                    aurc, _, _ = UNCERTAINTY_METRICS_MAPPING[metric](labels, pred_scores, coverages=coverages)
                    scores["aurc"].append(aurc)
                    # scores["prr"].append(prr)
                    # for c in coverages:
                    #     scores[f"selective_accuracy_at_{c:.2f}"].append(sel_acc[c])
                else:
                    scores[metric].append(UNCERTAINTY_METRICS_MAPPING[metric](labels, pred_scores))
        
        score_df = pd.DataFrame.from_dict(scores, orient='index', columns=methods).T
        self.result = score_df
        if verbose:
            print(tabulate(score_df, headers='keys', tablefmt='psql'))
        return self.result
    
    def to_excel(self, output_dir, name=None, **kwargs):
        if name is None:
            file_name = "error_detection_scores.xlsx"
        else:
            file_name = name + "_" + "error_detection_scores.xlsx"
        if os.path.isdir(output_dir):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            save_file = os.path.join(output_dir, file_name)
        else:
            assert output_dir.endswith(".xlsx"), f"the given file shoud end with .xlsx, but the path '{output_dir}' is given"
            save_file = output_dir
        
        logger.info(f"save result to {save_file}")
        
        self.result.to_excel(save_file, **kwargs)
        
    