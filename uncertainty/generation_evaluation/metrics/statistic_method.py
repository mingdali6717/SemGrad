from __future__ import print_function

from rouge import Rouge
import argparse
import json
import re
import string
import sys
import unicodedata
from collections import Counter
import sacrebleu



def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def bleu_score(candidate, references):
    """
    candidate: str
    references: List[str]
    """
    if isinstance(references, str):
        references = [references]
    bleu_score = sacrebleu.corpus_bleu(candidate, references)
    return bleu_score.score

def rouge1_score(candidate, reference):
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(normalize_answer(candidate), normalize_answer(reference))
        return rouge_scores[0]["rouge-1"]["f"]
    except ValueError:
        return 0
    
    

def rouge2_score(candidate, reference):
    # Calculate ROUGE scores
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(normalize_answer(candidate), normalize_answer(reference))
        return rouge_scores[0]["rouge-2"]["f"]
    except ValueError:
        return 0
    
    

def rougeL_score(candidate, reference):
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(normalize_answer(candidate), normalize_answer(reference))
        return rouge_scores[0]["rouge-l"]["f"]
    except ValueError:
        return 0
    

def em_score(candidate, reference):
    candidate = normalize_answer(candidate)
    reference = normalize_answer(reference)
    pattern = r'\b' + re.escape(reference) + r'\b'
    if re.search(pattern, candidate):
        return 1
    else:
        return 0
