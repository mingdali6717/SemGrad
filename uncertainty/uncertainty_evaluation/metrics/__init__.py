from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
import numpy as np

def auroc_score(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    score = roc_auc_score(y_true, y_scores)
    return score


def risk_coverage_from_errors(errors, uncertainty):
    """
    Compute risk-coverage and selective accuracy curves 
    given per-sample error indicators and confidence scores.
    """
    errors = np.asarray(errors).astype(np.float64)
    uncertainty = np.asarray(uncertainty).astype(np.float64)
    # uncertainty:  larger = more uncertain
    idx = np.argsort(uncertainty)  # sort ascending
    errors_sorted = errors[idx]
    total_error = np.sum(errors_sorted)
    base_error = total_error / len(errors)

    n = len(errors)
    cum_errors = np.cumsum(errors_sorted)
    
    k = np.arange(1, n + 1)

    # risk = cumulative error rate among top-k confident predictions
    risk = cum_errors / k
    
    random_risk = k/n  * base_error
    oracle_risk = np.cumsum(np.sort(errors)) / k
    sel_acc = 1.0 - risk
    coverage = k / n

    return coverage, risk, sel_acc, random_risk, oracle_risk

def selective_accuracy_at_coverages(sel_acc, coverages=(0.95, 0.90, 0.80)):
    n = len(sel_acc)
    out = {}
    for c in coverages:
        k = max(1, int(np.floor(c * n)))
        out[c] = float(sel_acc[k-1])
    return out

def aurc_from_rc(risk):
    # risk is length-n array aligned with coverage k/n
    return float(risk.mean())

def prr_from_malinin(risk, random_risk, oracle_risk):
    AR_uns = random_risk - risk
    AR_orc = random_risk - oracle_risk
    return float(AR_uns.mean() / AR_orc.mean())


def coverage_scores(errors, uncertainty, coverages=(0.90, 0.80, 0.70)):
    _, risk, sel_acc, random_risk, oracle_risk = risk_coverage_from_errors(errors, uncertainty)
    aurc = aurc_from_rc(risk)
    sel_acc = selective_accuracy_at_coverages(sel_acc, coverages)
    prr = prr_from_malinin(risk, random_risk, oracle_risk)
    return aurc, sel_acc, prr
    
def acc_score(y_true, y_pred, is_label=False, threshold=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if not is_label:
        acc_list = []
        if threshold is None:
            thresholds = np.sort(y_pred)
        else:
            thresholds = [threshold]
        
        for t in thresholds:
            acc = accuracy_score(y_true, (y_pred >= t).astype(int))
            acc_list.append(acc)
        acc_score = np.max(np.array(acc_list))
        threshold = thresholds[np.argmax(np.array(acc_list))]
    else:
        acc_score = accuracy_score(y_true, y_pred)
        threshold = None
    
    return acc_score, threshold

def p_r_f1(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    auc_pr = auc(recall, precision)
    f1_list = []
    for p, r in zip(precision, recall):
        if (p + r) == 0:
            f1_list.append(0)
            print("divide zero error!")
            continue
        f1 = (2 * p * r) / (p + r)
        f1_list.append(f1)
    f1 = max(f1_list)
    p, r = list(zip(precision, recall))[np.argmax(np.array(f1_list))]

    
    
    return {"precision": p, "recall": r, "f1": f1, "aucpr": auc_pr}
# y_true = [1, 0, 0, 0, 1, 0, 1, 0]  # 真实标签
# y_scores = [91, 0.8, 0.3, 0.1, 0.4, 91, 0.66, 0.7] # 模型预测为正类的概率
# auroc_score = auroc_score(y_true, y_scores)

# print("AUROC score:", auroc_score)

# y_true = [0, 0, 1, 1]  # 真实标签
# y_scores = [0.1, 0.4, 0.35, 0.8] # 模型预测为正类的概率
# acc_score = max_f1(y_true, y_scores)

# print("f1 score:", acc_score)
