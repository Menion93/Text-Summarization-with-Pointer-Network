from rouge import Rouge
import numpy as np

r = Rouge()

def mmap(fn, elem):
    return list(map(fn, elem))

def rouge_score(y, y_):
    true_tok = [' '.join(mmap(str,sent)) for sent in y]
    pred_tok = [' '.join(mmap(str,sent)) for sent in y_]
    scores = r.get_scores(true_tok, pred_tok)

    values = []
    for key in scores[0].keys():
        for sub_metric in ['p', 'r', 'f']:
            mean_score = np.mean(mmap(lambda x: x[key][sub_metric], scores))
            values.append(mean_score)
    return values