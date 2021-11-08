import numpy as np
import torch

from torch.nn import CrossEntropyLoss
from utils import get_n_subwords

def predict(eval_dataloader, model, device, tokenizer, N=7, M=1, mc_dropout=False):
    if mc_dropout:
        model.train()
    else:
        model.eval()
    
    loss_func = CrossEntropyLoss(reduce=False)
    with torch.no_grad():
        preds = []
        preds_prob = []
        for b, inputs in enumerate(eval_dataloader):
            output = model(
                input_ids=inputs['input_ids'].to(device),
                token_type_ids=inputs['token_type_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                labels=inputs['labels'].to(device),
            )
            logits = output.logits.permute(0, 2, 1)
            labels = inputs['labels'].to(device)
            L = list(loss_func(logits, labels).cpu().numpy())
            I = list(inputs["input_ids"].numpy())
            pos_sep = []
            for s in I:
                for i, t in enumerate(s):
                    if t == tokenizer.sep_token_id:
                        pos_sep.append(i)
                        break
            for i, l in enumerate(L):
                preds_prob.append(-np.exp(-l[1 : pos_sep[i]]))
                preds.append(l[1 : pos_sep[i]])
            if b % 10 == 0:
                print('Finished batch %d' % b)
        
        preds_final = []
        for i, p in enumerate(preds):
            if i % N == 0:
                preds_final.append(p)
            else:
                preds_final[-1] += p
        for p in preds_final:
            p /= float(M)

        preds_prob_final = []
        for i, p in enumerate(preds_prob):
            if i % N == 0:
                preds_prob_final.append(p)
            else:
                preds_prob_final[-1] += p
        for p in preds_prob_final:
            p /= float(M)

    model.train()
    model.zero_grad()
    return preds_final, preds_prob_final

def f1(hit, pred, gold):
    if hit == 0:
        return 0.0
    precision = float(hit) / pred
    recall = float(hit) / gold
    return 2 * precision * recall / (precision + recall)

def make_word_outputs_final(word_outputs, input_filename, tokenizer, threshold=None, threshold_tune=None):
    assert((threshold is not None) or (threshold_tune is not None))
    
    fin = open(input_filename, 'r', encoding='utf-8')
    inputs = [x.strip() for x in fin]
    fin.close()
    
    word_scores_final = []
    for x, w in zip(inputs, word_outputs):
        z = []
        n_subwords = get_n_subwords(x, tokenizer)
        start = 0
        for n in n_subwords:
            end = start + n
            z.append(float(np.mean(w[start : end])))
            start = end
        word_scores_final.append(z)
    
    if threshold_tune is not None:
        flabels = open(threshold_tune, 'r')
        labels = [x.strip().split(' ')[1 : : 2] for x in flabels]
        flabels.close()
        
        S = []
        L = []
        for s, l in zip(word_scores_final, labels):
            L += [int(x == 'OK') for x in l]
            S += s
        assert(len(L) == len(S))
        A = list(zip(S, L))
        A.sort()
        
        gold_ok = int(np.sum(L))
        gold_bad = len(L) - gold_ok
        pred_bad = len(L)
        pred_ok = 0
        hit_bad = gold_bad
        hit_ok = 0
        best_f1_mul = 0.0
        threshold = -1e19
        for s, l in A:
            pred_ok += 1
            pred_bad -= 1
            if l == 1:
                hit_ok += 1
            else:
                hit_bad -= 1
            f1_bad = f1(hit_bad, pred_bad, gold_bad)
            f1_ok = f1(hit_ok, pred_ok, gold_ok)
            f1_mul = f1_bad * f1_ok
            if f1_mul > best_f1_mul:
                best_f1_mul = f1_mul
                best_f1_bad = f1_bad
                best_f1_ok = f1_ok                                
                threshold = s 
        print('F1_BAD: %.6f, F1_OK: %.6f, F1_MUL: %.6f' % (best_f1_bad, best_f1_ok, best_f1_mul))
    else:
        best_f1_mul = 0.0
    
    word_outputs_final = []
    for s in word_scores_final:
        word_outputs_final.append(['BAD' if x >= threshold else 'OK' for x in s])
    return word_scores_final, word_outputs_final, threshold, best_f1_mul
