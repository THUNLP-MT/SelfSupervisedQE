def make_mask(sent, tokens, total_sent_length):
    sent_len = len(tokens)
    mask_ids = list(range(total_sent_length))
    orig_tokens = sent.split()
    j = 0
    for t in orig_tokens:
        lt = len(t)
        curr_len = 0
        curr_token = ''
        old_j = j
        while curr_len < lt:
            T = tokens[j]
            if '##' in T:
                T = T[2 : ]
            curr_len += len(T)
            curr_token += T
            j += 1
        assert(curr_token == t)
        for k in range(old_j, j):
            mask_ids[k + 1] = old_j + 1
    return mask_ids

def get_n_subwords(sent, tokenizer):
    tokens = tokenizer.tokenize(sent)
    n_tokens = len(tokens)
    mask_ids = make_mask(sent, tokens, n_tokens + 1)[1 : ]
    n_subwords = []
    for i in range(n_tokens):
        if (i == 0) or (mask_ids[i] != mask_ids[i - 1]):
            n_subwords.append(0)
        n_subwords[-1] += 1
    return n_subwords