import copy
import numpy as np
import random
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from utils import make_mask

class TrainDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer, block_size, wwm):
        block_size -= tokenizer.num_special_tokens_to_add(pair=True)

        self.examples = []
        num_discarded = 0
        fsrc = open(src_path, encoding='utf-8')
        ftgt = open(tgt_path, encoding='utf-8')
        for i, (__text1, __text0) in enumerate(zip(fsrc, ftgt)):
            if i % 1000 == 0:
                print('%d examples loaded' % i)
            text0 = __text0.strip()
            text1 = __text1.strip()
            tok0 = tokenizer.tokenize(text0)
            tok1 = tokenizer.tokenize(text1)
            id0 = tokenizer.convert_tokens_to_ids(tok0)
            id1 = tokenizer.convert_tokens_to_ids(tok1)
            if len(id0) + len(id1) > block_size:
                num_discarded += 1
                continue
            input_ids = tokenizer.build_inputs_with_special_tokens(id0, id1)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(id0, id1)
            attention_mask = [1] * len(input_ids)
            if wwm:
                mask_ids = make_mask(text0, tok0, len(input_ids))
            else:
                mask_ids = list(range(len(input_ids)))
            self.examples.append({
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'mask_ids': mask_ids,
            })
        fsrc.close()
        ftgt.close()
        print('Due to length limits, %d examples discarded' % num_discarded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

class TrainCollator:
    def __init__(self, tokenizer, mlm_probability):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, examples):
        inputs = {}
        for key in examples[0]:
            inputs[key] = []
            for e in examples:
                inputs[key].append(e[key])
            inputs[key] = pad_sequence([torch.from_numpy(np.array(x)) for x in inputs[key]], batch_first=True, padding_value=0).long()
        masked_input_ids, labels = self.mask_tokens(inputs['input_ids'], inputs['token_type_ids'], inputs['mask_ids'])
        inputs['input_ids'] = masked_input_ids
        inputs['labels'] = labels
        del(inputs['mask_ids'])
        return inputs

    def mask_tokens(self, inputs, token_type_ids, mask_ids):
        labels = inputs.clone()
        
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        token_type_mask = token_type_ids.eq(1)
        probability_matrix.masked_fill_(token_type_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = torch.gather(masked_indices, 1, mask_ids)
        labels[~masked_indices] = -100

        indices_replaced = torch.gather(masked_indices, 1, mask_ids)
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

class EvalDataset(Dataset):
    def __init__(self, src_path, tgt_path, tokenizer, block_size, wwm, N, M):
        block_size -= tokenizer.num_special_tokens_to_add(pair=True)
        #n_predicts = [((i + 1) * N) // M - (i * N) // M  for i in range(0, M)]

        self.examples = []
        num_discarded = 0
        num_truncated = 0
        fsrc = open(src_path, encoding='utf-8')
        ftgt = open(tgt_path, encoding='utf-8')
        for i, (__text1, __text0) in enumerate(zip(fsrc, ftgt)):
            if i % 1000 == 0:
                print('%d examples loaded' % i)
            text0 = __text0.strip()
            text1 = __text1.strip()
            tok0 = tokenizer.tokenize(text0)
            tok1 = tokenizer.tokenize(text1)
            id0 = tokenizer.convert_tokens_to_ids(tok0)
            id1 = tokenizer.convert_tokens_to_ids(tok1)
            len_0 = len(id0)
            len_1 = len(id1)
            if len_0 > block_size:
                num_discarded += 1
                continue
            elif len_0 + len_1 > block_size:
                num_truncated += 1
                id1 = id1[ : block_size - len_0]
            __input_ids = tokenizer.build_inputs_with_special_tokens(id0, id1)
            __labels = [-100] * len(__input_ids)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(id0, id1)
            attention_mask = [1] * len(__input_ids)
            if wwm:
                mask_idx = make_mask(text0, tok0, len(__input_ids))[1 : len_0 + 1]
                mask_order = list(set(mask_idx))
            else:
                mask_idx = list(range(1, len_0 + 1))
                mask_order = list(range(1, len_0 + 1))
            len_m = len(mask_order)
            mask_orders = []
            for i in range(0, N):
                mask_orders.append([])
            for x in mask_order:
                A = random.sample(range(0, N), M)
                for a in A:
                    mask_orders[a].append(x)
            for i in range(0, N):
                input_ids = copy.deepcopy(__input_ids)
                labels = copy.deepcopy(__labels)
                for __idx in mask_orders[i]:
                    idx = __idx
                    while (idx <= len_0) and (mask_idx[idx - 1] == __idx):
                        labels[idx] = __input_ids[idx]
                        input_ids[idx] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
                        idx += 1
                self.examples.append({
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                })
        fsrc.close()
        ftgt.close()
        print('Due to length limits, %d examples discarded' % num_discarded)
        print('Due to length limits, %d examples truncated' % num_truncated)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def eval_collate_fn(examples):
    inputs = {}
    for key in examples[0]:
        inputs[key] = []
        for e in examples:
            inputs[key].append(e[key])
        padding_value = 0
        if key == 'labels':
            padding_value = -100
        inputs[key] = pad_sequence([torch.from_numpy(np.array(x)) for x in inputs[key]], batch_first=True, padding_value=padding_value).long()
    return inputs
