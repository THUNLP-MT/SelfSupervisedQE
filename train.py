import argparse
import numpy as np
import os
import pandas as pd
import time
import torch

from data import (
    eval_collate_fn,
    EvalDataset,
    TrainCollator,
    TrainDataset,
)
from evaluate import predict, make_word_outputs_final
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--train-src', type=str, required=True)
parser.add_argument('--train-tgt', type=str, required=True)

parser.add_argument('--dev-src', type=str, required=True)
parser.add_argument('--dev-tgt', type=str, required=True)
parser.add_argument('--dev-hter', type=str)
parser.add_argument('--dev-tags', type=str)

parser.add_argument('--block-size', type=int, default=256)
parser.add_argument('--eval-block-size', type=int, default=512)
parser.add_argument('--wwm', action='store_true')
parser.add_argument('--mlm-probability', type=float, default=0.15)

parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--update-cycle', type=int, default=16)
parser.add_argument('--eval-batch-size', type=int, default=4)
parser.add_argument('--train-steps', type=int, default=100000)
parser.add_argument('--eval-steps', type=int, default=1000)
parser.add_argument('--learning-rate', type=float, default=5e-5)

parser.add_argument('--pretrained-model-path', type=str, required=True)
parser.add_argument('--save-model-path', type=str, required=True)

parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()
print(args)

set_seed(args.seed)
device = torch.device('cuda')
torch.cuda.set_device(0)

config = AutoConfig.from_pretrained(args.pretrained_model_path, cache_dir=None)
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, cache_dir=None, use_fast=False, do_lower_case=False)

model = AutoModelWithLMHead.from_pretrained(args.pretrained_model_path, config=config, cache_dir=None)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

train_dataset = TrainDataset(
    src_path=args.train_src,
    tgt_path=args.train_tgt,
    tokenizer=tokenizer,
    block_size=args.block_size,
    wwm=args.wwm,
)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=TrainCollator(tokenizer=tokenizer, mlm_probability=args.mlm_probability),
)

dev_dataset = EvalDataset(
    src_path=args.dev_src,
    tgt_path=args.dev_tgt,
    tokenizer=tokenizer,
    block_size=args.eval_block_size,
    wwm=args.wwm,
    N=7,
    M=1,
)
dev_dataloader = DataLoader(
    dataset=dev_dataset,
    batch_size=args.eval_batch_size,
    shuffle=False,
    collate_fn=eval_collate_fn,
)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
            0.01
    },
    {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.train_steps)

dirs = ['checkpoint_best', 'checkpoint_last']
files_to_copy = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']
for d in dirs:
    os.system('mkdir -p %s' % os.path.join(args.save_model_path, d))
    for f in files_to_copy:
        os.system('cp %s %s' % (
            os.path.join(args.pretrained_model_path, f),
            os.path.join(args.save_model_path, d, f)
        ))
print('Configuration files copied')

def save_model(model, save_dir):
    print('Saving Model to', save_dir)
    if os.path.exists(save_dir):
        print('%s already exists. Removing it...' % save_dir)
        os.remove(save_dir)
        print('%s removed successfully.' % save_dir)
    torch.save(model.state_dict(), save_dir)
    print('%s saved successfully.' % save_dir)

total_minibatches = len(train_dataloader)
best_score = 0.0
num_steps = 1
model.train()
model.zero_grad()

epoch = 1
total_loss = 0.0
current_time = time.time()
while True:
    for i, inputs in enumerate(train_dataloader):
        n_minibatches = i + 1

        output = model(
            input_ids=inputs['input_ids'].to(device),
            token_type_ids=inputs['token_type_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            labels=inputs['labels'].to(device),
        )
        loss = output.loss / float(args.update_cycle)
        total_loss += float(loss)
        loss.backward()
        
        if (n_minibatches == total_minibatches) or (n_minibatches % args.update_cycle == 0):
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()
            old_time = current_time
            current_time = time.time()
            print('epoch = %d, step = %d, loss = %.6f (%.3fs)' % (epoch, num_steps, total_loss, current_time - old_time))
            
            if (num_steps == args.train_steps) or (num_steps % args.eval_steps == 0):
                print('Evaluating...')
                preds = predict(
                    eval_dataloader=dev_dataloader,
                    model=model,
                    device=device,
                    tokenizer=tokenizer,
                    N=7,
                    M=1,
                    mc_dropout=False,
                )
                if args.dev_tags is not None:
                    assert(args.dev_hter is None)
                    eval_score = make_word_outputs_final(preds, args.dev_tgt, tokenizer, threshold_tune=args.dev_tags)[-1]
                else:
                    sent_outputs = pd.Series([float(np.mean(w)) for w in preds])
                    fhter = open(args.dev_hter, 'r', encoding='utf-8')
                    hter = pd.Series([float(x.strip()) for x in fhter])
                    fhter.close()
                    eval_score = float(sent_outputs.corr(hter))
                print('Validation Score: %.6f, Previous Best Score: %.6f' % (eval_score, best_score))
                
                if eval_score > best_score:
                    save_model(model, os.path.join(args.save_model_path, 'checkpoint_best/pytorch_model.bin'))
                    best_score = eval_score
                save_model(model, os.path.join(args.save_model_path, 'checkpoint_last/pytorch_model.bin'))
                
            if num_steps >= args.train_steps:
                exit(0)
            num_steps += 1
            total_loss = 0.0
    epoch += 1
