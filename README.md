# SelfSupervisedQE

Self-Supervised Quality Estimation for Machine Translation

## Requirements

Python 3.6

torch>=1.4.0

transformers>=4.4.2

pandas>=1.0.5

Note that we should slightly modify the file `tokenization_bert.py` in Transformers library to normally run the code, therefore please copy the file `tokenization_bert.py` in this repository to replace the corresponding file in the Transformer library.

## Usage

Training: please refer to the script `train.sh`.

Inference: please refer to the script `predict.sh`.