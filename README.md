# SelfSupervisedQE

Self-Supervised Quality Estimation for Machine Translation

Please cite:

```
@inproceedings{Zheng:2021:EMNLP,
    title = "Self-Supervised Quality Estimation for Machine Translation",
    author = "Zheng, Yuanhang and Tan, Zhixing and Zhang, Meng and Maimaiti, Mieradilijiang and Luan, Huanbo and Sun, Maosong and Liu, Qun and Liu, Yang",
    booktitle = "EMNLP",
    year = "2021"
}
```

## Requirements

Python 3.6

torch>=1.4.0

transformers>=4.4.2

pandas>=1.0.5

Note that we should slightly modify the file `tokenization_bert.py` in Transformers library to normally run the code, therefore please copy the file `tokenization_bert.py` in this repository to replace the corresponding file in the Transformer library.

## Usage

Data download: please refer to the script `download_data.sh`.

Training: please refer to the script `train.sh`.

Inference: please refer to the script `predict.sh`.

## Contact

If you have questions, suggestions and bug reports, please email [zyh971025@126.com](mailto:zyh971025@126.com).