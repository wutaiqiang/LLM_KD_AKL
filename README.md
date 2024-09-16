# AKL

This is the offcial github for the paper [Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2404.02657).

# Toy Examples

To reproduce the toy examples, you can refer to the toy_examples/FR_KL.ipynb and toy_examples/FR_compare.ipynb

# KD Experiments

Please follow the [minillm](https://github.com/microsoft/LMOps/tree/main/minillm) for the environment and dataset.

Introduce the AKL into the KD setting. (Mainly on this [line](https://github.com/microsoft/LMOps/blob/1d6ca760f2f8b712d85bdefae67518c140b8a4a5/minillm/finetune.py#L166))

And then run the experiments and evaluate the student. For results on Winogrande, OpenBookQA, BoolQ, ARC, please use [this tool](https://github.com/EleutherAI/lm-evaluation-harness).


