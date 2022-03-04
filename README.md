# Fine-tuning of 'DistillBert' NLP transformer model for text classification, with 'AdaFamily' optimizer and 'mini-batch trimming'

Demonstrates finetuning of a NLP model with novel 'AdaFamily' optimizer and 'mini-batch trimming'.
Code is taken and adapted from https://github.com/hfwittmann/transformer_finetuning_lightning
Uses pytorch lightning. Demonstrates also how to modify (via 'monkey-patching') a huggingface transformer model so that it employs a _custom_ loss function.

Regarding 'mini-batch trimming' (curriculum learning method), see my arxiv preprint at https://arxiv.org/abs/2110.13058

Regarding 'AdaFamily' (a family of novel adaptive gradient methods), see my arxiv preprint at https://arxiv.org/abs/2203.01603
We use the AdaFamily variant with myu = 0.25
