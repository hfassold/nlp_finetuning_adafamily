# FAH
# These are my 'monkey patches' for modifying the 'forward' member function of certain HuggingFace models
# WITHOUT inheriting from them.
# Regarding the technique of monkey patching, see our WIKI at https://digital-wiki.joanneum.at/pages/viewpage.action?pageId=98697788
# or google for 'python monkey patching'

import torch
from torch import nn
from torch import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader

import math

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.modeling_outputs import SequenceClassifierOutput

# class is in file '<python_root>>\Lib\site-packages\transformers\models\distilbert\modeling_distilbert.py'
from transformers import DistilBertForSequenceClassification

# Taken from respective 'forward' fn. in class 'DistilBertForSequenceClassification' (in 'transformers' package)
# and adapted so that it additionally supports the config parameter 'self.jrs_do_loss_reduction'.
# Note I cannot change the signature of this function (add a respective parameter), therefore we rely on
# the presence of member variable 'self.jrs_do_loss_reduction' (which must have been set properly)
def forwardDistilBertForSequenceClassificationPatched(
    self,
    input_ids=None,
    attention_mask=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None
):
    r"""
    labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
        Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
        If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    distilbert_output = self.distilbert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
    pooled_output = hidden_state[:, 0]  # (bs, dim)
    pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
    pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
    pooled_output = self.dropout(pooled_output)  # (bs, dim)
    logits = self.classifier(pooled_output)  # (bs, num_labels)

    loss = None
    if labels is not None:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            # FAH modified
            if self.jrs_do_loss_reduction == True:
                loss_fct = CrossEntropyLoss()
            else:
                # we do no reduction (via 'mean'), so for each sample of the minibatch the loss value is returned
                loss_fct = CrossEntropyLoss(reduction = 'none')
            # ~FAH
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

    if not return_dict:
        output = (logits,) + distilbert_output[1:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=distilbert_output.hidden_states,
        attentions=distilbert_output.attentions,
    )

