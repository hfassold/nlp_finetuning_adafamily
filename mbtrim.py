# Functions related to 'mini-batch trimming'

import torch

# Get a linear mapping from interval [A, B] to interval [a, b], and evaluate this mapping fn. for value 'val'
def linear_interval_mapping_eval_at(val, A, B, a, b):
    # taken from https://stackoverflow.com/questions/12931115/algorithm-to-map-an-interval-to-a-smaller-interval/45426581
    return (val - A) * (b - a) / (B - A) + a

def get_adapted_loss_for_trimmed_minibatch(loss, current_epoch, max_epochs, mbt_a, mbt_b, mbt_epoch_scheme):
    # Returns the 'trimmed' loss, containing only the samples of the mini-batch with the _highest_ loss
    # Parameter 'loss' must be a vector containing the per-sample loss for all samples in the (original) minibatch
    minibatch_size = loss.size()[0]
    if minibatch_size == 1:
        # If this line triggers an error, you might have forgotten to set parameter 'reduction' to 'none' when calling the pytorch loss function
        raise ValueError("mbtrim.get_adapted_loss_for_trimmed_minibatch: mini-batch size must be > 1")
    if mbt_epoch_scheme == 'constant':
        # r is set to 'average(mbt_a, mbt_b) * M'
        r = (0.5 * (mbt_a * mbt_b)) * minibatch_size
    elif mbt_epoch_scheme == 'linear':
        # r varies according to a linear function which maps interval [0, max_epochs - 1] to interval [mbt_a * M, mbt_b * M]
        r = linear_interval_mapping_eval_at(current_epoch, 0, max_epochs - 1, mbt_a * minibatch_size, mbt_b * minibatch_size)
    else:
        raise ValueError("mbtrim.get_adapted_loss_for_trimmed_minibatch: Unsupported value for parameter: mbt_epoch_scheme")
    # round r to integer, safeguard if r is 0
    r = max(round(r), 1)
    # The 'topk' function does the actual trimming of the minibatch:
    # it returns the loss for the 'r' samples with the _highest_ loss in the minibtach
    # See documentation at https://pytorch.org/docs/stable/generated/torch.topk.html
    # Note the 'topk' operation is differentiable, see https://stackoverflow.com/questions/67570529/derive-the-gradient-through-torch-topk
    # and https://math.stackexchange.com/questions/4146359/derivative-for-masked-matrix-hadamard-multiplication
    # and https://discuss.pytorch.org/t/backward-through-topk-operation-on-variable/9197
    loss_trimmed = torch.topk(loss, r, sorted = False, dim = 0)[0]
    # return it
    return loss_trimmed
