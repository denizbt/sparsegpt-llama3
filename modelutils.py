import torch
import torch.nn as nn
import transformers


if torch.cuda.is_available():
    DEV = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    DEV = torch.device('mps')
else:
    DEV = torch.device('cpu')


def empty_cache(dev):
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    elif dev.type == 'mps':
        torch.mps.empty_cache()

def find_layers(module, layers=None, name=''):
    if layers is None:
        # GPT-2 stores its dense projections as Hugging Face Conv1D modules.
        # SparseGPT already knows how to transpose their weights; make sure the
        # model traversal can discover them as well.
        layers = [nn.Conv2d, nn.Linear, transformers.pytorch_utils.Conv1D]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
