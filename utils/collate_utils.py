from torch.utils.data import default_collate
from collections.abc import Mapping, Sequence
import torch
def custom_collate_fn(batch):
    '''
    Custom default collate function just to handle sparse tensors by not handling them.
    CAUTION: for proper batch operation with diffusion net(unimplemented), need to do them better
    '''
    elem = batch[0]
    if isinstance(elem, torch.Tensor) and elem.is_sparse  :
        return elem.unsqueeze(0) # very lazy way: we just assume batch size is always 1 for sparse
    elif isinstance(elem, str):  # Strings -> List
        return batch  # Just return as a list
    elif isinstance(elem, Mapping):  # Dict -> Recursively collate per key
        return {key: custom_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, Sequence):  # List/Tuple -> Recursively collate
        return [custom_collate_fn(items) for items in zip(*batch)]
    else: # dense tensor, np, int, float, bool
        return default_collate(batch)