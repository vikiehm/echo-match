import torch
import random
import numpy as np
from contextlib import contextmanager

@contextmanager
def temp_seed(seed):
    """
    Temporarily sets the seed for torch, CUDA, numpy, and the Python random module.
    Saves the current states and then restores them after the context block is executed.
    """
    # Save current states
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    random_state = random.getstate()
    np_state = np.random.get_state()
    
    try:
        # Set seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        # Restore the saved states
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)
        random.setstate(random_state)
        np.random.set_state(np_state)