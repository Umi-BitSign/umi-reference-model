"""Explicit device selection for the author demo compatibility path."""
import os
import torch

def selected_device() -> torch.device:
    requested = os.environ.get('SHUBERT_DEVICE', 'auto')
    if requested == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    if requested == 'mps' and (not torch.backends.mps.is_available()):
        raise RuntimeError('MPS explicitly requested but unavailable')
    if requested == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA explicitly requested but unavailable')
    if requested not in {'cpu', 'mps', 'cuda'}:
        raise ValueError(f'unsupported SHUBERT_DEVICE={requested}')
    return torch.device(requested)
