import os
import numpy as np
import torch
import torch_xla.core.xla_model as xm

def check_xla_bf16_flags():
    return os.getenv("XLA_USE_BF16") == '1' or os.getenv("XLA_DOWNCAST_BF16") == '1'

def get_seed(dropout_p, device):
    if dropout_p > 0.0:
        seed = np.array([xm.get_rng_state()]).astype(np.int32)
        return torch.from_numpy(seed).to(device)
    return None

def move_seed(dropout_p):
    if dropout_p > 0.0:
        orig_seed = xm.get_rng_state()
        running_seed = (orig_seed * 214013 + 2531011) & 0xFFFFFFFFFFFFFFFF
        xm.set_rng_state(int(running_seed))

def permute(q, k, v):
    q, k, v = [t.permute(0, 1, 3, 2) for t in (q, k, v)]
    return q, k, v

def cast(q, k, v):
    if check_xla_bf16_flags():
        q, k, v = [t.to(torch.bfloat16) for t in (q, k, v)]
    elif q.dtype == torch.bfloat16 and os.getenv("XLA_DOWNCAST_BF16") == '0':
        """In mixed_precision (XLA_DOWNCAST_BF16) we cast the q,k,v to bf16,
        which is a no-op in a way because q,k,v are already in BF16. In the torch land, 
        the q,k,v tensors show up as FP32 (but in XLA due to flags, these tensors are in BF16), 
        so the cast operation to bf16 in the torch land is an operation which will lead to a 
        new memory (going from fp32 to bf16). But in XLA it is a no-op. cause. Changing the 
        cast to torch.float32 in mixed_precision (XLA_DOWNCAST_BF16) flags, actual dtype of q,k,v
        is unaffected, but it ends up taking more memory. So removing the cast leads to more memory. 
        A possible reason this is happening is because of the NKI kernel tracing issues (it might not be the 
        true root cause), When autocast is enabled, the same extra memory (exactly same as what was 
        mentioned previously) was seen, creating a new tensor just before passing to the kernel, tried to
        emulate a no-op cast using the clone here. It helped to reduce the memory. So now with this change, 
        the before(mixed_precision) and after(autocast), memory is the same.
        
        This can be removed in future if the true root cause is determined."""

        q, k, v = [t.clone() for t in (q, k, v)]
    return q, k, v