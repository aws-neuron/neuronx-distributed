import numpy as np
from neuronxcc.starfish.support.dtype import as_native_type
import torch
from torch import bfloat16


def isclose(a, b, *, rtol=1e-05, atol=1e-08, equal_nan=False, equal_inf=False,
            mode='max'):
    """Return a boolean array indicating where `a` and `b` are close within
    the specified tolerance.

    This function is similar to numpy's `isclose`, but has lower sensitivity
    due to considering overall value range.

    See Issue: NCC-181 for the discussion that lead to
    this algorithm.

    Two numbers x and y are considered close if |x-y| <= atol + rtol * brange.

    In the original numpy algorithm, brange is the absolute value of y. That is,
    brange is different for each element of `a` and `b`. That algorithm can
    still be used by choosing `mode='py'`. However, when `mode='max'` (the
    default), brange will be the maximum absolute value of any element in `b`.
    Thus it doesn't vary for each comparison.

    Parameters
    ----------
    a, b: input tensors to compare. b is considered the "golden" value against
    which we are computing.

    rtol: relative tolerance (float)

    atol: absolute tolerance (float)

    equal_nan: if True, nans will be considered close (default is False)

    equal_inf: if True, infinities of the same sign will be considered close

    mode: 'py' for the usual numpy algorithm; 'max' (the default) for the
    less sensitive version

    verbose: 0 to print nothing; >= 1 to print a summary of differences;
    >=3 to print information about the difference at each failed comparison.

    Returns
    -------
    boolean array
    """
    # int 64-> int32
    if b.dtype.itemsize == 2 * a.dtype.itemsize:
        b = b.astype(a.dtype)
    assert a.shape == b.shape

    assert mode == 'max' or mode == 'py', 'Unknown isclose mode {}'.format(mode)

    brange = abs(b)
    if mode == 'max':
        brange = np.amax(brange[~np.isnan(brange)])

    with np.errstate(invalid='ignore'):
        abs_diff = abs(a.astype(float) - b.astype(float))
        close = (abs_diff <= atol + rtol * brange) & np.isfinite(a) & np.isfinite(b)
    if equal_inf:
        close |= (a == b)
    if equal_nan:
        close |= (np.isnan(a) & np.isnan(b))

    largest_abs_diff = np.amax(abs_diff)
    # If all abs_diff < atol, let maxAbsDiffElementRelDiff be 0 instead of negative value
    maxAbsDiffElementRelDiff = (largest_abs_diff - atol) / brange if largest_abs_diff >= atol else np.float32(0.0)

    return close, largest_abs_diff, maxAbsDiffElementRelDiff


def allclose(a, b, *, rtol=1e-05, atol=1e-08, equal_nan=False, equal_inf=False, mode='max'):
    """Return True if each value of a and b are close within the specified tolerance.

    See the documentation of `isclose` for further details.

    Returns
    ------
    boolean
    """
    close, largest_abs_diff, maxAbsDiffElementRelDiff = isclose(as_native_type(a), as_native_type(b), rtol=rtol, atol=atol, 
                    equal_nan=equal_nan, equal_inf=equal_inf, 
                    mode=mode)
    return bool(np.all(close)), largest_abs_diff, maxAbsDiffElementRelDiff

def verify_accuracy(kernel_output, golden_output, np_dtype, tensor_name, kernel_name):
    try:
        if kernel_name == "forward":
            status, LargestAbsDiff, MaxAbsDiffElementRelDiff = allclose(kernel_output.clone().cpu().detach().to(torch.float32).numpy().astype(np_dtype), golden_output.clone().cpu().detach().to(torch.float32).numpy().astype(np_dtype), rtol=2e-2, atol=1e-5)
        else:
            status, LargestAbsDiff, MaxAbsDiffElementRelDiff = allclose(kernel_output.to(torch.float32).cpu().numpy().astype(np_dtype), golden_output.clone().to(torch.float32).detach().cpu().numpy().astype(np_dtype), rtol=2e-2, atol=1e-5)
        if not status:
            raise Exception(f"[{kernel_name}] Accuracy Mismatch for {tensor_name} Tensor with largest_abs_diff={LargestAbsDiff, MaxAbsDiffElementRelDiff} and ")
    
    except Exception as e:
       raise Exception(f"[{kernel_name}] Call Failed for {tensor_name} {str(e)}", True)
    
    return LargestAbsDiff.item(), MaxAbsDiffElementRelDiff.item()
    
def generate_kernel_parameters(test_line):
    if len(test_line) < 11: 
        return []
    
    # Check Dropping and Perfectly balanced flags
    dropping = test_line['dropping']
    perfectly_balanced = False if test_line['perfectly_balanced'] is None else test_line['perfectly_balanced']
    
    # Skip if neither condition is true
    if not dropping and not perfectly_balanced:
        return []
    
    # Try to convert values to integers, skip row if any conversion fails
    params = []
    try:
        H = test_line['H']
        T = test_line['T']
        E = test_line['E']
        TOPK = test_line['TOPK']
        Intermediate = test_line['Intermediate']
        BS = test_line['BS']
        TP_degree = test_line['TP_degree']
        EP_degree = test_line['EP_degree']
        capacity_factor = test_line['capacity_factor']
        fail = False
        sweep_name = test_line['SweepName']
        model = test_line['Model']
        fail = test_line["xfail"]
        # Calculate I_TP
        I_TP = Intermediate // TP_degree

        E = E // EP_degree
        
        # Collect parameters for each valid case
        if dropping:
            param = [test_line, model, sweep_name, H, T, E, TOPK, I_TP, BS, bfloat16, 0, 1, capacity_factor, EP_degree, fail]
            params.append(param)
        
        if perfectly_balanced:
            param = [test_line, model, sweep_name, H, T, E, TOPK, I_TP, BS, bfloat16, 0, 0, capacity_factor, EP_degree, fail]
            params.append(param)
        return params
    except (ValueError, IndexError):
        # Skip rows that can't be properly parsed
        return []
