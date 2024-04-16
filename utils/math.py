'''
math.py
Project: utils
Created: 2023-08-10 23:51:59
Author: Bill Chen (bill.chen@live.com)
-----
Last Modified: 2023-11-28 23:25:46
Modified By: Bill Chen (bill.chen@live.com)
'''
import numpy as np
from scipy.stats import entropy
import bottleneck as bn
from numba import njit


def norm01(a: np.ndarray, percentile: float=100, nanmapping: float | str=0, peraxis: bool = False) -> np.ndarray:
    """Normalize a numpy array to [0,1]. All nan values will be set to 0.
    """
    if not peraxis:
        maxv = np.nanpercentile(a, percentile)
        minv = np.nanpercentile(a, 100 - percentile)
        a = (a - minv) / (maxv - minv)
        # a = np.where(np.isnan(a), 0, a)
        a = bn.replace(a, np.nan, 0)
    else:
        for i in range(a.shape[1]):
            a[: ,i] = norm01(a[:, i], percentile, nanmapping, False)
    return a


@njit
def norm01_jit(a: np.ndarray, percentile: float=100, nanmapping: float | str=0) -> np.ndarray:
    maxv = np.nanpercentile(a, percentile)
    minv = np.nanpercentile(a, 100 - percentile)
    a = (a - minv) / (maxv - minv)
    a = np.where(np.isnan(a), 0, a)
    return a


def tolist_rounded(a: np.ndarray, decimals: int):
    """Convert a numpy array to a list with rounded values that support nested np arrays.
    """
    if a.ndim == 0:  # scalar value
        return round(float(a), decimals)
    else:
        return [tolist_rounded(sub_array, decimals) for sub_array in a]

def conditional_entropy(x: np.ndarray, y: np.ndarray):
    """Compute conditional entropy H(X|Y)
    """
    x = x.flatten()
    y = y.flatten()
    return entropy(x, y) - entropy(x)

def variation_of_information(x: np.ndarray, y: np.ndarray):
    """Compute variation of information
    """
    return conditional_entropy(x, y) + conditional_entropy(y, x)

if __name__ == '__main__':
    x = np.random.rand(10, 10)
    y = np.random.rand(10, 10)
    # print(variation_of_information(x, y))
    t = np.array([[1.1111,1.1111],[2.2222,2.2222]])
    print(tolist_rounded(t, 2))