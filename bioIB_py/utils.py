import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.stats import entropy
from scipy.special import logsumexp, rel_entr
from jax.nn import softmax
from scipy.signal import find_peaks
from operator import itemgetter
import collections
import copy
import pyparsing
import pickle
from jax import jit
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from mycolorpy import colorlist as mcp
import seaborn as sns
import scanpy as sc

def D_kl(p, q):
    """
    Calculate Kullback-Leibler divergence using numpy
    
    D_kl[p || q] = \sum_x p(x) * ln (p(x)/q(x))
    
    """ 
    _, dp = p.shape
    _, dq = q.shape
    dkl = np.zeros((dp, dq))
    for i in range(dp):
        for j in range(dq):
            dkl[i,j] = np.sum(rel_entr(p[:, i], q[:, j]))  

    return dkl



def kl_fn(x, y):
    r=jnp.where((x > 0) & (y > 0), x * jnp.log(x / y), jnp.inf)
    r=jnp.where(x == 0, 0, r)
    res=jnp.sum(r)
    return res



def D_kl_jax(p, q):
    """
    Calculate Kullback-Leibler divergence using jax
    
    D_kl[p || q] = \sum_x p(x) * ln (p(x)/q(x))
    
    """
    
    kl_vec = jnp.vectorize(kl_fn, signature="(n),(m)->()")

    return kl_vec(p[:, None, :], q[None, :, :])

D_kl_jit = jit(D_kl_jax)



def calc_I_xy(p_x_mid_y, p_x, p_y):
    """
    :param p_x_mid_y: the conditional probability (|X|, |T|)
    :param p_y, p_x: The marginal probabilities
    :return: I_xy: The Mutual information I(X;Y) = <D_kl[p(x|y) || p(x)]>_{p(y)}
    """
    I_xy = 0
    for i, p_y_i in enumerate(p_y):
        I_xy += p_y_i * kl_fn(p_x_mid_y[:, i], p_x)
    return I_xy
