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
from .utils import D_kl, D_kl_jit, calc_I_xy


def reverse_annealing(beta, p_x, p_y, y_given_x):
    """ Produces a series of clustering solutions for each value of beta from the indicated initial beta to 1.
    Params:
    beta: Initial beta value to start the compression with;
    p_x: vector of gene probabilities;
    p_y: vector of probabilities of cell labels of interest;
    y_given_x: conditional probability matrix of cell labels given the genes
    
    Returns:
    I_x_xhat: list of the Mutual Information values between genes (x) and metagenes (xHat) for each value of beta;
    I_xhat_y: list of the Mutual Information values between metagenes (xHat) and cell labels (y) for each value of beta;
    xHat_given_x_list: list of conditional probability matrices of metagenes (xHat) given the genes (x) for each value of beta;
    y_given_xHat_list: list of conditional probability matrices of cell labels (y) given the metagenes (xHat) for each value of beta;
    x_given_xHat_list: list of conditional probability matrices of genes (x) given the metagenes (xHat) for each value of beta;
    p_xHat_list: list of metagene probability vectors for each value of beta;
    iters: list of iteration numbers that took bioIB to converge at each value of beta
    """
    I_x_xhat=[]
    I_xhat_y=[]
    xHat_given_x_list=[]
    y_given_xHat_list=[]
    x_given_xHat_list=[]
    p_xHat_list=[]
    iters = []
    Dkl_list=[]
    
    num_betas=400
    logbeta=np.log2(beta)
    betas = 2**np.linspace(logbeta, 0, num_betas)
    
    p_xHat = p_x.copy()
    y_given_xHat = y_given_x.copy()
    d_x = p_x.size
    d_xHat = p_xHat.size
    new_xHat_given_x=jnp.identity(d_x)

    betas_to_print=betas[::20]

    for beta in betas:
        if beta in betas_to_print:
            print('beta=%s' % beta)
        epsilon = 1e-8 
        min_iter = 1e+7
        itr=0
        err=epsilon+1

        while err > epsilon and itr < min_iter:
            xHat_given_x = new_xHat_given_x
            dkl_jnp=D_kl_jit(y_given_x.T, y_given_xHat.T)
            log_new_xHat_given_x = jnp.log(p_xHat.reshape(d_xHat, 1)) - (beta * dkl_jnp).T
            new_xHat_given_x = jax.nn.softmax(log_new_xHat_given_x, axis=0)
            xHat_given_x_avg = 0.5*(xHat_given_x + new_xHat_given_x)
            err = 0.5*jnp.mean(jnp.diag(D_kl_jit(new_xHat_given_x.T, xHat_given_x_avg.T)) 
                                     + jnp.diag(D_kl_jit(xHat_given_x.T, xHat_given_x_avg.T))) if (itr > 0) else np.inf


            p_xHat=jnp.dot(new_xHat_given_x, p_x)
            x_given_xHat = (new_xHat_given_x * p_x.reshape(1, d_x)).T / p_xHat.reshape(1, d_xHat)

            if np.isnan(x_given_xHat).any():
                x_given_xHat = x_given_xHat.at[np.isnan(x_given_xHat)].set(0)
            if np.isinf(x_given_xHat).any():
                x_given_xHat = x_given_xHat.at[np.isinf(x_given_xHat)].set(0)

            y_given_xHat = jnp.dot(y_given_x, x_given_xHat)

            if np.isnan(y_given_xHat).any():
                y_given_xHat = y_given_xHat.at[np.isnan(y_given_xHat)].set(0)

            itr += 1

        I_xt=calc_I_xy(x_given_xHat, p_x, p_xHat)
        I_ty=calc_I_xy(y_given_xHat, p_y, p_xHat)

        
        I_x_xhat.append(I_xt)
        I_xhat_y.append(I_ty)
        xHat_given_x_list.append(xHat_given_x)
        y_given_xHat_list.append(y_given_xHat)
        x_given_xHat_list.append(x_given_xHat)
        p_xHat_list.append(p_xHat)
        iters.append(itr)

    return [I_x_xhat, I_xhat_y, xHat_given_x_list, y_given_xHat_list, x_given_xHat_list, p_xHat_list, iters]


def iIB_random_init(n_genes, p_x, y_given_x, n_clus):
    """ Generates random probability matrices for the initiation of the bioIB flat clustering function.
    
    Params:
    n_genes: number of original genes to be clustered;
    p_x: vector of gene probabilities;
    y_given_x: conditional probability matrix of cell labels given the genes;
    n_clus: the desired number of flat gene clusters.

    Returns:
    xHat_given_x: random conditional probability matrix of metagenes (xHat) given the genes (x);
    p_xHat: random metagene probability vector;
    x_given_xHat: random conditional probability matrix of genes (x) given the metagenes (xHat);
    y_given_xHat: random conditional probability matrix of cell labels (y) given the metagenes (xHat).
    """
    
    n_repeats=np.floor_divide(n_genes, n_clus)
    remainder=n_genes%n_clus
    ones_indeces=list(range(n_clus))*n_repeats
    ones_indeces.extend(np.random.randint(0,n_clus-1,remainder))
    np.random.shuffle(ones_indeces)
    xHat_given_x=np.zeros([n_clus, n_genes])
    for i in range(n_genes):
        xHat_given_x[ones_indeces[i],i]=1
    p_xHat=xHat_given_x@p_x
    p_xHat=p_xHat/np.sum(p_xHat)
    x_given_xHat=(xHat_given_x.T*p_x)/(p_xHat.reshape(1,-1))
    x_given_xHat=x_given_xHat/np.sum(x_given_xHat, axis=0)
    y_given_xHat=y_given_x@x_given_xHat
    return [xHat_given_x, p_xHat, x_given_xHat, y_given_xHat]


def iIB_alg_jax(beta, n_genes, p_x_sorted, p_y, y_given_x, y_given_xHat, p_xHat, xHat_given_x):
    """ bioIB algorithm for flat gene clustering.
    Params:
    beta: beta value;
    n_genes: number of genes to be clustered;
    p_x_sorted: vector of gene probabilities;
    p_y: vector of probabilities of cell labels of interest;
    y_given_x: conditional probability matrix of cell labels given the genes;
    y_given_xHat:random conditional probability matrix of cell labels (y) given the metagenes (xHat);
    p_xHat: random metagene probability vector;
    xHat_given_x: random conditional probability matrix of metagenes (xHat) given the genes (x).

    Returns:
    xHat_given_x: optimized conditional probability matrix of metagenes (xHat) given the genes (x);
    p_xHat: optimized metagene probability vector;
    y_given_xHat: optimized conditional probability matrix of cell labels (y) given the metagenes (xHat);
    x_given_xHat: optimized conditional probability matrix of genes (x) given the metagenes (xHat);
    itr: number of iterations that bioIB took to converge;
    I_xt: associated mutual information between genes and metagenes;
    I_ty: associated mutual information between cell labels and metagenes; 
    obj - the value of the objective function [I_xt - beta * I_ty].
    """
    epsilon=1e-8
    itr=0
    err=100
    d_x = p_x_sorted.size
    d_xHat = p_xHat.size
    while err>epsilon:

        dkl_jnp=D_kl_jit(y_given_x.T, y_given_xHat.T)

        log_new_xHat_given_x = jnp.log(p_xHat.reshape(d_xHat, 1)) - (beta * dkl_jnp).T
        new_xHat_given_x = jax.nn.softmax(log_new_xHat_given_x, axis=0)
        xHat_given_x_avg = 0.5*(xHat_given_x + new_xHat_given_x)

        err=np.sum(abs(new_xHat_given_x-xHat_given_x)) if (itr > 0) else np.inf

        p_xHat=jnp.dot(new_xHat_given_x, p_x_sorted)
        
        x_given_xHat = (new_xHat_given_x * p_x_sorted.reshape(1, d_x)).T / p_xHat.reshape(1, d_xHat)

            
        if np.isnan(x_given_xHat).any():
            x_given_xHat = x_given_xHat.at[np.isnan(x_given_xHat)].set(0)

            
        y_given_xHat = jnp.dot(y_given_x, x_given_xHat)

        
        if np.isnan(y_given_xHat).any():)
            y_given_xHat = y_given_xHat.at[np.isnan(y_given_xHat)].set(0)

        xHat_given_x=new_xHat_given_x
        itr += 1

    I_xt=calc_I_xy(x_given_xHat, p_x_sorted, p_xHat)
    I_ty=calc_I_xy(y_given_xHat, p_y, p_xHat)
    obj=I_xt-beta*I_ty

    return [xHat_given_x, p_xHat, y_given_xHat, x_given_xHat, itr, I_xt, I_ty, obj]

def flat_clustering(p_x_sorted, p_y, y_given_x, n_cl, n_iters, beta=1e4):
    """ bioIB function for flat clustering, merging the genes into |n_cl| metagenes.
    
    Params:
    p_x_sorted: vector of gene probabilities;
    p_y: vector of probabilities of cell labels of interest;
    y_given_x: conditional probability matrix of cell labels given the genes;
    n_cl: the desired number of metagenes to be produced;
    n_iters: number of random initializations for bioIB;
    beta: beta value (default=1e4).

    Returns:

    xHat_given_x: optimized conditional probability matrix of metagenes (xHat) given the genes (x);
    p_xHat: optimized metagene probability vector;
    y_given_xHat: optimized conditional probability matrix of cell labels (y) given the metagenes (xHat);
    x_given_xHat: optimized conditional probability matrix of genes (x) given the metagenes (xHat);
    itr: number of iterations that bioIB took to converge;
    I_xt: associated mutual information between genes and metagenes;
    I_ty: associated mutual information between cell labels and metagenes; 
    obj - the value of the objective function [I_xt - beta * I_ty].

     """
    n_genes=p_x_sorted.size
    all_res=[]
    I_ty_list=[]
    obj_list=[]
    successful_iterations=0
    while successful_iterations<n_iters:
        [xHat_given_x, p_xHat, x_given_xHat, y_given_xHat]=iIB_random_init(n_genes, p_x_sorted, y_given_x, n_cl)
        [xHat_given_x, p_xHat, y_given_xHat, x_given_xHat, itr, I_xt, I_ty, obj]=iIB_alg_jax(beta, n_genes, p_x_sorted, 
                                                                                     p_y, y_given_x, y_given_xHat, 
                                                                                     p_xHat, xHat_given_x)
        res=[xHat_given_x, p_xHat, y_given_xHat, x_given_xHat, itr, I_xt, I_ty, obj]
        real_n_clus=np.unique(xHat_given_x, axis=0).shape[0]
        if real_n_clus < n_cl:
            continue
        successful_iterations+=1
        I_ty_list.append(res[-2])
        obj_list.append(res[-1])
        all_res.append(res)
    max_Ity=np.argmax(I_ty_list)
    min_obj=np.argmin(obj_list)
    final_res=all_res[min_obj]
    return final_res