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

def get_counts(counts, coldata, Y, Yvals, sample_id_column=None):
    """Reorganizes the cells by the parameter Y
    Params: 
    counts: dataframe with cellXgene scRNA-seq counts;
    coldata: dataframe with row names - cell indeces and a column with cellular labels representing the signal of interest Y;
    Y: the label of the signal of interest;
    Yvals: list of unique Y labels to be taken into analysis;
    sample_id_column (optional): Name of the column in coldata to be used for cell indeces instead of coldata row names.
    
    Returns:
    counts_all: a dataframe with scRNA-seq counts, organized by Y (in the order specified by Yvals)
    sample_lists: nested list of cell indeces belonging to each Y label, in the order specified by Yvals"""

    sample_lists=[]
    for y in Yvals:
        if sample_id_column==None:
            lst=list(coldata.index[coldata[Y]==y])
        else:
            lst=list(coldata[sample_id_column][coldata[Y]==y])
        sample_lists.append([x for x in lst if x in counts.columns])
    counts_by_states=[]
    for y in range(len(Yvals)):
        counts_by_states.append(np.array(counts[sample_lists[y]]))
    counts_all=np.concatenate(tuple(counts_by_states), axis=1)
    return counts_all, sample_lists

def get_probability_matrices(adata, Y, Y_vals):
    """Extracts the probability matrices from the AnnData object
    Params:
    adata: AnnData object;
    Y: signal of interest (should be in adata.obs);
    Y_vals: list of unique Y labels to be taken into analysis;
    
    Returns:
    p_x: vector of gene probabilities;
    p_y: vector of probabilities of cell labels specified in Y_vals;
    p_y_x_state: conditional probability matrix of cell labels given the genes;
    p_xy: joint probability matrix of cell labels and genes;
    p_y_x: conditional probability matrix of individual cells given the genes"""

    all_genes=list(adata.var_names)
    counts_df = pd.DataFrame(data=adata.X.todense().T, index=all_genes, columns=list(adata.obs_names))
    coldata=pd.DataFrame(list(adata.obs[Y]))
    coldata.index=list(adata.obs_names)
    coldata.columns=[Y]
    [counts_all,sample_lists]=get_counts(counts_df, coldata, Y, Y_vals)
    p_xy=counts_all/np.sum(counts_all)
    p_x=np.sum(p_xy, axis=1)
    p_y=np.sum(p_xy, axis=0)
    zero_genes=np.where(p_x==0)[0]
    if len(zero_genes)>0:
        print('There are zero genes (sum of counts = 0) in the matrix. Should filter them out before the analysis!')
    p_x=p_x.reshape(-1,1)
    p_y_x=p_xy/p_x

    y_x=[]
    p_y_state=[]
    border1_ind=0
    border2_ind=0
    for y in range(len(Y_vals)):
        try:
            border2_ind+=len(sample_lists[y])
        except:
            border2_ind+=sample_lists[y]
        y_x.append(np.sum(p_y_x[:,border1_ind:border2_ind], axis=1).reshape(1,-1))
        p_y_state.append(np.sum(p_y[border1_ind:border2_ind]))
        try:
            border1_ind+=len(sample_lists[y])
        except:
            border1_ind+=sample_lists[y]

    p_y_x_state=np.concatenate(tuple(y_x), axis=0)
    p_y=np.array(p_y_state)
    p_x=p_x.reshape(-1,1)
    p_y=p_y.reshape(-1,1)
    
    return [p_x, p_y, p_y_x_state, p_xy, p_y_x]

def sort_by_infogain(p_y_x_state, p_y, p_x, all_genes):
    """ Sorts genes by their Information Gain value
    Params:
    p_y_x_state: conditional probability matrix of cell labels given the genes;
    p_y: vector of probabilities of cell labels specified in Y_vals;
    p_x: vector of gene probabilities;
    all_genes: list of gene symbols, corresponding to p_x;

    Returns:
    sorted_indeces: list of gene indeces sorted by decreasing information gain value;
    sorted_gene_names: gene symbols sorted by decreasing information gain value;
    sorted_infogain: Information gain values of all genes, sorted in decreasing order
    """
    Dkl_vec=D_kl(p_y_x_state, p_y)
    infogain=Dkl_vec*p_x.reshape(-1,1)
    sorted_infogain=np.sort(infogain.T.tolist()[0])[::-1]
    sorted_indeces=list(np.argsort(infogain.T.tolist()[0]))
    sorted_indeces=sorted_indeces[::-1]
    sorted_gene_names=[all_genes[i] for i in sorted_indeces]
    return [sorted_indeces, sorted_gene_names, sorted_infogain]