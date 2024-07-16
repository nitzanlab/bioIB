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

def filter_collapsed_genes(y_xHat_arr, xHat_x_arr, x_xHat_arr, p_xHat_arr):
    """ Filters the reverse_annealing output by deleting the collapsed metagenes (with zero probability)
    Params:
    y_xHat_arr: numpy array containing conditional probability matrices of cell labels (y) given the metagenes (xHat) for each value of beta;
    xHat_x_arr: numpy array containing conditional probability matrices of metagenes (xHat) given the genes (x) for each value of beta;
    x_xHat_arr: numpy array containing conditional probability matrices of genes (x) given the metagenes (xHat) for each value of beta;
    p_xHat_arr: numpy array containing metagene probability vectors for each value of beta.
    
    Returns:
    New y_xHat_arr, xHat_x_arr, x_xHat_arr, p_xHat_arr without the collapsed metagenes (xHat)"""
    nan_genes=np.where(np.sum(y_xHat_arr[-1,:,:], axis=0)==0)[0]
    xHat_x_copy=list(xHat_x_arr).copy()
    y_xHat_copy=list(y_xHat_arr).copy()
    x_xHat_copy=list(x_xHat_arr).copy()
    xHat_copy=list(p_xHat_arr).copy()

    for i, arr in enumerate(xHat_x_copy):
        new_arr=np.delete(arr, nan_genes, 0)
        xHat_x_copy[i]=new_arr

    for i, arr in enumerate(y_xHat_copy):
        new_arr=np.delete(arr, nan_genes, 1)
        y_xHat_copy[i]=new_arr

    for i, arr in enumerate(x_xHat_copy):
        new_arr=np.delete(arr, nan_genes, 1)
        x_xHat_copy[i]=new_arr

    for i, arr in enumerate(xHat_copy):
        new_arr=np.delete(arr, nan_genes, 0)
        xHat_copy[i]=new_arr
        
    xHat_x_arr=np.array(xHat_x_copy)
    y_xHat_arr=np.array(y_xHat_copy)
    x_xHat_arr=np.array(x_xHat_copy)
    p_xHat_arr=np.array(xHat_copy)
    
    return [y_xHat_arr, xHat_x_arr, x_xHat_arr, p_xHat_arr]

def round_IB_res(beta, array, constant=None):
    """ Rounds the reverse_annealing output matrix as a function of beta, or, optionally, to a constant number of digits
    Params:
    beta: beta value used to produce the provided matrix;
    array: conditional probability matrix produced by reverse annealing with the given beta value;
    constant (optional): constant number of digits for rounding, independent of beta.
    
    Returns:
    array_rounded: rounded array"""
    if constant:
        array_rounded=np.around(array, constant)
        return array_rounded
    prec=1/(beta*100)
    acc=int(abs(np.round(np.log10(prec))))
    array_rounded=np.around(array, acc)
    return array_rounded



def get_compressed_matrices(betas, probability_matrices):
    """Extracts a list of compressed data representations (cell labels/genes X metagenes) from the array of conditional probability matrices (reverse annealing output)
    Params:
    betas: array of betas used in reverse annealing;
    probability_matrices: array of conditional probability matrices (y_xHat_arr or xHat_x_arr) representing the reverse annealing output
    
    Returns:
    n_metagenes: list of numbers of unique metagenes poduced at each beta value;
    compressed_matrices: list of compressed matrices with the number of metagenes specified in n_metagenes;
    unique_indeces_list: indeces of unique metagenes from the original probability matrix that that can be used for its recovery.
    """
    n_metagenes=[]
    compressed_matrices=[]
    unique_indeces_list=[]
    
    for ii in range(len(betas)):
        prob_mat=probability_matrices[ii]
        rounded_mat=round_IB_res(betas[ii], prob_mat)
        

        [compressed_mat, recovering_indeces, unique_indeces]=np.unique(rounded_mat, 
                                                   axis=1, 
                                                   return_index=True,
                                                   return_inverse=True)
        
        compressed_mat=rounded_mat[:,sorted(recovering_indeces)]
        compressed_matrices.append(compressed_mat)
        fixed_unique_indeces=[recovering_indeces[i] for i in unique_indeces]

        unique_indeces_list.append(np.array(fixed_unique_indeces))
        n_metagenes.append(compressed_mat.shape[1])
        
    return [n_metagenes, compressed_matrices, unique_indeces_list]



def get_rand_mat(inp):
    """Constructs a clustering matrix indicating the mapping of genes to metagenes for further calculation of Rand index.
    Params:
    inp: list of metagene indeces associated with each gene.
    Returns:
    rand_mat: a matrix of mapping of genes to metagenes """
    dim=len(inp)
    rand_mat=np.zeros([dim, dim])
    for i in range(dim):
        other_locs=np.where(inp==i)[0]
        rand_mat[i, other_locs]=[1]*len(other_locs)
    return rand_mat


def get_selected_indeces(n_metagenes, unique_indeces_list_x, unique_indeces_list_y):
    """Get beta indeces that yielded the same gene-to-metagene mapping both based on y|xHat and on x|xHat.
    Params:
    n_metagenes: list of numbers of unique metagenes generated at each beta value;
    unique_indeces_list_x: indeces of unique metagenes generated from p(x|xHat) that can be used for its recovery;
    unique_indeces_list_y:indeces of unique metagenes generated from p(y|xHat) that can be used for its recovery.

    Returns:
    selected_indeces: list of beta indeces at which the gene-to-metagene mapping based on y|xHat and on x|xHat is the same (Rand measure=1);
    n_clus: list of number of unique metagenes for each beta value in selected_indeces.
    """
    rand_matrices_x=[]
    rand_matrices_y=[]
    rand_measures=[]

    for p in range(len(n_metagenes)):
        rand_mat_x=get_rand_mat(unique_indeces_list_x[p])
        dim=rand_mat_x.shape[0]
        rand_matrices_x.append(rand_mat_x)
        rand_mat_y=get_rand_mat(unique_indeces_list_y[p])
        rand_matrices_y.append(rand_mat_y)
        rand=1-(np.sum((rand_mat_x-rand_mat_y)**2)/(dim*(dim-1)))
        rand_measures.append(rand)
        
    matching_points=np.where(np.array(rand_measures)==1)[0]
    n_matching_metagenes=np.array(n_metagenes)[matching_points]
    [n_clus, indeces, all_indeces]=np.unique(n_matching_metagenes, 
                                             return_index=True,
                                             return_inverse=True)
    selected_indeces=np.array(matching_points)[indeces]
    
    return [selected_indeces, n_clus]



def get_representative_genes(compressed_matrices_x, selected_indeces, n_clus, saving_repository=None):
    """ Get the matrix of genes representing every metagene.
    Params:
    compressed_matrices_x: list of compressed conditional probability matrices of genes given the metagenes (x|xHat);
    selected_indeces:  list of beta indeces at which the gene-to-metagene mapping based on y|xHat and on x|xHat is the same (Rand measure=1);
    n_clus: list of number of unique metagenes for each beta value in selected_indeces;
    saving_repository (optional): if given, will be used to save the csv table with conditional probabilities for each gene given each metagene, 
    for every number of metagenes in n_clus.
    
    Returns:
    representative_genes: a dictionary mapping each metagene to the index of its most representative gene.
    """
    
    representative_genes={}

    for n in range(len(selected_indeces)):
        gene_clus_dic={}
        ind=selected_indeces[n]

        x_xHat=compressed_matrices_x[ind]

        representative_genes[n_clus[n]]=[]
        
        for cl in range(n_clus[n]):
            x_xHat_vec=x_xHat[:,cl]
            representative_genes[n_clus[n]].append(np.argmax(x_xHat_vec))
            gene_clus_dic['Metagene_%s' % cl]=list(x_xHat_vec)

        if saving_repository:
            gene_clus_df=pd.DataFrame(gene_clus_dic)
            gene_clus_df.index=sorted_gene_names[:n_genes]
            gene_clus_df.to_csv(saving_repository+'/%s_clusters_Gene_Cluster_probabilities.csv' % n_clus[n])

    return representative_genes

def annotate_bioIB_metagenes(adata, depth, selected_indeces, compressed_matrices_x, metagene_order=None):
    """ Annotate metagene expression at the single cell level in the original AnnData.
    
    Params:
    adata: original AnnData object
    depth: hierarchy depth to be plotted;
    selected_indeces: list of beta indeces at which the gene-to-metagene mapping based on y|xHat and on x|xHat is the same (Rand measure=1);
    compressed_matrices_x: list of compressed conditional probability matrices p(x|xHat);
    metagene_order (optional): if given, the metagenes will be ordered accordingly;
    """
    sel_ind=selected_indeces[::-1][-depth]
    x_given_xHat=compressed_matrices_x[sel_ind]
    n_metagenes=x_given_xHat.shape[1]
    if not metagene_order:
        metagene_order=np.arange(n_metagenes)
    for i, n in enumerate(metagene_order):
        factors=x_given_xHat[:,n].reshape(1,-1)
        multiplied_by_factors=np.multiply(adata.X.todense(),factors)
        average_vector=np.sum(multiplied_by_factors, axis=1)
        adata.obs['Metagene %s' % i]=list(average_vector)
    return None
    

def get_bioIB_compressed_adata(adata, depth, selected_indeces, compressed_matrices_x, metagene_order=None):
    """ Create a new AnnData object of single cells by metagenes, representing the achieved compressed data representation.
    
    Params:
    adata: original AnnData object
    depth: hierarchy depth to be plotted;
    selected_indeces: list of beta indeces at which the gene-to-metagene mapping based on y|xHat and on x|xHat is the same (Rand measure=1);
    compressed_matrices_x: list of compressed conditional probability matrices p(x|xHat);
    metagene_order (optional): if given, the metagenes will be ordered accordingly;

    Returns:
    compressed_adata: AnnData object of cells X metagenes with the annotation of all the observations from the original adata.

    """
    sel_ind=selected_indeces[::-1][-depth]
    x_given_xHat=compressed_matrices_x[sel_ind]
    average_mat=np.zeros([adata.shape[0],compressed_matrices_x[sel_ind].shape[1]])
    n_metagenes=x_given_xHat.shape[1]
    if not metagene_order:
        metagene_order=np.arange(n_metagenes)
    for i, n in enumerate(metagene_order):
        factors=x_given_xHat[:,n].reshape(1,-1)
        multiplied_by_factors=np.multiply(adata.X.todense(),factors)
        average_vector=np.sum(multiplied_by_factors, axis=1)
        average_mat[:,n]=average_vector.reshape(-1,)
    compressed_adata=sc.AnnData(average_mat)
    for obs in list(adata.obs):
        compressed_adata.obs[obs]=list(adata.obs[obs])
    return compressed_adata