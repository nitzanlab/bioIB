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

#Plotting bifurcations
def plot_bifurcations(depth, selected_indeces, betas, compressed_matrices_y, unique_indeces_list_y, y_xHat_arr, Y_labels, 
                      n_rows, n_cols,fig_size, y_order=None, metagene_order=None, file_name=None):
    """Plot the bioIB metagene hierarchy
    
    Params:
    depth: hierarchy depth to be plotted;
    selected_indeces: list of beta indeces at which the gene-to-metagene mapping based on y|xHat and on x|xHat is the same (Rand measure=1);
    betas: array of betas used in reverse annealing;
    compressed_matrices_y: list of compressed conditional probability matrices p(y|xHat);
    unique_indeces_list_y: indeces of unique metagenes generated from p(y|xHat) that can be used for its recovery;
    y_xHat_arr: numpy array containing conditional probability matrices of cell labels (y) given the metagenes (xHat) for each value of beta;
    Y_labels: labels of cell groups of interest to be used in the plot;
    n_rows, n_cols,fig_size: figure parameters;
    y_order (optional): if given, the cell labels will be ordered accordingly;
    metagene_order (optional): if given, the metagenes will be ordered accordingly;
    file_name (optional): if given, the figure will be saved under a given file name.

    """
    sel_ind=selected_indeces[::-1][-depth]
    n_metagenes=compressed_matrices_y[sel_ind].shape[1]
    n_labels=compressed_matrices_y[sel_ind].shape[0]
    print('%s metagenes' % n_metagenes)
    color1=mcp.gen_color(cmap="tab20",n=n_metagenes)
    logbetas=np.log2(betas)
    #depth=len(n_clus)
    if not y_order:
        y_order=np.arange(n_labels)
    if not metagene_order:
        metagene_order=np.arange(n_metagenes)
    metagene_labels=['MG %s' % (i) for i in range(n_metagenes)]
    title_Y_vals=np.array(Y_labels)[y_order]
    
    fig, axs=plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)

    axs=axs.ravel()

    limit=selected_indeces[depth-1]

    for y in range(n_labels):
        real_y=y_order[y]
        axs[y].spines['top'].set_visible(False)
        axs[y].spines['right'].set_visible(False)
        axs[y].set_xlabel('log(\u03B2)')
        axs[y].set_ylim(-0.05,1.05)
        for mtg_i, metagene in enumerate(metagene_order):
            g=np.unique(unique_indeces_list_y[limit])[metagene]
            axs[y].scatter(logbetas[limit:][::-1], y_xHat_arr[:,real_y,g][limit:][::-1], zorder=1, s=10, color=color1[metagene])
            #axs[y].plot(logbetas[limit:][::-1], y_xHat_arr[:,real_y,g][limit:][::-1], zorder=1, linewidth=3, color=color1[metagene])
            axs[y].scatter(logbetas[limit:][::-1][-1],
                           y_xHat_arr[:,real_y,g][limit],
                       s=150,
                       c=color1[metagene],
                       label=metagene_labels[mtg_i],
                       zorder=2)

        if y==n_labels-1:
            axs[y].legend(loc=(1.1,0), ncol=1, frameon=False)
        axs[y].set_ylabel('p ( y = %s | x\u0302 )' % Y_vals[real_y])
        axs[y].set_title('%s' % (title_Y_vals[y])) 
    plt.subplots_adjust(wspace=0.5)
    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
    return None


def plot_representative_genes(depth, selected_indeces, 
                              compressed_matrices_x, 
                              sorted_gene_names,
                              metagenes_to_show,
                              figsize, vmax,
                              n_genes, metagene_order=None,
                              file_name=None):

    """ Plots a heatmap of p(x|xhat) values of the most representative genes for the selected metagenes.
    Params:
    depth: hierarchy depth to be plotted;
    selected_indeces: list of beta indeces at which the gene-to-metagene mapping based on y|xHat and on x|xHat is the same (Rand measure=1);
    compressed_matrices_x: list of compressed conditional probability matrices p(x|xHat);
    sorted_gene_names: list of genes sorted by the information gain values in decreasing order;
    metagenes_to_show: list of metagene indeces to be plotted;
    figsize: figure size (width, height);
    vmax: maximal p(x|xHat) to anchor the colormap;
    n_genes: number of representative genes to plot for every metagene;
    metagene_order (optional): if given, the metagenes will be ordered accordingly;
    file_name (optional): if given, the figure will be saved under a given file name.
    """
    sel_ind=selected_indeces[::-1][-depth]
    
    chosen_indeces=[]
    x_given_xHat=compressed_matrices_x[sel_ind]
    n_metagenes=x_given_xHat.shape[1]
    if not metagene_order:
        metagene_order=np.arange(n_metagenes)

                
    for m in range(len(metagenes_to_show)):
        m_new=metagene_order[metagenes_to_show[m]]
        
        representative_genes_arr=np.argsort(x_given_xHat[:,m_new])[::-1][:n_genes]
        for g in representative_genes_arr:
            if g not in chosen_indeces:
                chosen_indeces.append(g)
    gene_labels=np.array(sorted_gene_names)[chosen_indeces]
    print(chosen_indeces)
    fig, axs=plt.subplots(figsize=figsize)
    axs=sns.heatmap(x_given_xHat[chosen_indeces,:][:, metagene_order][:,metagenes_to_show], 
                    #annot=True,
                    xticklabels=metagenes_to_show,
                   yticklabels=gene_labels,
                   vmax=vmax,
                   cmap='Reds')
    axs.set_xlabel('Metagenes')
    
    if file_name:
        plt.savefig(file_name, bbox_inches='tight')
        
    return None


def plot_metagene_hierarchy(Z, n_metagenes, metagene_to_Y_dict=None, file_name=None):
    """
    

    Parameters
    ----------
    Z : list
        linkage matrix for plotting the dendrogram.
    n_metagenes : int
        number of unique metagenes
    metagene_to_Y_dict : dictionary, optional
        dictionary linking every metagene with its associated Y label 
        (the one maximizing the probability p(y|xHat)).
        If provided, the plotted hierarchy will feature the linkage between the metagenes and the Y labels
        The default is None.
    file_name (optional): if given, the figure will be saved under a given file name.

    Returns
    -------
    None.

    """
    
    if not metagene_to_Y_dict:
        labels=['MG %s' % i for i in range(n_metagenes)]
    else:
        labels=['MG %s: %s' % (i, metagene_to_Y_dict['MG %s' % i]) for i in range(n_metagenes)]
        
    plt.figure(figsize=(4,6))
    ax=dendrogram(Z, color_threshold=0.1, orientation='left', 
                  labels=labels,
                  above_threshold_color='black')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']: 
        ax.spines[pos].set_visible(False) 
    plt.show()
    return None