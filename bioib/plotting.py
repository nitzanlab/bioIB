import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 20})
from mycolorpy import colorlist as mcp
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from typing import Any, Optional
import os
import seaborn as sns
import pandas as pd

from .utils import _get_selected_index


def plot_infogain(
        bioib,
        n_visualized_genes: Optional[int] = None,
        save_path: Optional[os.PathLike] = None,
        **kwargs: Any
    ):
    """ Plots the genes ordered by their infogain value.
    Parameters
    ----------
    bioib
        bioIB object.
    n_visualized_genes
        Number of genes to visualize on the plot.
    save_path
        Path where to save the figure.

    Returns
    ----------
    None
    """
    if not n_visualized_genes:
        n_visualized_genes = bioib.adata.shape[1]

    init_prob=bioib._init_probability_matrices()
    infogain_vals=bioib._calculate_infogain(init_prob['p_x'], init_prob['p_y'], init_prob['p_y_mid_x'])

    plt.rc('font', family='serif', size=18)
    plt.figure(figsize=(20, 8))
    plt.scatter(range(n_visualized_genes),
                infogain_vals['sorted_infogain'][:n_visualized_genes])
    plt.xticks(ticks=range(n_visualized_genes),
               labels=infogain_vals['sorted_gene_names'][:n_visualized_genes], rotation=90)
    plt.xlabel('Genes (sorted by infogain)')
    plt.ylabel('infogain')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.show()

def plot_trade_off(bioib, save_path=None):
    """ Plots the trade-off between complexity (I(x,xHat)) and accuracy (I(xHat, y)).
    Scatters the points of selected betas with the labels indicating the corresponding numbers of metagenes.
    Parameters
    ----------
    bioib
        bioIB object.
    save_path
        Path where to save the figure.

    Returns
    ----------
    None
    """
    plt.rc('font', family='serif', size=18)
    color1 = mcp.gen_color(cmap="tab20", n=len(bioib._selected_indices))
    plt.plot(bioib._info_vals["I_x_xhat"], bioib._info_vals["I_xhat_y"], zorder=0)
    for i, index in enumerate(bioib._selected_indices):
        plt.scatter(bioib._info_vals["I_x_xhat"][index], bioib._info_vals["I_xhat_y"][index],
                    label='%s MG' % bioib._selected_n_metagene_list[i], zorder=1, color=color1[i])
    plt.xlabel('I(X, X\u0302)')
    plt.ylabel('I(X\u0302, Y)')

    labels_per_column=6
    if len(bioib._selected_indices) % labels_per_column == 0:
        ncols = len(bioib._selected_indices) / labels_per_column
    else:
        ncols = (len(bioib._selected_indices) / labels_per_column) + 1
    plt.legend(loc=(1.1, 0), ncols=ncols)
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)

def plot_bifurcations(bioib, n_leaves=None,  ncols=None, save_path=None):
    """ Plots the metagene bifurcations during the gradual compression with reverse-annealing.
    Parameters
    ----------
    bioib
        bioIB object.
    n_leaves
        The starting resolution of the bioIB hierarchy - maximal number of metagenes.
    ncols
        Number of subplot columns in the figure.
    save_path
        Path where to save the figure.

    Returns
    ----------
    None
    """
    if not n_leaves:
        n_leaves = bioib._selected_n_metagene_list[-1]
    selected_index = _get_selected_index(n_leaves, bioib._selected_n_metagene_list, bioib._selected_indices)

    plt.rc('font', family='serif', size=18)
    color1 = mcp.gen_color(cmap="tab20", n=n_leaves)
    metagene_labels = ['MG %s' % (i) for i in range(n_leaves)]

    n_subplots_to_delete = 0
    if ncols:
        rem = len(bioib.groups_of_interest) % ncols
        if rem > 0:
            nrows = len(bioib.groups_of_interest) // ncols + 1
            n_subplots_to_delete = ncols - rem
    else:
        ncols = len(bioib.groups_of_interest)
        nrows = 1

    [width, height] = [ncols * 4, nrows * 6]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height))
    axs = axs.ravel()

    for i, y in enumerate(bioib.groups_of_interest):
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].set_xlabel('log(beta)')

        for mtg in range(n_leaves):
            g = np.unique(bioib._unique_indices['p_y_mid_xhat'][selected_index])[mtg]
            axs[i].scatter(bioib._betas[selected_index:][::-1], np.asarray(bioib._probs_dict["p_y_mid_xhat"])[:, i, g][selected_index:][::-1],
                           zorder=1, s=10, color=color1[mtg])
            axs[i].scatter(bioib._betas[selected_index:][::-1][-1],
                           np.asarray(bioib._probs_dict["p_y_mid_xhat"])[:, i, g][selected_index],
                           s=150, c=color1[mtg], label=metagene_labels[mtg], zorder=2)

        axs[i].set_ylabel('p(y|x\u0302)')
        axs[i].set_title(y)

    for p in range(n_subplots_to_delete):
        axs[-p - 1].remove()

    plt.legend(loc=(1.1, 0))
    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)


def plot_metagene_hierarchy(bioib, n_leaves=None, map_metagenes_to_cell_states=True, save_path=None):
    """
    Parameters
    ----------
    bioib
        bioIB object.
    n_leaves
        The maximal number of metagenes (compression level)  - number of leaves in the plotted metagene hierarchy
    map_metagenes_to_cell_states
        Whether to specify the connection of metahenes to cell states of interest in the hierarchy.
    save_path
        The path to save the plot.

    Returns
    -------
    None.
    """
    if not n_leaves:
        n_leaves = bioib._selected_n_metagene_list[-1]

    Z = bioib._get_linkage_matrix(n_leaves)

    if map_metagenes_to_cell_states:
        bioib.map_metagenes_to_cell_states(n_leaves)
        labels = ['MG %s: %s' % (i, bioib.metagenes_to_cell_states['MG %s' % i]) for i in range(n_leaves)]

    else:
        labels = ['MG %s' % i for i in range(n_leaves)]

    plt.figure(figsize=(4, 6))
    ax = dendrogram(Z, color_threshold=0.1, orientation='left',
                    labels=labels,
                    above_threshold_color='black')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']:
        ax.spines[pos].set_visible(False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.show()


def plot_representative_genes(bioib_adata,
                              gene_to_metagene_map='bioIB_gene_MG_mapping',
                              n_top_genes=5,
                              metagenes_to_show='all',
                              save_path=None):

    """ Plots a heatmap of conditional probability values p(x|xhat) of the most representative genes for the selected metagenes.
    Parameters
    ----------
    bioib_adata
        adata annotated with bioIB
    gene_to_metagene_map
        Label of bioib_adata.uns containing the mapping of values of interest.
        The default is "bioIB_gene_MG_mapping", annotated with bioib.compress() function.
        To plot the result of the flat clustering use "bioIB_flat_%s_MGs_gene_to_MG" % n_metagenes.
    n_top_metagenes
        Number of the most representative genes to plot for each metagene
    metagenes_to_show
        List of metagene indices to display on the heatmap
    save_path
        The path to save the plot.

    Returns
    ----------
    Dataframe with the matrix plotted in the heatmap (p(x|xhat), genes X metagenes)

    """

    chosen_indeces = []
    x_given_xHat = bioib_adata.uns[gene_to_metagene_map]
    n_metagenes = x_given_xHat.shape[1]

    if metagenes_to_show == 'all':
        metagenes_to_show = np.arange(n_metagenes)

    for mtg in metagenes_to_show:
        representative_genes_arr = np.argsort(x_given_xHat[:, mtg])[::-1][:n_top_genes]
        for g in representative_genes_arr:
            if g not in chosen_indeces:
                chosen_indeces.append(g)
    gene_labels = np.array(bioib_adata.var_names)[chosen_indeces]

    fig, axs = plt.subplots(figsize=(n_metagenes,n_metagenes*1.5))
    axs = sns.heatmap(x_given_xHat[chosen_indeces, :][:, metagenes_to_show],
                      # annot=True,
                      xticklabels=metagenes_to_show,
                      yticklabels=gene_labels,
                      #vmax=vmax,
                      cmap='Reds')
    axs.set_xlabel('Metagenes')
    axs.set_ylabel('Genes')
    plt.tight_layout()
    plt.show()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    res_df=pd.DataFrame(x_given_xHat[chosen_indeces, :][:, metagenes_to_show])
    res_df.index=gene_labels
    res_df.columns=metagenes_to_show

    return res_df