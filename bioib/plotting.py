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
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.show()

def plot_bifurcations(bioib, n_leaves=None,  metagene_order=None, group_order=None, ncols=None, cmap="tab20", save_path=None):
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
    if not metagene_order:
        metagene_order=np.arange(n_leaves)
    metagene_order=list(metagene_order)

    if not group_order:
        group_order = bioib.groups_of_interest
    selected_index = _get_selected_index(n_leaves, bioib._selected_n_metagene_list, bioib._selected_indices)

    plt.rc('font', family='serif', size=18)

    color1 = mcp.gen_color(cmap=cmap, n=n_leaves)

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

    for i, y in enumerate(group_order):
        ind=list(bioib.groups_of_interest).index(y)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].set_xlabel('\u03B2')

        for ii, mtg in enumerate(metagene_order):
            g = np.unique(bioib._unique_indices['p_y_mid_xhat'][selected_index])[mtg]
            axs[i].scatter(bioib._betas[selected_index:][::-1], np.asarray(bioib._probs_dict["p_y_mid_xhat"])[:, ind, g][selected_index:][::-1],
                           zorder=1, s=10, color=color1[ii])
            axs[i].scatter(bioib._betas[selected_index:][::-1][-1],
                           np.asarray(bioib._probs_dict["p_y_mid_xhat"])[:, ind, g][selected_index],
                           s=150, c=color1[ii], label=metagene_labels[ii], zorder=2)

        axs[i].set_ylabel('p(y=%s|x\u0302)' % y)
        axs[i].set_title(y)
        axs[i].set_ylim(-0.01,1.01)

    for p in range(n_subplots_to_delete):
        axs[-p - 1].remove()

    plt.legend(loc=(1.1, 0))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.show()

def plot_metagene_hierarchy(bioib, n_leaves=None, map_metagenes_to_cell_states=True, metagene_order=None, save_path=None):
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
    if not metagene_order:
        metagene_order = np.arange(n_leaves)
    metagene_order=list(metagene_order)
    if map_metagenes_to_cell_states:
        bioib.map_metagenes_to_cell_states(n_leaves)
        labels = ['MG %s: %s' % (metagene_order.index(i), bioib.metagenes_to_cell_states['MG %s' % i]) for i in range(n_leaves)]

    else:
        labels = ['MG %s' % metagene_order.index(i) for i in range(n_leaves)]

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
                              vmax=0.5,
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
                      vmax=vmax,
                      cmap='Reds')
    axs.set_xlabel('Metagenes')
    axs.set_ylabel('Genes')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    res_df=pd.DataFrame(x_given_xHat[chosen_indeces, :][:, metagenes_to_show])
    res_df.index=gene_labels
    res_df.columns=metagenes_to_show

    return res_df


def get_represented_groups(adata_bioib,
                           bioib_obj,
                           group_order,
                           key='bioIB_MG_group_of_interest_mapping',
                           metagene_order=None,
                           cmap='Blues',vmax=0.8,
                           title=None,
                           figsize=(10,8),
                           save_dir=None):
    if metagene_order is not None:
        cell_to_metagene = adata_bioib.uns[key][:, metagene_order]
    else:
        cell_to_metagene = adata_bioib.uns[key]

    sorted_indeces = [bioib_obj.groups_of_interest.index(g) for g in group_order]
    cell_to_metagene=cell_to_metagene[sorted_indeces,:]

    n_metagenes=cell_to_metagene.shape[1]
    cell_mg_df = pd.DataFrame(cell_to_metagene, index=group_order,
                              columns=np.arange(n_metagenes))
    plt.figure(figsize=figsize)
    ax=sns.heatmap(cell_mg_df, cmap=cmap, vmax=vmax,
                   cbar_kws={'label': 'Score'})
    ax.collections[0].colorbar.set_ticks([])
    plt.yticks(rotation=0)
    plt.xlabel('Metagenes')
    plt.ylabel('Cell groups')

    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir, bbox_inches='tight')
    plt.show()
    return cell_mg_df

def gene_to_MG_map(adata,
                   gene_to_metagene,
                    n_top_genes=5):

    chosen_indeces = []

    for mtg in range(gene_to_metagene.shape[1]):
        representative_genes_arr = np.argsort(gene_to_metagene[:, mtg])[::-1][:n_top_genes]
        for g in representative_genes_arr:
            print(g)
            print(chosen_indeces)
            if g in chosen_indeces:
                continue
            chosen_indeces.append(g)
    gene_labels = np.array(adata.var_names)[chosen_indeces]

    res_df=pd.DataFrame(gene_to_metagene[chosen_indeces, :])
    res_df.index=gene_labels
    res_df.columns=np.arange(gene_to_metagene.shape[1])
    return res_df

def get_representative_genes(adata,
                             n_top_genes=5,
                             key='bioIB_gene_MG_mapping',
                             figsize=(10,8),
                             cmap='Reds',
                             vmax=0.2,
                             title=None,
                             save_dir=None,
                             metagene_order=None,
                             ):
    if metagene_order is not None:
        gene_to_metagene = adata.uns[key][:, metagene_order]
    else:
        gene_to_metagene = adata.uns[key]
    df=gene_to_MG_map(adata, gene_to_metagene, n_top_genes=n_top_genes)
    # Step 1: Get top 5 genes for each metagene based on score
    top_genes_per_meta = {
        meta: df[meta].nlargest(n_top_genes).sort_values(ascending=False)
        for meta in df.columns
    }
    #Step 2: Construct a 5x5 DataFrame with gene names
    gene_table = pd.DataFrame({
        meta: top_genes_per_meta[meta].index.tolist()
        for meta in df.columns
    }, index=[f'Top{i+1}' for i in range(n_top_genes)])
    #Step 3: Construct a matching 5x5 DataFrame with scores
    score_table = pd.DataFrame({
        meta: top_genes_per_meta[meta].values.tolist()
        for meta in df.columns
    }, index=[f'Top{i+1}' for i in range(n_top_genes)])
    #Step 4: Plot heatmap with gene names as annotations
    plt.figure(figsize=figsize)
    # score_table=score_table/np.sum(score_table, axis=0)
    ax = sns.heatmap(score_table, annot=gene_table,
                     fmt='', cmap=cmap,
                     vmax=vmax,
                     cbar_kws={'label': 'Score'})
    plt.yticks(rotation=0)
    ax.collections[0].colorbar.set_ticks([])
    if not title:
        plt.title('Top Genes Representing Each Metagene')
    else:
        plt.title(title)
    plt.ylabel('Ranked Genes')
    plt.xlabel('Metagenes')
    plt.tight_layout()
    if save_dir is not None:
        plt.savefig(save_dir, bbox_inches='tight')
    plt.show()
    return gene_table, score_table


