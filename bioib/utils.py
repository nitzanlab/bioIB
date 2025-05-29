import numpy as np
from jax import jit
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.special import rel_entr


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
            dkl[i, j] = np.sum(rel_entr(p[:, i], q[:, j]))

    return dkl


@jit
def kl_fn(x, y):
    r = jnp.where((x > 0) & (y > 0), x * jnp.log(x / y), 0.0)
    return r


@jit
def D_kl_jax(p, q):
    """
    Calculate Kullback-Leibler divergence using jax

    D_kl[p || q] = \sum_x p(x) * ln (p(x)/q(x))
    """
    p_expanded = p[:, None, :]
    q_expanded = q[None, :, :]
    kl_matrix = kl_fn(p_expanded, q_expanded)
    kl_divergence = jnp.sum(kl_matrix, axis=-1)
    return kl_divergence


D_kl_jit = jit(D_kl_jax)


def calc_I_xy(p_x_mid_y, p_x, p_y):
    """
    :param p_x_mid_y: the conditional probability (|X|, |T|)
    :param p_y, p_x: The marginal probabilities
    :return: I_xy: The Mutual information I(X;Y) = <D_kl[p(x|y) || p(x)]>_{p(y)}
    """
    I_xy = 0
    for i, p_y_i in enumerate(p_y):
        I_xy += p_y_i * jnp.sum(kl_fn(p_x_mid_y[:, i], p_x))
    return I_xy


def _round_IB_res(beta, array):
    """ Rounds the reverse_annealing output matrices as a function of beta
    Parameters
    ----------
    beta
        The value of beta used to compute the input matrix
    array
        The input matrix to be rounded

    Returns
    ----------
    array_rounded
        The rounded matrix
    """
    prec = 1 / (beta * 10) #initially = 100
    acc = int(abs(np.round(np.log10(prec))))
    array_rounded = np.around(array, acc)
    return array_rounded

def _compress_matrix(betas, cond_probability_array):  # move to utils?
    """Extracts a list of compressed data representations (cell labels X metagenes and genes X metagenes) from the reverse annealing output
    Parameters
    ----------
    betas
        List of beta values used in reverse annealing
    cond_probability_array
        Array of conditional probability matrices (p_y_mid_xhat or p_x_mid_xhat) representing the reverse annealing output.

    Returns
    ----------
    n_metagene_list
        List of numbers of unique metagenes produced at each beta value.
    compressed_matrices
        List of compressed matrices for each beta value.
    unique_indices_list
        Lists of indices (for each beta) of unique metagenes from the original probability matrix that can be used for its recovery.
    rounded_mat_list
        List of rounded, non-compressed input probability matrices for each beta value
    """
    n_metagene_list = []
    compressed_matrices = []
    unique_indices_list = []
    rounded_mat_list = []
    for ii in range(len(cond_probability_array)):
        prob_mat = cond_probability_array[ii]
        rounded_mat = _round_IB_res(betas[ii], prob_mat)
        rounded_mat_list.append(rounded_mat)
        [_, recovering_indices, unique_indices] = np.unique(
                                                    rounded_mat,
                                                    axis=1,
                                                    return_index=True,
                                                    return_inverse=True
                                                )
        compressed_mat = rounded_mat[:, sorted(recovering_indices)]
        compressed_matrices.append(compressed_mat)
        fixed_unique_indices = [recovering_indices[i] for i in unique_indices]
        unique_indices_list.append(np.array(fixed_unique_indices))
        n_metagene_list.append(compressed_mat.shape[1])
    return [n_metagene_list, compressed_matrices, unique_indices_list, rounded_mat_list]


def _get_rand_mat(gene_metagene_mapping):
    """Constructs a clustering matrix indicating the mapping of genes to metagenes for further calculation of the Rand index.
    Parameters:
    ----------
    gene_metagene_mapping
        List of metagene indices associated with each gene.

    Returns:
    ----------
    rand_mat
        Indicator matrix of mapping of genes to metagenes """

    dim = len(gene_metagene_mapping)
    rand_mat = np.zeros([dim, dim])
    for i in range(dim):
        other_locs = np.where(gene_metagene_mapping == i)[0]
        rand_mat[i, other_locs] = [1] * len(other_locs)
    return rand_mat

def _get_selected_indices(betas, p_y_mid_xhat, p_x_mid_xhat):
    """Selects beta indices that yielded the same gene-to-metagene mapping both based on y|xHat and on x|xHat.
    Parameters:
    ----------
    betas
        List of beta values used in reverse annealing
    p_y_mid_xhat
        Array of conditional probability matrices of cell states of interest (Y) given the metagenes (Xhat) representing the reverse annealing output.
    p_x_mid_xhat
        Array of conditional probability matrices of genes (X) given the metagenes (Xhat) representing the reverse annealing output.

    Returns:
    ----------
    selected_indices
        List of beta indices at which the gene-to-metagene mapping based on y|xHat and on x|xHat is the same (Rand measure=1).
    selected_n_metagene_list
        List of numbers of unique metagenes for each beta value in selected_indeces.
    compressed_matrices
        Dictionary with compressed matrices with the number of unique metagenes specified in selected_n_metagene_list:
            p_y_mid_xhat - compressed matrices of cell states of interest (Y) given the metagenes (Xhat)
            p_x_mid_xhat - compressed matrices of genes (X) given the metagenes (Xhat)
    unique_indices
        Dictionary with lists of indices of unique metagenes in compressed_matrices to recover the original probability matrices.
            p_y_mid_xhat -  for recovery of the original probability of cell states of interest (Y) given the metagenes (Xhat).
            p_x_mid_xhat - for recovery of the original probability of genes (X) given the metagenes (Xhat).
    rounded_matrices
        Dictionary with uncompressed but rounded matrices.
            p_y_mid_xhat -  rounded matrices of conditional probability of cell states of interest (Y) given the metagenes (Xhat).
            p_x_mid_xhat - frounded matrices of conditional probability of genes (X) given the metagenes (Xhat).
    """

    [_, compressed_matrices_y, unique_indices_list_y, rounded_mat_list_y] = _compress_matrix(betas, p_y_mid_xhat)
    [n_metagene_list_x, compressed_matrices_x, unique_indices_list_x, rounded_mat_list_x] = _compress_matrix(betas, p_x_mid_xhat)

    rand_matrices_x = []
    rand_matrices_y = []
    rand_measures = []

    for p in range(len(n_metagene_list_x)):
        rand_mat_x = _get_rand_mat(unique_indices_list_x[p])
        dim = rand_mat_x.shape[0]
        rand_matrices_x.append(rand_mat_x)
        rand_mat_y = _get_rand_mat(unique_indices_list_y[p])
        rand_matrices_y.append(rand_mat_y)
        rand = 1 - (np.sum((rand_mat_x - rand_mat_y) ** 2) / (dim * (dim - 1)))
        rand_measures.append(rand)

    matching_points = np.where(np.array(rand_measures) == 1)[0]
    n_matching_metagenes = np.array(n_metagene_list_x)[matching_points]
    [selected_n_metagene_list, indices, _] = np.unique(n_matching_metagenes,
                                                                 return_index=True,
                                                               return_inverse=True)
    selected_indices = np.array(matching_points)[indices]
    compressed_matrices={'p_y_mid_xhat': compressed_matrices_y,
                         'p_x_mid_xhat': compressed_matrices_x}
    unique_indices={'p_y_mid_xhat': unique_indices_list_y,
                    'p_x_mid_xhat': unique_indices_list_x}
    rounded_matrices={'p_y_mid_xhat': rounded_mat_list_y,
                         'p_x_mid_xhat': rounded_mat_list_x}


    return [selected_indices, selected_n_metagene_list, compressed_matrices, unique_indices, rounded_matrices]

def _iIB_random_init(p_x, p_y_mid_x, n_metagenes):
    """ Generates random probability matrices for the initiation of the bioIB flat clustering function.

    Parameters
    ----------
    p_x
        Marginal gene (X) probability
    p_y_mid_x
        Conditional probability matric of cell sttes of interest (Y) given the genes.
    n_metagenes
        The desired number of flat gene clusters.

    Returns
    ----------
    p_xHat
        Random metagene probability vector.
    p_x_mid_xhat
        Random conditional probability matrix of genes (X) given the metagenes (Xhat).
    p_y_mid_xhat
        Random conditional probability matrix of cell labels (Y) given the metagenes (Xhat).
    p_xhat_mid_x
        Random conditional probability matrix of metagenes (Xhat) given the genes (X);

    """
    n_genes=p_x.size
    n_repeats = np.floor_divide(n_genes, n_metagenes)
    remainder = n_genes % n_metagenes
    ones_indices = list(range(n_metagenes)) * n_repeats
    ones_indices.extend(np.random.randint(0, n_metagenes - 1, remainder))
    np.random.shuffle(ones_indices)
    p_xhat_mid_x = np.zeros([n_metagenes, n_genes])
    for i in range(n_genes):
        p_xhat_mid_x[ones_indices[i], i] = 1

    p_xhat = p_xhat_mid_x @ p_x.reshape(-1, 1)
    p_xhat = p_xhat / np.sum(p_xhat)

    p_x_mid_xhat = np.multiply(np.array(p_xhat_mid_x.T), np.array(p_x.reshape(-1, 1))) / (
        p_xhat.reshape(1, -1))
    p_x_mid_xhat = p_x_mid_xhat / np.sum(p_x_mid_xhat, axis=0)
    p_y_mid_xhat = p_y_mid_x @ p_x_mid_xhat

    return [p_xhat, p_x_mid_xhat, p_y_mid_xhat, p_xhat_mid_x]


def _get_selected_index(n_metagenes, selected_n_metagene_list, selected_indices_list):

    return selected_indices_list[np.where(selected_n_metagene_list == n_metagenes)[0][0]]

def _get_representative_genes(n_leaves, selected_index, unique_indices_y, rounded_matrices_y):  # move to utils
    """
    Returns the indices of the genes most representative of the metagenes,
    and extracts the subset of y_given_xHat matrix for hierarchy plotting.
    Parameters
    ----------
    n_leaves
        The maximal number of metagenes (compression level)  - number of leaves in the plotted metagene hierarchy
    selected_index
        THe index (in the reverse annealing output arrays) of the desired compressed representation
    unique_indices_y
        List of indices of unique metagenes in compressed p_y_mid_xhat to recover the original p_y_mid_xhat.
    rounded_matrices_y
        List of uncompressed rounded p_y_mid_xhat matrices.

    Returns
    ----------
    gene_indices_list
        List of gene indices represenattive of the metagenes in the specified order.
    p_y_mid_xhat_sbst
        A subset of the original rounded_y_given_xHat matrix, including only the representative genes and starting from the chosen hierarchy depth.
    """

    gene_indices_list = []
    for mtg in range(n_leaves):
        gene_indices_list.append(np.unique(unique_indices_y[selected_index])[mtg])
    p_y_mid_xhat_sbst = np.array(rounded_matrices_y)[:, :, gene_indices_list][selected_index:]
    return [gene_indices_list, p_y_mid_xhat_sbst]

def _get_merge_list(p_y_mid_xhat_sbst):  # move to utils
    """

    Parameters
    ----------
    p_y_mid_xhat_sbst
        A subset of the original rounded_y_given_xHat matrix, including only the representative genes and starting from the chosen hierarchy depth.

    Returns
    -------
    merge_list
        List of metagene merging events along the reverse annealing process.

    """

    merge_list = []
    shape = p_y_mid_xhat_sbst.shape[2]
    for b in range(p_y_mid_xhat_sbst.shape[0]):
        [unique_array, inverse_indices] = np.unique(p_y_mid_xhat_sbst[b, :, :], axis=1, return_inverse=True)
        new_shape = unique_array.shape[1]
        if new_shape < shape:
            merge_list.append([])
            for new_mg in range(new_shape):
                old_indices = np.where(inverse_indices == new_mg)[0]
                if len(old_indices) > 1:
                    merge_list[-1].append(old_indices)
            shape = new_shape
    return merge_list

def _get_linkage_matrix(n_leaves, selected_index, unique_indices_y, rounded_matrices_y):
    """ Produces the linkage matrix Z for plotting the metagene dendrogram corresponding to reverse-annealing output.
    Parameters
    ----------
    n_leaves
        The maximal number of metagenes (compression level)  - number of leaves in the plotted metagene hierarchy.
    selected_index
        THe index (in the reverse annealing output arrays) of the desired compressed representation
    unique_indices_y
        List of indices of unique metagenes in compressed p_y_mid_xhat to recover the original p_y_mid_xhat.
    rounded_matrices_y
        List of uncompressed rounded p_y_mid_xhat matrices.

    Returns
    -------
    Z
        Linkage matrix for plotting the dendrogram.

    """
    [_, p_y_mid_xhat_sbst] = _get_representative_genes(n_leaves, selected_index, unique_indices_y, rounded_matrices_y)
    merge_list = _get_merge_list(p_y_mid_xhat_sbst)

    new_clusters_dict = {}
    Z = []
    distance = 0.1
    for b in merge_list:
        for merge in b:
            exists = False
            merged_new_mgs = []
            additional_mgs_merged = []
            for cl in new_clusters_dict:
                overlap = [g for g in merge if g in new_clusters_dict[cl]]
                if len(overlap) == len(merge):
                    exists = True
                    break
                elif len(overlap) > 0:
                    exists = True
                    merged_new_mgs.append(cl)
                    additional_mgs_merged = [g for g in merge if g not in new_clusters_dict[cl]]
            if not exists:
                Z.append(list(merge) + [distance, 2])
                new_clusters_dict[n_leaves] = list(merge)
                n_leaves += 1
                distance += 0.1
            elif exists:
                if len(merged_new_mgs) == 2:
                    Z.append(merged_new_mgs + [distance, len(merge)])
                    new_clusters_dict[n_leaves] = list(merge)
                    n_leaves += 1
                    distance += 0.1
                    for old_clus in merged_new_mgs:
                        del new_clusters_dict[old_clus]
                elif len(merged_new_mgs) == 1:
                    Z.append(merged_new_mgs + additional_mgs_merged + [distance, len(merge)])
                    new_clusters_dict[n_leaves] = list(merge)
                    n_leaves += 1
                    distance += 0.1
                    del new_clusters_dict[merged_new_mgs[0]]
    return Z