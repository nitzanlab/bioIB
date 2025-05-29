import numpy as np
import jax
import jax.numpy as jnp
from anndata import AnnData
from typing import Any, Optional

from .utils import (
    D_kl,
    D_kl_jit,
    calc_I_xy,
    _get_selected_indices,
    _iIB_random_init,
    _get_selected_index,
    _get_linkage_matrix,
)





class bioIB:
    """bioIB object.

    Parameters
    ----------
    adata
        Annotated data object.
    signal_of_interest
        Observation key in adata defining the signal of interest.
    groups_of_interest
        List of labels of signal of interest chosen for the analysis.
    beta
        The value of initial beta for reverse annealing.
    n_markers
        Number of top gene markers per metagene to return
    num_betas
        Number of betas with respect to which the data is compressed in reverse-annealing
    epsilon
        a threshold parameter used to define convergence based on the difference between previous and current iterations
    min_iter
        a minimum number of iterations before convergence
    kwargs
        Keyword arguments for bioIB compression.


    Examples
    ----------
    > import scanpy as sc
    > import bioib
    > adata = sc.read(...)
    > bioib_organ = bioib.bioIB(
        adata=adata,
        signal_of_interest='organ',
        groups_of_interest=['Yolk_Sac', 'Spleen', 'Liver', 'Skin', 'Kidney'],
        copy=True)
    > bioib_organ.compress()
    """

    def __init__(
            self,
            adata: AnnData,
            signal_of_interest: str,
            groups_of_interest: list,
            bulk=False,
            beta: Optional[float] = 20.0,
            n_markers: Optional[int] = 50,
            num_betas: Optional[int] = 400,
            epsilon: Optional[float] = 1e-2,
            min_iter: Optional[float] = 1e+7,
            copy: bool = False,
            **kwargs: Any,
    ):
        self.adata = adata.copy() if copy else adata
        self._raw_adata = adata.copy()
        self.signal_of_interest = signal_of_interest
        self.groups_of_interest = groups_of_interest
        self.bulk = bulk
        self.beta = beta
        self.n_markers = n_markers
        self._num_betas = num_betas
        self.epsilon = epsilon
        self._min_iter = min_iter

        self._betas = None
        self._info_vals = None
        self._probs_dict = None # ZP: may be unnecessary - SD: I think it's good to have as an attribute, because it stores all the major IB output

        self._compressed_matrices = None
        self._unique_indices = None
        self._rounded_matrices = None
        self._selected_indices = None
        self._selected_n_metagene_list = None

        self.metagenes_to_cell_states = None
        self._copy = copy



    @property
    def adata(self) -> AnnData:
        """The bioIB annotated data object."""
        return self._adata

    @adata.setter
    def adata(self, adata: AnnData) -> None:
        self._adata = adata

    def _init_sc_probability_matrices(self):
        """Computes the initial probability matrices for the input to Information bottleneck

        Parameters
        ----------
        No additional parameters, uses the input count matrix and annotations

        Returns:
        ----------
        p_cx
            Joint probability matrix of cells (C) and genes (X).
        p_c_mid_x
            Conditional probability matrix of cells (C) given genes (X).
        p_y_mid_x
            Conditional probability matrix of cell states of interest (Y) given genes (X).
        p_y
            Cell state of interest (Y) probability vector.
        p_x
            Marginal gene (X) probability vector.
        """
        from scipy.sparse import issparse
        if issparse(self.adata.X):
            cell_gene_joint_prob = self.adata.X.todense() / np.sum(self.adata.X)  # cells X genes
        else:
            cell_gene_joint_prob = self.adata.X / np.sum(self.adata.X)  # cells X genes
        marginal_gene_prob = np.sum(cell_gene_joint_prob, axis=0)
        marginal_cell_prob = np.sum(cell_gene_joint_prob, axis=1)
        cell_given_gene_prob = cell_gene_joint_prob / marginal_gene_prob  # cells X genes

        group_given_gene_prob = np.zeros([len(self.groups_of_interest), self.adata.shape[1]])  # groups X genes
        group_prob = np.zeros(len(self.groups_of_interest))

        for i, group in enumerate(self.groups_of_interest):
            cell_indices = np.where(self.adata.obs[self.signal_of_interest] == group)[0]
            group_given_gene_prob[i, :] = np.sum(cell_given_gene_prob[cell_indices, :], axis=0)
            group_prob[i] = np.sum(marginal_cell_prob[cell_indices])

        return {
            "p_cx": cell_gene_joint_prob,
            "p_c_mid_x": cell_given_gene_prob,
            "p_y_mid_x": group_given_gene_prob,
            "p_y": group_prob.reshape(-1, 1),
            "p_x": marginal_gene_prob.reshape(-1,1)
        }

    def _init_bulk_probability_matrices(self):
        """Computes the initial probability matrices for the bulk input to Information bottleneck

        Parameters
        ----------
        No additional parameters, uses the input count matrix and annotations

        Returns:
        ----------
        p_cx
            Joint probability matrix of cells (C) and genes (X).
        p_c_mid_x
            Conditional probability matrix of cells (C) given genes (X).
        p_y_mid_x
            Conditional probability matrix of cell states of interest (Y) given genes (X).
        p_y
            Cell state of interest (Y) probability vector.
        p_x
            Marginal gene (X) probability vector.
        """
        mean_expression = self.adata.to_df().groupby(self.adata.obs[self.signal_of_interest]).mean()
        mean_expression=mean_expression.loc[self.groups_of_interest, :]
        mean_array=np.array(mean_expression)
        group_gene_joint_prob=mean_array/np.sum(mean_array)
        marginal_gene_prob=np.sum(group_gene_joint_prob, axis=0)
        marginal_group_prob=np.sum(group_gene_joint_prob, axis=1)
        group_given_gene_prob=group_gene_joint_prob/marginal_gene_prob


        return {
            "p_y_mid_x": group_given_gene_prob,
            "p_y": marginal_group_prob.reshape(-1, 1),
            "p_x": marginal_gene_prob.reshape(-1,1)
        }

    def _init_probability_matrices(self):
        if self.bulk:
            probs_dict=self._init_bulk_probability_matrices()
        else:
            probs_dict=self._init_sc_probability_matrices()
        return probs_dict


    def _calculate_infogain(self, p_x, p_y, p_y_mid_x):

        """ Sorts genes by their Information Gain value;
        Annotates adata with the calculated gene infogain values and their indeces of sorted infogain;
        Returns a dictionary with infogain calculations.
        Parameters
        ----------
        p_x
            Marginal gene (X) probability vector
        p_y
            Cell state of interest (Y) probability vector
        p_y_mid_x
            Conditional probability matrix of cell states of interest given the genes

        Returns
        ----------
        sorted_infogain
            A vector with sorted infogain values
        soted_indeces
            A vector with gene indeces corresponding to the infogain values in sorted_infogain
        sorted_gene_names
            Gene names corresponding to the infogain values in sorted_infogain
        """

        Dkl_vec = D_kl(p_y_mid_x, p_y.reshape(-1, 1))
        infogain = np.array(Dkl_vec) * np.array(p_x.reshape(-1, 1))
        infogain = infogain.T[0, :]
        self.adata.varm['Infogain value'] = infogain
        self.adata.varm['Infogain index'] = np.argsort(infogain)[::-1]

        sorted_infogain = np.sort(infogain)[::-1]
        sorted_indices = np.argsort(infogain)[::-1]
        sorted_gene_names = np.array(self.adata.var_names)[sorted_indices]

        return {
            "sorted_infogain": sorted_infogain,
            "sorted_indeces": sorted_indices,
            "sorted_gene_names": sorted_gene_names
        }




    def reduce_to_infogenes(self, n_infogenes):
        """Reduces the input adata to n genes with the highest infogain value
        Parameters
        n_infogenes
            Number of top genes with the highest infogain value for which to compute the compressed representation
        Returns
            Nothing, just updates the input adata
        """
        init_prob =self._init_probability_matrices()
        infogain_vals = self._calculate_infogain(init_prob['p_x'], init_prob['p_y'], init_prob['p_y_mid_x'])
        infogenes = infogain_vals['sorted_gene_names'][:n_infogenes]
        self.adata = self.adata[:, infogenes]

    def _reverse_annealing(self):
        """ Gradually compresses the data in reverse-annealing process
        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        logbeta = np.log2(self.beta)
        betas = 2 ** np.linspace(logbeta, 0, self._num_betas)
        self._betas = betas
        betas_to_print = betas[::self._num_betas//20]
        probs_dict = self._init_probability_matrices()
        p_xhat=probs_dict["p_x"].copy()
        p_y_mid_xhat=probs_dict["p_y_mid_x"].copy()
        d_x = probs_dict["p_x"].size
        d_xhat = d_x
        p_xhat_mid_x=jnp.identity(d_x)


        probs_dict["p_xhat"] = []
        probs_dict["p_y_mid_xhat"] = []
        probs_dict["p_xhat_mid_x"] = []
        probs_dict["p_x_mid_xhat"] = []

        I_x_xhat = []
        I_xhat_y = []

        for beta in betas:
            if beta in betas_to_print:
                print('beta=%s' % beta)
            itr = 0
            err = self.epsilon + 1
            while err > self.epsilon and itr < self._min_iter:
                dkl_jnp = D_kl_jit(probs_dict["p_y_mid_x"].T, p_y_mid_xhat.T)
                log_new_xHat_mid_x = jnp.log(p_xhat.reshape(d_xhat, 1)) - (beta * dkl_jnp).T
                p_xhat_mid_x_update=jax.nn.softmax(log_new_xHat_mid_x, axis=0)
                err = jnp.sum(abs(p_xhat_mid_x_update - p_xhat_mid_x)) if (itr > 0) else np.inf
                p_xhat=jnp.dot(p_xhat_mid_x_update, probs_dict["p_x"])
                p_x_mid_xhat=(p_xhat_mid_x_update * probs_dict["p_x"].reshape(1, d_x)).T / p_xhat.reshape(1, d_xhat)

                if np.isnan(p_x_mid_xhat).any():
                    p_x_mid_xhat = p_x_mid_xhat .at[np.isnan(p_x_mid_xhat)].set(0)
                if np.isinf(p_x_mid_xhat).any():
                    p_x_mid_xhat = p_x_mid_xhat .at[np.isinf(p_x_mid_xhat)].set(0)

                p_y_mid_xhat = jnp.dot(probs_dict["p_y_mid_x"], p_x_mid_xhat)

                if np.isnan(p_y_mid_xhat).any():
                    p_y_mid_xhat = p_y_mid_xhat.at[np.isnan(p_y_mid_xhat)].set(0)

                itr += 1
                p_xhat_mid_x = p_xhat_mid_x_update

            probs_dict["p_xhat"].append(p_xhat)
            probs_dict["p_y_mid_xhat"].append(p_y_mid_xhat)
            probs_dict["p_xhat_mid_x"].append(p_xhat_mid_x)
            probs_dict["p_x_mid_xhat"].append(p_x_mid_xhat)

            I_x_xhat.append(calc_I_xy(p_x_mid_xhat, probs_dict["p_x"], p_xhat))
            I_xhat_y.append(calc_I_xy(p_y_mid_xhat, probs_dict["p_y"], p_xhat))

        self._info_vals = {
            "I_x_xhat": I_x_xhat,
            "I_xhat_y": I_xhat_y
        }

        self._probs_dict = probs_dict

        return

    def _filter_collapsed_genes(self):
        """ Filters the reverse_annealing output by deleting the collapsed metagenes (with zero probability)
        Parameters
        ----------
        None

        Returns
        ----------
        updated probs dict
        """

        nan_genes = np.where(np.sum(np.asarray(self._probs_dict["p_y_mid_xhat"])[-1, :, :], axis=0) == 0)[0]
        probs_dict_update = self._probs_dict.copy()

        for key in ["p_xhat_mid_x", "p_xhat"]:
            for i, arr in enumerate(self._probs_dict[key]):
                probs_dict_update[key][i] = np.delete(arr, nan_genes, 0)


        for key in ["p_y_mid_xhat", "p_x_mid_xhat"]:
            for i, arr in enumerate(self._probs_dict[key]):
                probs_dict_update[key][i] = np.delete(arr, nan_genes, 1)

        self._probs_dict = probs_dict_update

        return



    def annotate_adata(self, n_leaves=None, initial_annotation=True):
        """ Annotates adata with the metagene information
        Parameters
        ----------
        n_leaves
            Number of metagenes (out of possible reverse-annealing outcomes) for annotation.
            If not provided, the maximal number of metagenes generated for the given beta will be used.
        initial_annotation
            True if this is the function ic called for the first annotation of adata with bioiB output.
            False if the function is used for re-annotation for a different number of metagenes.
        Returns
        ----------
        adata
            Annotated adata
        """
        from scipy.sparse import issparse
        if not n_leaves:
            n_leaves = self._selected_n_metagene_list[-1]
        if initial_annotation:
            adata = self.adata
        else:
            adata = self._raw_adata.copy()

        # annotating the metagene expression as observations in single cells
        selected_ind = self._selected_indices[np.where(self._selected_n_metagene_list == n_leaves)[0][0]]
        x_given_xHat = self._compressed_matrices['p_x_mid_xhat'][selected_ind]
        average_mat = np.zeros([adata.shape[0], n_leaves])
        marker_table = []
        for mtg in range(n_leaves):
            factors = x_given_xHat[:, mtg]
            if issparse(self.adata.X):
                multiplied_by_factors = np.multiply(adata.X.todense(), factors.reshape(1, -1))
            else:
                multiplied_by_factors = np.multiply(adata.X, factors.reshape(1, -1))
            average_vector = np.sum(multiplied_by_factors, axis=1)
            average_mat[:, mtg] = average_vector.reshape(-1, )
            adata.obs['Metagene %s' % mtg] = np.array(average_vector)
            top_genes = adata.var_names[np.argsort(factors)[::-1][:self.n_markers]]
            marker_table.append(list(top_genes))

        # creating a new layer in adata.obsm to contain the compressed adata
        adata.obsm['bioIB_compressed_data'] = np.array(average_mat)
        # adding a metagene-gene mapping and markers
        adata.uns['bioIB_gene_MG_mapping'] = np.array(x_given_xHat)
        # adding the marker list
        adata.uns['bioIB_markers'] = np.array(marker_table)
        # adding the MG-cell state of interest link
        y_given_xHat = self._compressed_matrices['p_y_mid_xhat'][selected_ind]
        adata.uns['bioIB_MG_group_of_interest_mapping'] = np.array(y_given_xHat)

        return adata

    def compress(self):
        """ Gradually compresses the data.
        Parameters
        ----------
        None.

        Returns
        ----------
        adata
            Adata annotated with bioIB output.
        """

        self._reverse_annealing()
        self._filter_collapsed_genes()
        [selected_indices, selected_n_metagene_list,
         compressed_matrices, unique_indices, rounded_matrices]=_get_selected_indices(self._betas,
                                                                                      self._probs_dict['p_y_mid_xhat'],
                                                                                      self._probs_dict['p_x_mid_xhat'])
        self._selected_indices=selected_indices
        self._selected_n_metagene_list=selected_n_metagene_list
        self._compressed_matrices=compressed_matrices
        self._unique_indices=unique_indices
        self._rounded_matrices=rounded_matrices

        self.annotate_adata()
        if self._copy:
            return self.adata
        return

    def reannotate_adata(self, n_leaves):
        """ Re-annotates adata with a different number of metagenes
        Parameters
        ----------
        n_leaves
            Number of metagenes for adata annotation.

        Returns
        ----------
        adata
            Adata annotated with bioIB output.
        """
        return self.annotate_adata(n_leaves=n_leaves, initial_annotation=False)



    def _iIB_alg_jax(self, n_metagenes, beta=1e4):
        """ IB algorithm for flat gene clustering.
        Parameters
        ----------
        n_metagenes
            The desired number of flat gene clusters.
        beta
            Beta value

        Returns
        ----------
        p_y_mid_xhat
            Optimized conditional probability matrix of cell labels (y) given the metagenes (xHat).
        p_xhat_mid_x
            Optimized conditional probability matrix of metagenes (Xhat) given the genes (X).
        p_x_mid_xhat
            Optimized conditional probability matrix of genes (x) given the metagenes (xHat).
        I_ty
            Associated mutual information between cell labels and metagenes.

        """
        # Initialization
        if not self._probs_dict:
            self._probs_dict=self._init_probability_matrices()
        [p_xhat, _, p_y_mid_xhat, p_xhat_mid_x]=_iIB_random_init(self._probs_dict['p_x'], self._probs_dict['p_y_mid_x'], n_metagenes)
        itr = 0
        err = 100
        d_x = self.adata.shape[1]
        d_xHat = p_xhat.size

        while err > self.epsilon and itr < self._min_iter:
            dkl_jnp = D_kl_jit(self._probs_dict["p_y_mid_x"].T, p_y_mid_xhat.T)
            log_new_xHat_given_x = jnp.log(p_xhat.reshape(d_xHat, 1)) - (beta * dkl_jnp).T
            p_xhat_mid_x_update = jax.nn.softmax(log_new_xHat_given_x, axis=0)
            err = jnp.sum(abs(p_xhat_mid_x_update - p_xhat_mid_x)) if (itr > 0) else np.inf
            p_xhat = jnp.dot(p_xhat_mid_x_update, self._probs_dict["p_x"].reshape(-1, 1))
            p_x_mid_xhat = (p_xhat_mid_x_update * self._probs_dict["p_x"].reshape(1, d_x)).T / p_xhat.reshape(1, d_xHat)

            if np.isnan(p_x_mid_xhat).any():
                p_x_mid_xhat = p_x_mid_xhat.at[np.isnan(p_x_mid_xhat)].set(0)
            if np.isinf(p_x_mid_xhat).any():
                p_x_mid_xhat = p_x_mid_xhat.at[np.isinf(p_x_mid_xhat)].set(0)

            p_y_mid_xhat = jnp.dot(self._probs_dict["p_y_mid_x"], p_x_mid_xhat)

            if np.isnan(p_y_mid_xhat).any():
                p_y_mid_xhat = p_y_mid_xhat.at[np.isnan(p_y_mid_xhat)].set(0)
            itr += 1
            p_xhat_mid_x = p_xhat_mid_x_update

        I_ty = calc_I_xy(p_y_mid_xhat, self._probs_dict["p_y"], p_xhat)

        return [p_y_mid_xhat, p_xhat_mid_x, p_x_mid_xhat, I_ty]

    def flat_clustering(self, n_metagenes, n_iters=3):
        """ bioIB function for flat clustering, merging the genes into |n_clusters| metagenes.
        Parameters
        ----------
        n_metagenes
            The desired number of flat gene clusters.
        n_iters
            Number of random initializations for bioIB.

        Returns:
        ----------
        Nothing, annotates the input adata with the generated metagenes

         """
        from scipy.sparse import issparse
        I_ty_list = []
        successful_iterations = 0
        all_y_given_xhat = []
        all_x_given_xhat = []
        while successful_iterations < n_iters:
            [p_y_mid_xhat, p_xhat_mid_x, p_x_mid_xhat, I_ty] = self._iIB_alg_jax(n_metagenes)
            real_n_clus = np.unique(p_xhat_mid_x, axis=0).shape[0]
            if real_n_clus < n_metagenes:
                continue
            successful_iterations += 1
            I_ty_list.append(I_ty)
            all_y_given_xhat.append(p_y_mid_xhat)
            all_x_given_xhat.append(p_x_mid_xhat)

        max_Ity = np.argmax(I_ty_list)

        average_mat = np.zeros([self.adata.shape[0], n_metagenes])
        marker_table = []
        for mtg in range(n_metagenes):
            factors = all_x_given_xhat[max_Ity][:, mtg]
            if issparse(self.adata.X):
                multiplied_by_factors = np.multiply(self.adata.X.todense(), factors.reshape(1, -1))
            else:
                multiplied_by_factors = np.multiply(self.adata.X, factors.reshape(1, -1))
            average_vector = np.sum(multiplied_by_factors, axis=1)
            average_mat[:, mtg] = average_vector.reshape(-1, )
            self.adata.obs['Metagene %s' % mtg] = np.array(average_vector)
            top_genes = self.adata.var_names[np.argsort(factors)[::-1][:self.n_markers]]
            marker_table.append(list(top_genes))

        self.adata.uns['bioIB_flat_%s_MGs_gene_to_MG' % n_metagenes] = np.array(all_x_given_xhat[max_Ity])
        self.adata.uns['bioIB_flat_%s_MGs_group_of_interest_to_MG' % n_metagenes] = np.array(all_y_given_xhat[max_Ity])
        # creating a new layer in adata.obsm to contain the compressed adata
        self.adata.obsm['bioIB_compressed_data_flat_%s_MGs' % n_metagenes] = np.array(average_mat)
        # adding the marker list
        self.adata.uns['bioIB_markers_flat_%s_MGs' % n_metagenes] = np.array(marker_table)

        return self.adata



    def map_metagenes_to_cell_states(self, n_leaves=None):
        """
        Parameters
        ----------
        n_leaves
            The maximal number of metagenes (compression level)  - number of leaves in the plotted metagene hierarchy

        Returns
        -------
        Nothing, annotates the abject with the self.metagenes_to_cell_states attribute, linking every metagene to the corresponding cell state of interest

        """
        if not n_leaves:
            n_leaves = self._selected_n_metagene_list[-1]

        selected_index=_get_selected_index(n_leaves, self._selected_n_metagene_list, self._selected_indices)
        y_given_xHat = self._compressed_matrices['p_y_mid_xhat'][selected_index]

        mg_to_lbl_dict = {}
        for mtg in range(n_leaves):
            max_ind = np.argmax(y_given_xHat[:, mtg])
            mg_to_lbl_dict['MG %s' % mtg] = self.groups_of_interest[max_ind]
        self.metagenes_to_cell_states = mg_to_lbl_dict

    def _get_linkage_matrix(self, n_leaves):
        """
        Parameters
        ----------
        n_leaves
            The maximal number of metagenes (compression level)  - number of leaves in the plotted metagene hierarchy.

        Returns
        -------
        Linkage matrix for plotting the dendrogram.

        """
        selected_index = _get_selected_index(n_leaves, self._selected_n_metagene_list, self._selected_indices)
        return _get_linkage_matrix(n_leaves, selected_index, self._unique_indices['p_y_mid_xhat'], self._rounded_matrices['p_y_mid_xhat'])


