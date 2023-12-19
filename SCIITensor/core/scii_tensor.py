import scanpy as sc
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import squidpy as sq
import os
from itertools import product
import pickle
from scipy import sparse
import time
import anndata
from anndata import AnnData
import tensorly as tl
from tensorly.decomposition import non_negative_tucker
from tqdm import tqdm
import sys
import logging as logg
import scvelo as scv
from sklearn.cluster import KMeans
from itertools import chain
import torch
from typing import Union
import matplotlib as mpl
import random

from SCIITensor.utils.utility import _generate_LRpairs, _get_LR_connect_matrix, _get_LR_intensity, _permutation_test, get_keys, _result_combined, _mh_translate, _sorted, find_max_column_indices
from SCIITensor.plot import sankey, scii_circos, scii_net


format="%(asctime)s-%(levelname)s-%(message)s"
logg.basicConfig(format=format, level=logg.INFO)

class InteractionTensor():
    """
    Class for constructing and analyzing Spatial Cellular Interaction Intensity (SCII) matrices
    based on spatial transcriptomics data.

    The InteractionTensor class provides methods to build SCII matrices, process them,
    evaluate tensor ranks, and perform tensor factorization for further analysis.

    Attributes:
    - adata: anndata
        The spatial transcriptomics data.
    - interactionDB: str
        Database of ligand-receptor pairs.
    """
    
    def __init__(self, adata, interactionDB: str=None):
        # Save variables for this class
        self.adata = adata
        self.interactionDB = interactionDB

def build_SCII(interactiontensor: InteractionTensor=None, radius: int=200, coord_type: str="generic",
          window_size: int=200, anno_col: str='cell2loc_anno', use_raw: bool=True, interactionDB: str=None) -> list:
    """
    Construct a Spatial Cellular Interaction Intensity (SCII) matrix based on spatial transcriptomics data.

    This function calculates the interactions between different cell types within specified spatial windows
    and generates a CCI matrix containing interaction strengths.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    radius : int
        The radius for spatial neighbor search, by default 200.
    coord_type : str
        The coordinate type for spatial neighbor calculation, by default "generic".
    window_size : int
        The size of the spatial windows, by default 200.
    anno_col : str
        Column name containing cell annotations, by default 'cell2loc_anno'.
    use_raw : bool
        whether use raw counts to build CCI matrix, by default True.
    interactionDB : str
        Database of ligand-receptor pairs.

    Returns
    -------
    list of pandas.DataFrame
        A list of DataFrames containing interaction strengths between cell types in each window.
    """
    interactionDB = interactiontensor.interactionDB
    adata = interactiontensor.adata

    interactiontensor.radius = radius
    interactiontensor.coord_type = coord_type
    interactiontensor.window_size = window_size
    interactiontensor.anno_col = anno_col
    interactiontensor.use_raw = use_raw

    time_start=time.time()

    if use_raw:
        adata = adata.raw.to_adata()
    else:
        sc.pp.normalize_total(adata, inplace=True, target_sum=1e4)
        sc.pp.log1p(adata)

    # filter LR genes
    logg.info("Filter LR genes")
    filter_LRpairs, adata_sub = _generate_LRpairs(interactionDB=interactionDB, adata=adata)
    interactiontensor.filter_LRpairs = filter_LRpairs
    # with open("filter_LRpairs.pkl", "wb") as f:
    #     pickle.dump(filter_LRpairs, f)

    # nearest neighbor graph
    logg.info("Create spatial neighbor graph")
    sq.gr.spatial_neighbors(adata_sub, radius=radius, coord_type=coord_type)

    connect_matrix = adata_sub.obsp['spatial_connectivities']
    exp_connect_list = []
    for genes in filter_LRpairs:
        exp_connect_matrix = _get_LR_connect_matrix(adata_sub, genes, connect_matrix)
        exp_connect_list.append(exp_connect_matrix)

    interactiontensor.exp_connect_list = exp_connect_list
    # with open("exp_connect_list.pkl", 'wb') as f:
    #     pickle.dump(exp_connect_list, f)

    adata_sub.obs[f'x_{window_size}'] = (adata_sub.obsm['spatial'][:, 0]//window_size)*window_size+(window_size/2)
    adata_sub.obs[f'y_{window_size}'] = (adata_sub.obsm['spatial'][:, 1]//window_size)*window_size+(window_size/2)
    adata_sub.obs[f'bin{window_size}'] = adata_sub.obs[f'x_{window_size}'].astype('str') + '_' + adata_sub.obs[f'y_{window_size}'].astype("str")

    adata_sub.obs['cellname'] = adata_sub.obs.index.tolist()
    df = adata_sub.obs[['cellname', f'bin{window_size}']].reset_index()
    df['index'] = df.index.tolist()
    bin_group = df[['index', f'bin{window_size}']].groupby(f'bin{window_size}')
    indices = [indices for t, indices in bin_group.groups.items()]

    interactiontensor.indices = indices
    # with open("indices.pkl", "wb") as f:
    #     pickle.dump(indices, f)

    # adata_sub.obs.to_csv("meta_bin.csv")

    logg.info("Start build SCII matrix")
    results = []
    # i=0
    for i in tqdm(range(len(indices))):
        celltype = adata_sub.obs[anno_col].values.unique().tolist()
        combinations = list(product(celltype, repeat=2))
        store_mt = pd.DataFrame(np.zeros((len(combinations), len(filter_LRpairs))), index = ['_'.join(comb) for comb in combinations], columns=["-".join(x) for x in filter_LRpairs]).astype('float32')

        adata_sub_sub = adata_sub[indices[i], :]

        cell_types = adata_sub_sub.obs[anno_col].unique()
        cell_type_dict = dict(zip(cell_types, range(0, len(cell_types))))
        cellTypeIndex = adata_sub_sub.obs[anno_col].map(cell_type_dict).astype(int).values

        cellTypeNumber = len(adata_sub_sub.obs[anno_col].unique())

        for j in range(len(filter_LRpairs)):
            # j=0
            lr = "-".join(filter_LRpairs[j])

            data = exp_connect_list[j].tocsr()[indices[i], :][:, indices[i]].tocoo()

            senders = cellTypeIndex[data.row]
            receivers = cellTypeIndex[data.col]
            interaction_matrix = sparse.csr_matrix((data.data, (senders, receivers)), shape=(cellTypeNumber, cellTypeNumber))

            nonzero_row, nonzero_col = interaction_matrix.nonzero()
            for r in range(len(nonzero_row)):
                for c in range(len(nonzero_col)):
                    cellpair = get_keys(cell_type_dict, nonzero_row[r])[0] + "_" + get_keys(cell_type_dict, nonzero_col[c])[0]

                    store_mt.loc[cellpair][lr] = interaction_matrix[nonzero_row[r], nonzero_col[c]]
        results.append(store_mt)

    # with open("lr_mt_list.pkl", "wb") as f:   #Pickling
    #     pickle.dump(results, f)
    interactiontensor.lr_mt_list = results
    time_end=time.time()
    logg.info(f"Finish build CCI matrix - time cost {(((time_end - time_start) / 60) / 60)} h")


def process_SCII(interactiontensor: InteractionTensor=None, 
            bin_zero_remove: bool=True,
            log_data: bool=True) -> np.ndarray:

    """
    Process the Spatial Cellular Interaction Intensity (SCII) matrix list.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    bin_zero_remove : bool
        Flag indicating whether to remove spatial bins with zero intensity, by default True.
    log_data : bool
        Flag indicating whether to log-transform the data, by default True.

    Returns
    -------
    np.ndarray
        Processed 3D Celltype-Celltype Interaction (CCI) matrix.
    """

    time_start=time.time()
    lr_mt_list = interactiontensor.lr_mt_list
    final_mt = np.dstack(lr_mt_list)

    if bin_zero_remove == True:
        bin_zero_submatrix_indices = np.all(final_mt == 0, axis=(0,1))
        logg.info(f"{sum(bin_zero_submatrix_indices)} window have zero intensity")

#     if zero_remove:

#     cellpair_zero_submatrix_indices = np.all(final_mt == 0, axis=(1,2))
#     lrpair_zero_submatrix_indices = np.all(final_mt == 0, axis=(0,2))
#     bin_zero_submatrix_indices = np.all(final_mt == 0, axis=(0,1))

#     zero_indices = [cellpair_zero_submatrix_indices, lrpair_zero_submatrix_indices, bin_zero_submatrix_indices]

#     with open('zero_indices.pkl', 'wb') as f:
#         pickle.dump(zero_indices, f)

#     nonzero_submatrices = final_mt[:, :, ~bin_zero_submatrix_indices]
#     nonzero_submatrices = nonzero_submatrices[~cellpair_zero_submatrix_indices, :, :]
#     nonzero_submatrices = nonzero_submatrices[:, ~lrpair_zero_submatrix_indices, :]
#     final_mt = nonzero_submatrices
    interactiontensor.window_zero_indices = bin_zero_submatrix_indices

    # with open("window_zero_indices.pkl", "wb") as f:
    #     pickle.dump(bin_zero_submatrix_indices, f)

    final_mt = final_mt[:, :, ~bin_zero_submatrix_indices]

    cellpair = lr_mt_list[0].index
    interactiontensor.cellpair = cellpair
    # with open("cellpair.pkl", "wb") as f:
    #     pickle.dump(cellpair, f)

    lrpair = lr_mt_list[0].columns
    interactiontensor.lrpair = lrpair
    # with open("lrpair.pkl", "wb") as f:
    #     pickle.dump(lrpair, f)

    if log_data:
        final_mt = np.log1p(final_mt)

    interactiontensor.cci_matrix = final_mt
    # with open("final_mt.pkl", "wb") as f:
    #     pickle.dump(final_mt, f)

    time_end=time.time()
    logg.info(f"Finish processing LR matrix - time cost {(((time_end - time_start) / 60) / 60)} h")

def eval_SCII_rank(interactiontensor: InteractionTensor=None, num_TME_modules: int=20, num_cellpair_modules: int=20,
              device: str='cuda:0', method: str= 'tucker', 
              use_gpu: bool=True, init: str='svd', backend: str="pytorch") -> np.ndarray:
    """
    Evaluate reconstruction error of tucker decomposition of Cell-Cell Interaction (CCI) tensor matrix.

    This function reads a CCI matrix from the specified path, processes it to handle zero submatrices,
    calculates a log-transformed tensor, performs rank evaluation, and generates visualization.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    num_TME_modules : int, optional
        Number of modules for the TME (Tumor Microenvironment) to test, by default 20.
    num_cellpair_modules : int, optional
        Number of modules for cell-cell pairs and ligand-receptor pairs to test, by default 20.
    device : str, optional
        Device for computation ('cpu' or 'cuda:X'), by default 'cuda:3'.
    method : str
        Method for tensor decomposition, by default 'tucker'.
    use_gpu : bool
        Whether to use GPU for computation, by default True.
    init : str
        Initialization method for tensor decomposition, by default 'svd'.
    backend : str
        Backend for tensor operations, by default 'pytorch'.

    Returns
    -------
    np.ndarray
        A matrix representing evaluated reconstruction errors.
    """

    time_start=time.time()

    final_mt = interactiontensor.cci_matrix

    tl.set_backend(backend)

    mat = evaluate_ranks(interactiontensor, num_TME_modules=num_TME_modules, num_cellpair_modules=num_cellpair_modules, device = device, 
                         init=init, n_iter_max=10, method=method, use_gpu = use_gpu)
    interactiontensor.reconstruction_error = mat

    # with open("reconstruction_error_mt.pkl", "wb") as f:
    #     pickle.dump(mat, f)

    num_TME_modules=num_TME_modules+1

    colors = plt.cm.tab20(np.linspace(0, 1, num_TME_modules-2))

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.figure()
    for j in range(3,num_TME_modules):
        plt.plot(3+np.arange(num_TME_modules-4),mat[j][3:],label = 'rank = (TME={},ct-ct,L-R)'.format(j), color=colors[j-2])
    # plt.xlabel('x')
    plt.ylabel('reconstruction error')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.savefig('rank_eval.pdf', bbox_inches='tight')
    plt.close()

    time_end=time.time()
    logg.info(f"Finish eval SCII tensor rank - time cost {(((time_end - time_start) / 60) / 60)} h")

def evaluate_ranks(interactiontensor: InteractionTensor=None, num_TME_modules: int=20, num_cellpair_modules: int=20, device: str="cuda:0", init: str='svd', n_iter_max: int=10, 
                   method: str='tucker', use_gpu: bool=True) -> np.ndarray:
    """
    Evaluate reconstruction error of tensor decomposition of Cell-Cell Interaction (CCI) tensor matrix.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    num_TME_modules : int, optional
        Number of modules for the TME (Tumor Microenvironment) to test, by default 20.
    num_cellpair_modules : int, optional
        Number of modules for cell-cell pairs and ligand-receptor pairs to test, by default 20.
    device : str, optional
        Device for computation ('cpu' or 'cuda:X'), by default 'cuda:0'.
    init : str, optional
        Initialization method for tensor decomposition, by default 'svd'.
    n_iter_max : int, optional
        Maximum number of iterations for tensor decomposition, by default 10.
    method : str, optional
        Method for tensor decomposition ('tucker' or 'hals'), by default 'tucker'.
    use_gpu : bool, optional
        Whether to use GPU for computation, by default True.

    Returns
    -------
    np.ndarray
        A matrix representing evaluated reconstruction errors.
    """
    dat = interactiontensor.cci_matrix
    torch.cuda.empty_cache()
    # pal = sns.color_palette('bright',10)
    # palg = sns.color_palette('Greys',10)
    num_TME_modules = num_TME_modules + 1
    mat1 = np.zeros((num_TME_modules,num_cellpair_modules))
    # mat2 = np.zeros((num_TME_modules,num_cellpair_modules))
    if use_gpu:
        tensor = tl.tensor(dat, device=device)
    else:
        tensor = tl.tensor(dat)
        tensor = tensor.to("cpu")

    for i in tqdm(range(2,num_cellpair_modules)):
        for j in range(1,num_TME_modules):
            # we use NNTD as described in the paper

            if method == 'tucker':
                facs_overall = non_negative_tucker(tensor,rank=[i,i, j],random_state = 2337, init=init, n_iter_max=n_iter_max)
                # facs_overall = [factor.cpu() for factor in facs_overall]
                mat1[j,i] = np.mean((dat- tl.to_numpy(tl.tucker_to_tensor((facs_overall[0],facs_overall[1]))))**2)
                # mat2[j,i] = np.linalg.norm(dat - tl.to_numpy(tl.tucker_to_tensor((facs_overall[0],facs_overall[1]))) ) / np.linalg.norm(dat)
            elif method == 'hals':
                facs_overall = non_negative_tucker_hals(tensor,rank=[i,i, j],random_state = 2337, init=init, n_iter_max=n_iter_max)
                mat1[j,i] = np.mean((dat- tl.to_numpy(tl.tucker_to_tensor((facs_overall[0],facs_overall[1]))))**2)

    return mat1

def SCII_Tensor(interactiontensor: InteractionTensor=None, rank: list=[8,8,8], random_state: int=32, init: str="svd", n_iter_max: int=100, backend: str="pytorch",
                top_n_cc: int=3, top_n_lr: int=3, figsize=(8,15), device='cuda:0'):
    """
    Perform tensor factorization and analysis on a Cell-Cell Interaction (CCI) matrix.

    This function reads CCI matrix data, performs non-negative tensor factorization, generates visualizations,
    and saves the results for further analysis.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    rank : list, optional
        Rank for the non-negative tensor factorization, by default [8, 8, 8].
    random_state : int, optional
        Random state for reproducibility, by default 32.
    init : str, optional
        Initialization method for tensor factorization, by default 'svd'.
    n_iter_max : int, optional
        Maximum number of iterations for tensor factorization, by default 100.
    backend : str, optional
        Backend for tensor operations, by default 'pytorch'.
    top_n_cc : int, optional
        Number of top cell-cell pairs to consider for visualization, by default 3.
    top_n_lr : int, optional
        Number of top LR pairs to consider for visualization, by default 3.
    figsize : tuple, optional
        Size of the figure for heatmaps, by default (8,15).
    device : str, optional
        Device for computation ('cpu' or 'cuda:X'), by default 'cuda:0'.
    """

    time_start=time.time()

    torch.cuda.empty_cache()
    tl.set_backend(backend)

    tensor = tl.tensor(interactiontensor.cci_matrix, device=device)
    core, factors = non_negative_tucker(tensor, rank=rank, random_state = random_state, init=init, n_iter_max=n_iter_max)

    factors = [x.data.cpu().numpy() for x in factors]
    core = core.data.cpu().numpy()

    interactiontensor.factors = factors
    interactiontensor.core = core

#         with open("core.pkl", "wb") as f:
#             pickle.dump(core, f)

#         with open("factors.pkl", "wb") as f:
#             pickle.dump(factors, f)

    # core = pd.read_pickle("core.pkl")
    # factors = pd.read_pickle("factors.pkl")

    cellpair = interactiontensor.cellpair
    lrpair = interactiontensor.lrpair

    mpl.rcParams.update(mpl.rcParamsDefault)
    plot_top_heatmap(interactiontensor, pair='cc', top_n=top_n_cc, figsize = figsize, module_name='CellPair')
    plot_top_heatmap(interactiontensor, pair='lr', top_n=top_n_lr, figsize = figsize, module_name='LRPair')

    # plt.figure(figsize=(num_TME_modules*5,5))
    # for p in range(num_TME_modules):
    #     plt.subplot(1, num_TME_modules, p+1)
    #     sns.heatmap(pd.DataFrame(core[:,:,p]))
    #     plt.title('TME module {}'.format(p))
    #     plt.ylabel('CellPair module')
    #     plt.xlabel('LRPair module')
    # plt.savefig("core_heatmap.pdf")
    
    plot_core_heatmap(interactiontensor, num_TME_modules=rank[2], nrow=3, filename="core_heatmap.pdf")


    tme_cluster = find_max_column_indices(factors[2])
    interactiontensor.tme_cluster = tme_cluster

    # with open("tme_cluster.pkl", "wb") as f:
    #     pickle.dump(tme_cluster, f)

    window_zero_indices = interactiontensor.window_zero_indices

    indices = interactiontensor.indices

    indices_filter = [k for k,v in zip(indices, (~window_zero_indices).tolist()) if v]

    interactiontensor.indices_filter = indices_filter

#         with open("indices_filter.pkl", "wb") as f:
#             pickle.dump(indices_filter, f)

    adata = interactiontensor.adata
    adata.obs['TME_module'] = None
    for k,v in enumerate(tme_cluster):
        adata.obs['TME_module'].iloc[indices_filter[k]] = str(v)


    adata.obs.TME_module = adata.obs.TME_module.astype('category')
    adata.obs.TME_module = adata.obs.TME_module.cat.set_categories(np.arange(adata.obs.TME_module.cat.categories.astype('int').max()+1).astype('str'))
    interactiontensor.adata = adata
    # adata.write("adata_TME_module.h5ad", compression='gzip')

    sc.settings.set_figure_params(dpi=150, facecolor='white', fontsize=14)
    # sc.pl.spatial(adata_list[i], color = 'TME_module', img_key=None, spot_size=20)
    sc.pl.embedding(adata, color = 'TME_module', size=1, basis='spatial', save = "_TME_module.png")

    scv.set_figure_params('scvelo', dpi_save=300, fontsize=40, figsize=(8,8))
    # cluster_small_multiples(adata_list[i], 'TME_module', spot_size=1, save = f"{ids[i]}_TME_module_split.png")
    scv.pl.scatter(adata, groups=[[c] for c in adata.obs['TME_module'].cat.categories], color='TME_module', 
                   ncols=4, basis='spatial', save = "TME_module_split.png")

    lr_df = pd.DataFrame(factors[1], index = lrpair)
    interactiontensor.lr_factor = lr_df

    cc_df = pd.DataFrame(factors[0], index = cellpair)
    interactiontensor.cc_factor = cc_df

    top_pair(interactiontensor, pair='cc', top_n=15)
    top_pair(interactiontensor, pair='lr', top_n=50)

    # sc.settings.set_figure_params(dpi=150, facecolor='white', fontsize=6)
    # for col in lr_df.columns:
    #     plt.figure(figsize=(6,6), dpi=300)
    #     sns.heatmap(lr_df.loc[top_lrpair.iloc[0:25][col]])
    #     plt.xlabel("LRPair module")
    #     plt.savefig('lrpair' + str(col) + '_heatmap.pdf', bbox_inches='tight')

    time_end=time.time()
    logg.info(f"Finish SCII tensor - time cost {(((time_end - time_start) / 60) / 60)} h")

def plot_top_heatmap(interactiontensor: InteractionTensor=None, pair: str='cc', top_n: int=3, figsize: tuple=(8,15), module_name: str='CellPair'):
    """
    Plot a heatmap of the top loadings onto Cell-Cell Interaction (CCI) or Ligand-Receptor (LR) pairs.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    pair : str, optional
        Type of pair to visualize ('cc' for Cell-Cell Interaction, 'lr' for Ligand-Receptor pairs), by default 'cc'.
    top_n : int, optional
        Number of top loadings to display, by default 3.
    figsize : tuple, optional
        Size of the resulting heatmap figure, by default (8, 15).
    module_name : str, optional
        Name of the module (CellPair or LRPair) for labeling, by default 'CellPair'.
    """

    if pair == 'cc':
        index = interactiontensor.cellpair
        factor = interactiontensor.factors[0]
    elif pair == 'lr':
        index = interactiontensor.lrpair
        factor = interactiontensor.factors[1]

    top_df = pd.DataFrame(factor, index=index).apply(lambda x: _sorted(x, top_n))

    flat_list = list(chain.from_iterable(top_df.T.values.tolist()))

    data = pd.DataFrame(factor, index=index).loc[flat_list]
    g = sns.clustermap(data, row_cluster=False, col_cluster=False, figsize=figsize)
    g.ax_heatmap.set_title('Loadings onto '+ module_name + ' modules')
    g.ax_heatmap.set_xlabel(module_name +'module')
    g.ax_row_dendrogram.set_visible(False)
    g.ax_cbar.set_position((1, .2, .03, .4))
    g.ax_cbar.set_title("")
    # plt.tight_layout()
    g.savefig("top_" + module_name + "_heatmap.pdf", bbox_inches='tight')

def SCII_Tensor_multiple(interactiontensor: InteractionTensor=None, rank: list=[8,8,8], random_state: int=32, init: str="svd", n_iter_max: int=100, backend: str="pytorch",
                         top_n_cc: int=3, top_n_lr: int=3, figsize: tuple=(8,15), device: str='cuda:0'):
    """
    Perform tensor factorization and analysis on a Cell-Cell Interaction (CCI) matrix.

    This function reads CCI matrix data, performs non-negative tensor factorization, generates visualizations,
    and saves the results for further analysis.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    rank : list, optional
        Rank for the non-negative tensor factorization, by default [8, 8, 8].
    random_state : int, optional
        Random state for reproducibility, by default 32.
    init : str, optional
        Initialization method for tensor factorization, by default 'svd'.
    n_iter_max : int, optional
        Maximum number of iterations for tensor factorization, by default 100.
    backend : str, optional
        Backend for tensor operations, by default 'pytorch'.
    top_n_cc : int, optional
        Number of top cell-cell pairs to consider for visualization, by default 3.
    top_n_lr : int, optional
        Number of top LR pairs to consider for visualization, by default 3.
    figsize : tuple, optional
        Size of the figure for heatmaps, by default (8,15).
    device : str, optional
        Device for computation ('cpu' or 'cuda:X'), by default 'cuda:0'.
    """

    time_start=time.time()

    torch.cuda.empty_cache()
    tl.set_backend(backend)
    tensor = tl.tensor(interactiontensor.cci_matrix, device=device)

    core, factors = non_negative_tucker(tensor, rank=rank, random_state = random_state, init=init, n_iter_max=n_iter_max)

    factors = [x.data.cpu().numpy() for x in factors]
    core = core.data.cpu().numpy()

    interactiontensor.factors = factors
    interactiontensor.core = core

#         with open("core.pkl", "wb") as f:
#             pickle.dump(core.data.cpu().numpy(), f)

#         with open("factors.pkl", "wb") as f:
#             pickle.dump(factors_list, f)

    cellpair = interactiontensor.cellpair
    lrpair = interactiontensor.lrpair

    mpl.rcParams.update(mpl.rcParamsDefault)
    plot_top_heatmap(interactiontensor, top_n=top_n_cc, figsize = figsize, module_name='CellPair', pair='cc')
    plot_top_heatmap(interactiontensor, top_n=top_n_lr, figsize = figsize, module_name='LRPair', pair='lr')

    num_TME_modules=rank[2]

    plt.figure(figsize=(num_TME_modules*5,5))
    for p in range(num_TME_modules):
        plt.subplot(1, num_TME_modules, p+1)
        sns.heatmap(pd.DataFrame(core[:,:,p]))
        plt.title('TME module {}'.format(p))
        plt.ylabel('CellPair module')
        plt.xlabel('LRPair module')
    plt.savefig("core_heatmap.pdf")

    tme_cluster = find_max_column_indices(factors[2])
    interactiontensor.tme_cluster = tme_cluster
    # with open("tme_cluster.pkl", "wb") as f:
    #     pickle.dump(tme_cluster, f)

    # patient_id = [k for k, v in zip([], indice_list) for indice in v]
    patient_id = list(interactiontensor.module_lr_mt.keys())
    module_lr_mt = interactiontensor.module_lr_mt
    
    individual_patient = [[k]*v.shape[2] for k,v in zip(patient_id, module_lr_mt.values())]
    individual_patient = [item for sublist in individual_patient for item in sublist]
    interactiontensor.patient_map = individual_patient 

    individual_module = [np.arange(v.shape[2]) for v in module_lr_mt.values()]
    individual_module = [item for sublist in individual_module for item in sublist]
    interactiontensor.module_map = individual_module

    adata_list = interactiontensor.adata

    adata_list[0].obs['tmp'] = [str(random.randint(0, num_TME_modules-1)) for _ in range(adata_list[0].n_obs)]

    sc.pl.embedding(adata_list[0], color = 'tmp', size=1, basis='spatial', show=False)

    color_map = dict(zip([str(i) for i in np.arange(0,num_TME_modules).tolist()], adata_list[0].uns['tmp_colors']))    

    ids = patient_id
    for i in range(len(ids)):

        tme_cluster_idx = [v for k,v in zip(individual_patient, tme_cluster) if k == ids[i]]

        tme_module_idx = [v for k,v in zip(individual_patient, individual_module) if k == ids[i]]

        adata_list[i].obs['TME_module_module'] = None

        adata_list[i].obs['TME_module_module'] = adata_list[i].obs['TME_module'].map(dict(zip([str(x) for x in tme_module_idx], [str(x) for x in tme_cluster_idx])))

        adata_list[i].obs['TME_module_module'] =  adata_list[i].obs['TME_module_module'].astype('category')

        sc.settings.set_figure_params(dpi=150, facecolor='white', fontsize=14)
        # sc.pl.spatial(adata_list[i], color = 'TME_module', img_key=None, spot_size=20)
        sc.pl.embedding(adata_list[i], color = 'TME_module_module', size=1, basis='spatial', save = f"{ids[i]}_TME_module.png"
                        , palette = color_map, title="TME_module")

        scv.set_figure_params('scvelo', dpi_save=300, fontsize=40, figsize=(8,8))
        if len(adata_list[i].obs['TME_module_module'].cat.categories)>1:
            scv.pl.scatter(adata_list[i], groups=[[c] for c in adata_list[i].obs['TME_module_module'].cat.categories], color='TME_module_module', 
                           ncols=4, basis='spatial', save = f"{ids[i]}_TME_module_split.png"
                           , palette = color_map)
        else:
            scv.pl.scatter(adata_list[i], groups=[c for c in adata_list[i].obs['TME_module_module'].cat.categories], color='TME_module_module', basis='spatial', save = f"{ids[i]}_TME_module_split.png", palette = color_map)

    interactiontensor.adata_list_TME = adata_list

    # with open("adata_list_TME_module.pkl", "wb") as f:
    #     pickle.dump(adata_list, f)

    lr_df = pd.DataFrame(factors[1], index = lrpair)
    interactiontensor.lr_factor = lr_df

    cc_df = pd.DataFrame(factors[0], index = cellpair)
    interactiontensor.cc_factor = cc_df

    top_pair(interactiontensor, pair='cc', top_n=15)
    top_pair(interactiontensor, pair='lr', top_n=50)

    # sc.settings.set_figure_params(dpi=150, facecolor='white', fontsize=6)
    # for col in lr_df.columns:
    #     plt.figure(figsize=(6,6), dpi=300)
    #     sns.heatmap(lr_df.loc[top_lrpair.iloc[0:25][col]])
    #     plt.xlabel("LRPair module")
    #     plt.savefig('lrpair' + str(col) + '_heatmap.pdf', bbox_inches='tight')

    time_end=time.time()
    logg.info(f"Finish multiple sample SCII tensor - time cost {(((time_end - time_start) / 60) / 60)} h")

def top_pair(interactiontensor: InteractionTensor=None, pair='cc', top_n: int=15):
    """
    Identify the top CellPairs or LRpairs based on their loadings.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    pair : str, optional
        Type of pair to consider, either 'cc' for CellPairs or 'lr' for LRpairs, by default 'cc'.
    top_n : int, optional
        Number of top pairs to identify, by default 15.
    """

    factors = interactiontensor.factors
    if pair == 'cc':
        cc_df = interactiontensor.cc_factor
        top_ccpair = cc_df.apply(lambda x: _sorted(x, top_n))
        interactiontensor.top_ccpair=top_ccpair
    elif pair == 'lr':
        lr_df = interactiontensor.lr_factor
        top_lrpair = lr_df.apply(lambda x: _sorted(x, top_n))
        interactiontensor.top_lrpair= top_lrpair

def plot_tme_mean_intensity_multiple(interactiontensor: InteractionTensor=None, sample: str=None, 
                                     tme_module: int=0, cellpair_module: int=0, lrpair_module: int=0, 
                                     n_lr: int=15, n_cc: int=5, 
                                     figsize: tuple=(10,2), save: bool=True, vmax: int=None, **kwargs):
    """
    Plot the mean intensity heatmap for a specific TME module, CellPair module, LRPair module, and patient sample.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    sample : str, optional
        Patient sample identifier, by default None.
    tme_module : int, optional
        Index of the TME module to plot, by default 0.
    cellpair_module : int, optional
        Index of the CellPair module to plot, by default 0.
    lrpair_module : int, optional
        Index of the LRPair module to plot, by default 0.
    n_lr : int, optional
        Number of LRPairs to include in the plot, by default 15.
    n_cc : int, optional
        Number of CellPairs to include in the plot, by default 5.
    figsize : tuple, optional
        Figure size, by default (10, 2).
    save : bool, optional
        Whether to save the plot, by default True.
    **kwargs
        Additional keyword arguments to pass to the sns.heatmap function.
    """

    module_lr_mt = interactiontensor.module_lr_mt
    cellpair = interactiontensor.cellpair
    lrpair = interactiontensor.lrpair
    core = interactiontensor.core
    factors = interactiontensor.factors
    tme_cluster = interactiontensor.tme_cluster

    individual_patient = interactiontensor.patient_map
    individual_module = interactiontensor.module_map

    cc_df = interactiontensor.cc_factor
    lr_df = interactiontensor.lr_factor

    top_lrpair = interactiontensor.top_lrpair

    sub_cci_matrix = module_lr_mt[sample][:, :, [y for x,y,z in zip(individual_patient, individual_module, tme_cluster.tolist()) if x==sample and z == tme_module]]    
    mean_sub_mt = np.mean(sub_cci_matrix, axis=2)
    mean_df = pd.DataFrame(mean_sub_mt, index = cellpair, columns = lrpair)    
    mean_mt = mean_df[top_lrpair[lrpair_module][0:n_lr]].loc[_sorted(cc_df[cellpair_module],n_cc).tolist()]    

    if save:
        plt.figure(figsize=figsize)
        sns.heatmap(mean_mt, vmax=vmax, **kwargs)
        plt.title(f"Patient{sample}_TME{tme_module}_Cellpair{cellpair_module}_LRpair{lrpair_module}")
        plt.show()
        plt.savefig(f"Patient{sample}_TME{tme_module}_cellpair{cellpair_module}_LRpair{lrpair_module}_intensity.pdf", format="pdf", bbox_inches='tight')
    else:
        plt.figure(figsize=figsize)
        sns.heatmap(mean_mt, vmax=vmax, **kwargs)
        plt.title(f"Patient{sample}_TME{tme_module}_Cellpair{cellpair_module}_LRpair{lrpair_module}")

def plot_tme_mean_intensity(interactiontensor: InteractionTensor=None, 
                            tme_module: int=0, cellpair_module: int=0, lrpair_module: int=0, 
                            n_lr: int=15, n_cc: int=5, 
                            figsize: tuple=(10,2), save: bool=True, size: int=2, vmax: int=None, **kwargs):
    """
    Plot the mean intensity heatmap for a specific TME module, CellPair module, and LRPair module.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    tme_module : int, optional
        Index of the TME module to plot, by default 0.
    cellpair_module : int, optional
        Index of the CellPair module to plot, by default 0.
    lrpair_module : int, optional
        Index of the LRPair module to plot, by default 0.
    n_lr : int, optional
        Number of LRPairs to include in the plot, by default 15.
    n_cc : int, optional
        Number of CellPairs to include in the plot, by default 5.
    figsize : tuple, optional
        Figure size, by default (10, 2).
    save : bool, optional
        Whether to save the plot, by default True.
    size : int, optional
        Font scale for the plot, by default 2.
    **kwargs
        Additional keyword arguments to pass to the sns.heatmap function.
    """

    lr_mt_list = interactiontensor.lr_mt_list
    cellpair = interactiontensor.cellpair
    lrpair = interactiontensor.lrpair
    core = interactiontensor.core
    factors = interactiontensor.factors
    tme_cluster = interactiontensor.tme_cluster

    cc_df = interactiontensor.cc_factor
    lr_df = interactiontensor.lr_factor

    top_lrpair = interactiontensor.top_lrpair
    top_ccpair = interactiontensor.top_ccpair

    sub_cci_matrix = [k for k,v in zip(lr_mt_list, pd.Series(tme_cluster).isin([tme_module])) if v]

    sub_mt = np.dstack(sub_cci_matrix)

    mean_sub_mt = np.mean(sub_mt, axis=2)

    mean_df = pd.DataFrame(mean_sub_mt, index = cellpair, columns = lrpair)

    mean_mt = mean_df[top_lrpair[lrpair_module][0:n_lr]].loc[_sorted(cc_df[cellpair_module],n_cc).tolist()]

    if save:
        plt.figure(figsize=figsize)
        sns.set(font_scale=size)
        sns.heatmap(mean_mt, vmax=vmax, **kwargs)
        plt.title(f"TME{tme_module}_Cellpair{cellpair_module}_LRpair{lrpair_module}")
        plt.savefig(f"TME{tme_module}_cellpair{cellpair_module}_LRpair{lrpair_module}_intensity.pdf", format="pdf", bbox_inches='tight')
    else:
        plt.figure(figsize=figsize)
        sns.set(font_scale=size)
        sns.heatmap(mean_mt, vmax=vmax, **kwargs)
        plt.title(f"TME{tme_module}_Cellpair{cellpair_module}_LRpair{lrpair_module}")

def plot_core_heatmap(interactiontensor: InteractionTensor=None, num_TME_modules: int=None, nrow: int=3, filename: str="core_heatmap.pdf"):
    """
    Plot the core matrix.

    Parameters
    ----------
    interactiontensor : InteractionTensor
        The InteractionTensor of the spatial transcriptomics data.
    num_TME_modules : int
        numer of TME modules.
    nrow : int
        number of row to plot.
    filename : str
        filename to save.
    """
    core = interactiontensor.core
    
    subplots_per_row = num_TME_modules // nrow
    remainder_subplots = num_TME_modules % nrow
    plt.figure(figsize=(subplots_per_row*5, nrow*5))
    for p in range(num_TME_modules):
        plt.subplot(num_rows, subplots_per_row + remainder_subplots, p+1)
        sns.heatmap(pd.DataFrame(core[:, :, p]))
        plt.title('TME module {}'.format(p))
        plt.ylabel('CellPair module')
        plt.xlabel('LRPair module')
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig(filename)

def merge_data(interactiontensor_list: list=None, patient_id: list=None) -> InteractionTensor:
    """
    Merge data from multiple InteractionTensor.

    Parameters
    ----------
    interactiontensor_list : list, optional
        List of InteractionTensor instances to merge, by default None.
    patient_id : list
        List of patient id

    Returns
    -------
    InteractionTensor
        A new InteractionTensor instance containing merged data.
    """
    lr_mt_list = []
    adata_list = []
    zero_indices_list = []
    tme_cluster_list = []
    indice_list = []
    for interactiontensor in interactiontensor_list:
        lr_mt_list.append(interactiontensor.lr_mt_list)
        adata_list.append(interactiontensor.adata)
        zero_indices_list.append(interactiontensor.window_zero_indices)
        tme_cluster_list.append(interactiontensor.tme_cluster)
        indice_list.append(interactiontensor.indices_filter)

    tmp = InteractionTensor(adata_list)

    common_columns = set(lr_mt_list[0][0].columns)
    for i in range(len(lr_mt_list)):
        common_columns = common_columns.intersection(set(lr_mt_list[i][0].columns))

    tmp.lrpair = list(common_columns)

    # with open("lrpair.pkl", "wb") as f:
    #     pickle.dump(list(common_columns), f)

    for i in range(len(lr_mt_list)):
        for j in range(len(lr_mt_list[i])):
            lr_mt_list[i][j] = lr_mt_list[i][j][list(common_columns)].loc[lr_mt_list[0][0].index]

    cellpair = lr_mt_list[0][0].index
    tmp.cellpair = cellpair
    # with open("cellpair.pkl", "wb") as f:
    #     pickle.dump(cellpair, f)

    cci_matrix = []
    for i in range(len(lr_mt_list)):
        cci_matrix.append(np.dstack(lr_mt_list[i]))

    for i in range(len(cci_matrix)):
        cci_matrix[i] = np.log1p(cci_matrix[i])

    for i in range(len(cci_matrix)):
        cci_matrix[i] = cci_matrix[i][:, :, ~zero_indices_list[i]]

    tmp.cci_matrix_individual = cci_matrix

    # with open("lr_mt_list.pkl", "wb") as f:
    #     pickle.dump(cci_matrix, f)

    # with open("tme_cluster_list.pkl", "wb") as f:
    #     pickle.dump(tme_cluster_list, f)

    indice = [item for indice in indice_list for item in indice]

    # with open("indices.pkl", "wb") as f:
    #     pickle.dump(indice, f)

    tmp.indices_filter = indice

    module_lr_mt = {}
    for i in range(len(cci_matrix)):
        result = np.zeros((cci_matrix[0].shape[0], cci_matrix[0].shape[1], np.unique(tme_cluster_list[i]).max()+1))

        for cluster in range(np.unique(tme_cluster_list[i]).max()+1):
            cluster_mask = (tme_cluster_list[i] == cluster)  # Create a boolean mask for the current cluster
            cluster_data = cci_matrix[i][:, :, cluster_mask]  # Extract data for the current cluster
            result[:, :, cluster] = np.mean(cluster_data, axis=2)  # Calculate the mean along the third dimension
            module_lr_mt[patient_id[i]] = result
    tmp.module_lr_mt = module_lr_mt

    # with open("module_lr_mt.pkl", "wb") as f:
    #     pickle.dump(module_lr_mt, f)

    final_mt = np.dstack(list(module_lr_mt.values()))

    tmp.cci_matrix = final_mt
    # with open("final_mt.pkl", "wb") as f:
    #     pickle.dump(final_mt, f)

    return tmp