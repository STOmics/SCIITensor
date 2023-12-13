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
import tensorly as tl
from tensorly.decomposition import non_negative_tucker, non_negative_tucker_hals
# from tensorly.contrib.sparse.decomposition import non_negative_tucker
import logging
from itertools import chain
import torch

def _generate_LRpairs(interactionDB:str,
                      adata):
    """
    process L-R database.
    """
    file_name = os.path.basename(interactionDB)
    SOURCE = 'source'
    TARGET = 'target'
    # if not "MOUSE" in file_name.upper():
    #     #translate mouse gene symbol to human gene symbol
    #     adata = _mh_translate(adata)
    #     print(adata)
    #     logging.info(f"gene homology translation finished.")

    if "CELLCHATDB" in file_name.upper():
        interactions = pd.read_csv(interactionDB)
        interactions[SOURCE] = interactions['ligand']
        interactions[TARGET] = interactions['interaction_name_2'].str.extract('- (.+)', expand=False).map(lambda x: x.strip().strip("(").strip(")").replace("+", "_"))
    else:
        interactions = pd.read_csv(interactionDB)      
        if SOURCE in interactions.columns:
            interactions.pop(SOURCE)
        if TARGET in interactions.columns:
            interactions.pop(TARGET)
        interactions.rename(
                    columns={"genesymbol_intercell_source": SOURCE, "genesymbol_intercell_target": TARGET}, inplace=True
                )
        interactions[SOURCE] = interactions[SOURCE].str.replace("^COMPLEX:", "", regex=True)
        interactions[TARGET] = interactions[TARGET].str.replace("^COMPLEX:", "", regex=True)
    LRpairs = interactions[[SOURCE, TARGET]].drop_duplicates().values
    LRlist = []
    for gene in LRpairs.flatten():
        LRlist.extend(gene.split("_"))
    adata = adata[:, adata.var_names.isin(LRlist)]
    filter_LRpairs = []
    for LRpair in LRpairs:
        ligand, receptor = LRpair
        if all(l in adata.var_names for l in ligand.split("_")) and all(r in adata.var_names for r in receptor.split("_")):
            filter_LRpairs.append(LRpair)
    return filter_LRpairs, adata


def _get_LR_connect_matrix(adata:'anndata', 
                           LRpair:list, 
                           connect_matrix:'sparse.csr_matrix',
                           complex_process:str = 'mean',                      
                           ) -> 'sparse.coo_matrix':
    ligand, receptor = LRpair    
    if "_" in ligand:
        ligands = ligand.split("_")
        if complex_process.upper() == 'MEAN':
            exp_l = adata[:, ligands].X.mean(axis=1).A1
        elif complex_process.upper() == 'MIN':       
            exp_l = adata[:, ligands].X.min(axis=1).toarray()[:,0]
        else:
            raise Exception("complex process model must be mean or min, but got: {0}".format(complex_process))         
    else:
        exp_l = adata[:, ligand].X.toarray()[:,0]
    if "_" in receptor:
        receptors = receptor.split("_")
        if complex_process.upper() == 'MEAN':
            exp_r = adata[:, receptors].X.mean(axis=1).A1
        elif complex_process.upper() == 'MIN':
            exp_r = adata[:, receptors].X.min(axis=1).toarray()[:,0]
        else:
            raise Exception("complex process model must be mean or min, but got: {0}".format(complex_process))
    else:
        exp_r = adata[:, receptor].X.toarray()[:,0]
    l_rows = np.where(exp_l > 0)[0]
    r_cols = np.where(exp_r > 0)[0]
    #dst = np.where(connect_matrix[l_rows,:][:,r_cols].todense() > 0)
    row_indices, col_indices = connect_matrix[l_rows,:][:,r_cols].nonzero()

    connect_exp_lr = exp_l[l_rows[row_indices]] + exp_r[r_cols[col_indices]]
    exp_connect_matrix = sparse.coo_matrix((connect_exp_lr, (l_rows[row_indices], r_cols[col_indices])), shape=connect_matrix.shape)
    return exp_connect_matrix


def _get_LR_intensity(exp_connect_matrix:'sparse.coo_matrix', 
                      cellTypeIndex:np.array, 
                      cellTypeNumber:int
                      ) -> np.matrix:
    senders = cellTypeIndex[exp_connect_matrix.row]
    receivers = cellTypeIndex[exp_connect_matrix.col]
    interaction_matrix = sparse.csr_matrix((exp_connect_matrix.data, (senders, receivers)), shape=(cellTypeNumber, cellTypeNumber))
    
    return interaction_matrix.todense()


def _permutation_test(interaction_matrix:np.matrix,
                      exp_connect_matrix:'sparse.coo_matrix',
                      cellTypeIndex:np.array,
                      cellTypeNumber:int,
                      n_perms = 1000,
                      seed = 101
                      ) -> np.array:
    pvalues = np.zeros((cellTypeNumber, cellTypeNumber), dtype=np.int32)
    for i in range(n_perms):
        cellTypeIndex_tmp = cellTypeIndex.copy()
        rs = np.random.RandomState(None if seed is None else i + seed)
        rs.shuffle(cellTypeIndex_tmp)
        interaction_matrix_tmp = _get_LR_intensity(exp_connect_matrix, cellTypeIndex_tmp, cellTypeNumber)
        pvalues += np.where(interaction_matrix_tmp > interaction_matrix, 1, 0)
    pvalues = pvalues/float(n_perms)
    pvalues[interaction_matrix == 0] = None
    return pvalues

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

def _result_combined(result_list:list, #list<np.array>
                     LRpairs:np.array,
                     cell_types:np.array
                     ) -> pd.DataFrame:
    columns = []
    for sender in cell_types:
        for receiver in cell_types:
            columns.append([sender, receiver])
    my_columns = pd.MultiIndex.from_tuples(columns, names=['cluster1', 'cluster2'])
    my_index = pd.MultiIndex.from_tuples(LRpairs, names=["source", "target"])
    values = [np.ravel(x) for x in result_list]
    result_df = pd.DataFrame(np.row_stack(values), index = my_index, columns = my_columns)
    return result_df


def _mh_translate(adata:anndata):
    if any([x.startswith("Gm") for x in adata.var.index]):
        adata = _m2h_homologene(adata)
        #self.adata
    return adata
    
def _m2h_homologene(adata: anndata):
    """
    homologous transition gene name from mouse to human

    Parameters
    ----------
    adata
        anndata
    """
    # Biomart tables
    biomart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets/biomart")
    #h2m_tab = pd.read_csv(os.path.join(biomart_path, "human_to_mouse_biomart_export.csv")).set_index('Gene name')
    m2h_tab = pd.read_csv(os.path.join(biomart_path, "mouse_to_human_biomart_export.csv")).set_index('Gene name')

    hdict = m2h_tab[~m2h_tab['Human gene name'].isnull()]['Human gene name'].to_dict()
    adata.var['original_gene_symbol'] = adata.var_names
    adata.var.index = [hdict[x] if x in hdict.keys() else x.upper() for x in adata.var_names]
    adata.var_names_make_unique()
    return adata

def _sorted(s, num):
        tmp = s.sort_values(ascending=False)[:num].index  # earlier s.order(..)
        tmp.index = range(num)
        return tmp

def find_max_column_indices(array):
    max_column_indices = np.argmax(array, axis=1)
    return max_column_indices




# def evaluate_ranks(dat, num_TME_modules = 6, num_cellpair_modules = 15, device = "cuda:0", init='svd', n_iter_max=10, method = 'tucker', use_gpu = True):
#     torch.cuda.empty_cache()
#     # pal = sns.color_palette('bright',10)
#     # palg = sns.color_palette('Greys',10)
#     num_TME_modules = num_TME_modules + 1
#     mat1 = np.zeros((num_TME_modules,num_cellpair_modules))
#     # mat2 = np.zeros((num_TME_modules,num_cellpair_modules))
#     if use_gpu:
#         tensor = tl.tensor(dat, device=device)
#     else:
#         tensor = tl.tensor(dat)
#         tensor = tensor.to("cpu")
                
#     for i in range(2,num_cellpair_modules):
#         for j in range(1,num_TME_modules):
#             # we use NNTD as described in the paper
#             print(i)
#             if method == 'tucker':
#                 facs_overall = non_negative_tucker(tensor,rank=[i,i, j],random_state = 2337, init=init, n_iter_max=n_iter_max)
#                 # facs_overall = [factor.cpu() for factor in facs_overall]
#                 mat1[j,i] = np.mean((dat- tl.to_numpy(tl.tucker_to_tensor((facs_overall[0],facs_overall[1]))))**2)
#                 # mat2[j,i] = np.linalg.norm(dat - tl.to_numpy(tl.tucker_to_tensor((facs_overall[0],facs_overall[1]))) ) / np.linalg.norm(dat)
#             elif method == 'hals':
#                 facs_overall = non_negative_tucker_hals(tensor,rank=[i,i, j],random_state = 2337, init=init, n_iter_max=n_iter_max)
#                 mat1[j,i] = np.mean((dat- tl.to_numpy(tl.tucker_to_tensor((facs_overall[0],facs_overall[1]))))**2)
#     return mat1


# def plot_cellpair_module_heatmap(factor, figsize = (5,10), index = None, filename = "cellpair_module_heatmap.pdf", vmax=None):
#     # plt.figure(figsize=figsize)
#     g = sns.clustermap(pd.DataFrame(factor,index = index), row_cluster=True, col_cluster=False, figsize = figsize, vmax=vmax)
#     g.ax_heatmap.set_title('Loadings onto CellPair modules')
#     g.ax_heatmap.set_xlabel('CellPair module')
#     g.ax_row_dendrogram.set_visible(False)
#     g.ax_cbar.set_position((1, .2, .03, .4))
#     g.ax_cbar.set_title("")
#     # plt.tight_layout()
#     g.savefig(filename, bbox_inches='tight')
    # plt.savefig("cellpair_module_heatmap.pdf", bbox_inches='tight')
    # g.ax_heatmap.set_ylabel('CellPair',loc='left')
    # plt.ylabel('CellPair')
    # plt.xlabel('CellPair module')
    # plt.title('Loadings onto CellPair modules')

# def plot_lrpair_module_heatmap(factor, figsize = (5,10), index = None, filename = 'LRpair_module_heatmap.pdf', vmax=None):
#     # plt.figure(figsize=figsize)
#     g = sns.clustermap(pd.DataFrame(factor,index = index), row_cluster=True, col_cluster=False, figsize = figsize, vmax=vmax)
#     g.ax_heatmap.set_title('Loadings onto LRPair modules')
#     g.ax_heatmap.set_xlabel('LRPair module')
#     g.ax_row_dendrogram.set_visible(False)
#     g.ax_cbar.set_position((1, .2, .03, .4))
#     g.savefig(filename, bbox_inches='tight')
    # plt.ylabel('LRPair')
    # plt.xlabel('LRPair module')
    # plt.title('Loadings onto LRPair modules')

# def plot_TME_module_heatmap(factor, figsize = (5,10), filename = 'TME_module_heatmap.pdf', vmax=None):
#     # plt.figure(figsize=figsize)
#     g = sns.clustermap(pd.DataFrame(factor), row_cluster=True, col_cluster=False, figsize = figsize, vmax=vmax)
#     g.ax_heatmap.set_title('Loadings onto TME modules')
#     g.ax_heatmap.set_xlabel('TME module')
#     g.ax_heatmap.set_yticklabels("")
#     # g.ax_heatmap.set_ylabel('TME', loc='top')
#     g.ax_row_dendrogram.set_visible(False)
#     g.ax_cbar.set_position((1, .2, .03, .4))
#     g.savefig(filename, bbox_inches='tight')

    # plt.ylabel('TME')
    # plt.xlabel('TME module')
    # plt.title('Loadings onto TME modules')

# def plot_top_heatmap(factor, index, top_n, figsize=(8,15), module_name = 'CellPair'):
#     top_cellpair = pd.DataFrame(factor, index=index).apply(lambda x: _sorted(x, top_n))

#     flat_list = list(chain.from_iterable(top_cellpair.T.values.tolist()))

#     data = pd.DataFrame(factor, index=index).loc[flat_list]
#     g = sns.clustermap(data, row_cluster=False, col_cluster=False, figsize=figsize)
#     g.ax_heatmap.set_title('Loadings onto '+ module_name + ' modules')
#     g.ax_heatmap.set_xlabel(module_name +'module')
#     g.ax_row_dendrogram.set_visible(False)
#     g.ax_cbar.set_position((1, .2, .03, .4))
#     g.ax_cbar.set_title("")
#     # plt.tight_layout()
#     g.savefig("top_" + module_name + "_heatmap.pdf", bbox_inches='tight')

# def save_pkl(data, filename):
#     if not os.path.exists(filename):
#         # If the file doesn't exist, save the data to it
#         with open(filename, 'wb') as file:
#             pickle.dump(data, file)
#         print(f"File '{filename}' saved.")
#     else:
#         print(f"File '{filename}' already exists. Skipping the save operation.")
