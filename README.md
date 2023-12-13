# SCIITensor
SCIITensor is a Python tool designed to decipher tumor microenvironment by deconvoluting spatial cellular interaction intensity.

# Installation

```bash
git clone https://github.com/STOmics/SCIITensor.git

cd SCIITensor

python setup.py install
```

# Tutorial

## Single sample analysis
```python
import SCIITensor as sct
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

adata = sc.read("/data/work/LR_TME/Liver/LC5M/sp.h5ad")
lc5m = sct.core.scii_tensor.InteractionTensor(adata, interactionDB="/data/work/database/LR/cellphoneDB_interactions_add_SAA1.csv")
sct.core.scii_tensor.build_SCII(lc5m)
sct.core.scii_tensor.process_SCII(lc5m, bin_zero_remove=True, log_data=True)
sct.core.scii_tensor.eval_SCII_rank(lc5m)
sct.core.scii_tensor.SCII_Tensor(lc5m)
with open("LC5M_res.pkl", "wb") as f:
    pickle.dump(lc5m, f)


# Visualization
## heatmap
sct.core.scii_tensor.plot_tme_mean_intensity(lc5m, tme_module = 0, cellpair_module = 2, lrpair_module = 4,
    n_lr = 15, n_cc = 5,
    figsize = (10, 2), save = False, size = 2, vmax=1)
factor_cc = lc5m.cc_factor.copy()
factor_cc.columns = factor_cc.columns.map(lambda x: f"CC_Module {x}")

factor_lr = lc5m.lr_factor.copy()
factor_lr.columns = factor_lr.columns.map(lambda x: f"LR_Module {x}")

factor_tme = pd.DataFrame(lc5m.factors[2])
factor_tme.columns = factor_tme.columns.map(lambda x: f"TME {x}")

#draw the heatmap based on the cell-cell factor matrix
fig = sns.clustermap(factor_cc.T, cmap="Purples", standard_scale=0, metric='euclidean', method='ward', 
                     row_cluster=False, dendrogram_ratio=0.05, cbar_pos=(1.02, 0.6, 0.01, 0.3),
                     figsize=(24, 10),
                     )
fig.savefig("./factor_cc_heatmap.pdf")

#select the top ligand-receptor pairs, then draw the heatmap based on ligan-receptor factor matrix
lr_number = 120 #number of ligand-receptor pairs on the top that will remain
factor_lr_top = factor_lr.loc[factor_lr.sum(axis=1).sort_values(ascending=False).index[0:lr_number]]
fig = sns.clustermap(factor_lr_top.T, cmap="Purples", standard_scale=0, metric='euclidean', method='ward', 
                     row_cluster=False, dendrogram_ratio=0.05, cbar_pos=(1.02, 0.6, 0.01, 0.3),
                     figsize=(28, 10),
                     )
fig.savefig("./factor_lr_heatmap.pdf")

## sankey
core_df = sct.plot.sankey.core_process(lc5m.core)
sct.plot.sankey.sankey_3d(core_df, link_alpha=0.5, interval=0.001, save="sankey_3d.pdf")

## circles
interaction_matrix = sct.plot.scii_circos.interaction_select(lc5m.lr_mt_list, factor_cc, factor_lr, factor_tme, 
                               interest_TME='TME 0',
                               interest_cc_module='CC_Module 3',
                               interest_LR_module='LR_Module 4',
                               lr_number=20,
                               cc_number=10)

plt.figure(figsize=(8, 3))
sns.heatmap(interaction_matrix, vmax=1)

#Draw the circos diagram, which includes cell types, ligand-receptor genes, and the links between ligands and receptors.
cells = ['Hepatocyte', 'Fibroblast', 'Cholangiocyte', 'Endothelial', 'Macrophage', 'Malignant', 'B_cell', 'T_cell', 'DC', 'NK'] #list contains names of all cell types
sct.plot.scii_circos.cells_lr_circos(interaction_matrix, cells, save="cells_lr_circos.pdf")

#Draw the circos which only contains cell types and the links between them.
sct.plot.scii_circos.cells_circos(interaction_matrix, cells, save="cells_circos.pdf")

#Draw circos which only contains ligand-receptor genes
sct.plot.scii_circos.lr_circos(interaction_matrix, cells)

## igraph
sct.plot.scii_net.grap_plot(interaction_matrix, cells,
                   save="igrap_network.pdf")

cc_df = sankey.factor_process(lc5m.factors[0], lc5m.cellpair)
sct.plot.sankey.sankey_2d(cc_df)
```

## Multiple sample analysis
```python
adata_LC5P = sc.read("/data/work/LR_TME/Liver/LC5P/FE1/cell2location_map/sp.h5ad")
lc5p = sct.core.scii_tensor.InteractionTensor(adata_LC5P, interactionDB="/data/work/database/LR/cellphoneDB_interactions_add_SAA1.csv")
sct.core.scii_tensor.build_SCII(lc5p)
sct.core.scii_tensor.process_SCII(lc5p)
sct.core.scii_tensor.eval_SCII_rank(lc5p)
sct.core.scii_tensor.SCII_Tensor(lc5p)
with open('LC5P_res.pkl', "wb") as f:
    pickle.dump(lc5p, f)


adata_LC5T = sc.read("/data/work/LR_TME/Liver/LC5T/FD3/cell2location_map/sp.h5ad")
lc5t = sct.core.scii_tensor.InteractionTensor(adata_LC5T, interactionDB="/data/work/database/LR/cellphoneDB_interactions_add_SAA1.csv")
sct.core.scii_tensor.build_SCII(lc5t)
sct.core.scii_tensor.process_SCII(lc5t)
sct.core.scii_tensor.eval_SCII_rank(lc5t)
sct.core.scii_tensor.SCII_Tensor(lc5t)
with open('LC5T_res.pkl', "wb") as f:
    pickle.dump(lc5t, f)

## merge data
all_data = sct.core.scii_tensor.merge_data([lc5t, lc5m, lc5p], patient_id = ['LC5T', 'LC5M', 'LC5P'])
sct.core.scii_tensor.SCII_Tensor_multiple(all_data)

## heatmap
mpl.rcParams.update(mpl.rcParamsDefault)
sct.core.scii_tensor.plot_tme_mean_intensity_multiple(all_data, sample='LC5T',
                                                     tme_module=0, cellpair_module=0, lrpair_module=0, vmax=1)
```
