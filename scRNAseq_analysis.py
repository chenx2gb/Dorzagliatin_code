#!/usr/bin/env python3
"""
Single-cell RNA-seq Analysis Pipeline
Dorzagliatin ob/ob Mouse Study
"""

import scanpy as sc
import scvi
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.sparse import issparse
import warnings
warnings.filterwarnings('ignore')

# Configuration
RAW_DATA_DIR = Path("../rawdata/obobscRNAdata")
OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {"Model": "25100538_modle", "Dorz": "25100538_ob_Dorz"}
GROUP_MAPPING = {'Model': 'ob/ob', 'Dorz': 'Dorzagliatin'}

QC_MIN_UMI = 500
QC_MIN_GENES = 500
QC_MAX_MITO_PCT = 10
N_TOP_GENES = 2000

SCVI_MAX_EPOCHS = 400
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.4

MARKERS = {
    "Beta": ["Ins1", "Ins2", "Mafa"],
    "Alpha": ["Gcg", "Arx", "Irx2"],
    "Delta": ["Sst"],
    "PP": ["Ppy", "Pyy"],
    "Ductal": ["Krt18", "Dsg2"],
    "Endothelial": ["Cdh5", "Kdr", "Pecam1"],
    "Fibroblast": ["Col1a2", "Col3a1", "Col6a3"],
    "Macrophage": ["Lyz2", "Adgre1", "Apoe"],
    "Pericyte": ["Rgs5", "Abcc9", "Pdgfrb"],
    "T_cell": ["Lck", "Cd28"],
    "B_cell": ["Cd79a", "Cd22"],
}

GENE_MODULES = {
    'GLP1R_Pathway': ['Glp1r', 'Gnas', 'Adcy5', 'Adcy6', 'Prkaca', 'Prkacb', 'Creb1', 'Pdx1', 'Nkx6-1'],
    'Glucose_Sensing': ['Gck', 'Slc2a2', 'Pfkl', 'Pfkm', 'Aldoa', 'Gapdh', 'Pkm'],
    'Proliferation': ['Mki67', 'Pcna', 'Ccnd1', 'Ccnd2'],
    'CREB_Activity': ['Prkaca', 'Prkacb', 'Creb1', 'Fos', 'Jun', 'Egr1', 'Nr4a1', 'Bdnf', 'Crtc2'],
    'MAPK_Pathway': ['Map2k1', 'Map2k2', 'Mapk1', 'Mapk3']
}

# Setup
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=150, dpi_save=300, facecolor='white')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42

print("\n" + "="*80)
print("Single-cell RNA-seq Analysis Pipeline")
print("="*80)

# Load data
print("\nLoading data...")
adatas = []
for sample_name, folder_name in SAMPLES.items():
    data_path = RAW_DATA_DIR / folder_name / "filtered_feature_bc_matrix"
    nested_path = data_path / "filtered_feature_bc_matrix"
    if nested_path.exists():
        data_path = nested_path
    adata = sc.read_10x_mtx(data_path, var_names='gene_symbols', cache=False)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obs['batch'] = sample_name
    adatas.append(adata)
    print(f"  {sample_name}: {adata.n_obs} cells")

adata = ad.concat(adatas, join="outer", label="sample", keys=list(SAMPLES.keys()))
adata.obs_names_make_unique()
adata.var_names_make_unique()

# QC
print("\nQuality control...")
adata.var['mt'] = adata.var_names.str.startswith('mt-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

n_before = adata.n_obs
adata = adata[adata.obs['total_counts'] > QC_MIN_UMI, :].copy()
adata = adata[adata.obs['n_genes_by_counts'] > QC_MIN_GENES, :].copy()
adata = adata[adata.obs['pct_counts_mt'] < QC_MAX_MITO_PCT, :].copy()
sc.pp.filter_genes(adata, min_cells=3)
print(f"  Cells: {n_before} → {adata.n_obs}")

# Normalization
adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_GENES, flavor='seurat_v3', batch_key='batch')
adata.raw = adata

# scVI integration
print("\nscVI integration...")
adata_scvi = adata.copy()
adata_scvi.X = adata_scvi.layers['counts'].copy()
adata_scvi = adata_scvi[:, adata_scvi.var['highly_variable']].copy()

scvi.settings.seed = 42
scvi.model.SCVI.setup_anndata(adata_scvi, layer=None, batch_key='batch')
model = scvi.model.SCVI(adata_scvi, n_layers=2, n_latent=30, gene_likelihood='nb')
model.train(max_epochs=SCVI_MAX_EPOCHS)

latent = model.get_latent_representation()
adata.obsm['X_scVI'] = latent

# Clustering
sc.pp.neighbors(adata, use_rep='X_scVI', n_neighbors=UMAP_N_NEIGHBORS)
sc.tl.leiden(adata, resolution=1.0)
sc.tl.umap(adata, min_dist=UMAP_MIN_DIST)

# Cell type annotation
print("\nCell type annotation...")
for ctype, markers in MARKERS.items():
    valid_markers = [m for m in markers if m in adata.var_names]
    if valid_markers:
        sc.tl.score_genes(adata, valid_markers, score_name=f'score_{ctype}')

score_cols = [f'score_{ct}' for ct in MARKERS.keys() if f'score_{ct}' in adata.obs.columns]
cluster_scores = adata.obs.groupby('leiden')[score_cols].mean()
cluster_map = {cid: cluster_scores.loc[cid].idxmax().replace('score_', '')
               for cid in cluster_scores.index}
adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_map).astype('category')

# DEG analysis
print("\nDifferential expression analysis...")
adata.obs['group'] = adata.obs['batch'].map(GROUP_MAPPING)
beta_mask = adata.obs['cell_type'].astype(str).str.contains('Beta', case=False)
adata_beta = adata[beta_mask].copy()

obob = adata_beta[adata_beta.obs['group'] == 'ob/ob']
dorz = adata_beta[adata_beta.obs['group'] == 'Dorzagliatin']

obob_mean = obob.X.mean(axis=0).A1 if issparse(obob.X) else obob.X.mean(axis=0)
dorz_mean = dorz.X.mean(axis=0).A1 if issparse(dorz.X) else dorz.X.mean(axis=0)
log2fc = np.log2((dorz_mean + 1e-9) / (obob_mean + 1e-9))

pvals = []
for i in range(adata_beta.n_vars):
    x = obob.X[:, i].toarray().flatten() if issparse(obob.X) else obob.X[:, i]
    y = dorz.X[:, i].toarray().flatten() if issparse(dorz.X) else dorz.X[:, i]
    _, p = mannwhitneyu(y, x, alternative='two-sided')
    pvals.append(p)

_, padj, _, _ = multipletests(np.array(pvals), method='fdr_bh')

deg_df = pd.DataFrame({
    'gene': adata_beta.var_names,
    'mean_obob': obob_mean,
    'mean_dorz': dorz_mean,
    'log2FC': log2fc,
    'pval': pvals,
    'padj': padj
}).sort_values('log2FC', ascending=False)

deg_df.to_csv(OUTPUT_DIR / 'DEG_all.csv', index=False)
sig = deg_df[(deg_df['padj'] < 0.05) & (np.abs(deg_df['log2FC']) > 0.25)]
sig.to_csv(OUTPUT_DIR / 'DEG_significant.csv', index=False)
sig[sig['log2FC'] > 0]['gene'].to_csv(OUTPUT_DIR / 'DEG_up.txt', index=False, header=False)
sig[sig['log2FC'] < 0]['gene'].to_csv(OUTPUT_DIR / 'DEG_down.txt', index=False, header=False)
print(f"  DEGs: {len(sig[sig['log2FC']>0])} up, {len(sig[sig['log2FC']<0])} down")

# Functional scoring
print("\nFunctional module scoring...")
for module_name, genes in GENE_MODULES.items():
    available = [g for g in genes if g in adata_beta.var_names]
    if available:
        sc.tl.score_genes(adata_beta, available, score_name=f'{module_name}_score')
        print(f"  {module_name}: {len(available)}/{len(genes)} genes")

# Save
print("\nSaving results...")
adata.write_h5ad(OUTPUT_DIR / 'adata_final.h5ad')
adata_beta.write_h5ad(OUTPUT_DIR / 'adata_beta_final.h5ad')

print("\nCompleted!")
print(f"Results in: {OUTPUT_DIR.absolute()}")
print("="*80 + "\n")
