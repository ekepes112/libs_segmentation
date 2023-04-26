from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
try:
    from src.MapData import MapData
    from src.LineFinder import LineFinder
    from src.plotting_functions import plot_embedding, plot_single_variable_map
except:
    from MapData import MapData
    from LineFinder import LineFinder
    from plotting_functions import plot_embedding, plot_single_variable_map

from joblib import dump
from time import time

file_name = '2022_03_22_P56B_307x532'

map_data = MapData(f'./data/{file_name}.libsdata')
map_data.get_metadata()
map_data.load_wavelenths()
print('loading data')
map_data.load_all_data()
##################
map_data.spectra = map_data.spectra[:200,:]
##################
map_data.get_map_dimensions()
map_data.trim_spectra(64)

maxima_spectrum = map_data.spectra.max(axis=0)

print('finding lines')
line_finder = LineFinder(
    maxima_spectrum,
    map_data.wvl,
    name='maxima'
)
line_finder.find_lines(
    height=250,
    threshold=None,
    distance=None,
    prominence=200,
    width=3,
    wlen=27,
    rel_height=1.2,
)
line_finder.load_nist_tables(
    Path('C:/Users/kepes/OneDrive - Vysoké učení technické v Brně/projects/marsData/inventory/nistTables')
)
line_finder.find_peaks_in_reference(
    maxima_spectrum,
    scale=False,
    show_cond=False
)

print('correcting baseline')
map_data.get_baseline(50,100)
map_data.baseline_correct()

print('integrating emission line intensities')
map_data.get_emission_line_intensities(
    line_finder.peaks[1].get('left_bases'),
    line_finder.peaks[1].get('right_bases'),
    line_centers=line_finder.peaks[0],
    intensity_func=np.max
)

# Embeddings
LATENT_SPACE_DIM = 3
## PCA from emission lines ----------------------------------------------
model_id = f'pca_{file_name}_from_lines_{time():.0f}'

pca_model = PCA(LATENT_SPACE_DIM)

print('PCA - embedding')
embeddings = pca_model.fit_transform(
    pd.DataFrame(map_data.line_intensities)
)
### plotting
fig = plot_embedding(
    embeddings,
    explained_variances=pca_model.explained_variance_ratio_.copy(),
    # colors=predicted_labels[clustering_method],
    marker_size=8,
    return_figure=True
)
fig.suptitle(
    f'PCA {LATENT_SPACE_DIM} comp.; emission lines'
)
fig.patch.set_alpha(0)
fig.tight_layout()
fig.savefig(
    f'./temp/{model_id}.png',
    transparent=True
)
### saving model and embeddings
print('PCA - saving model')
dump(
    pca_model,
    f'./temp/{model_id}.joblib'
)
print('PCA - saving embeddings')
np.save(
    file=f'./temp/embeddings_{model_id}.npy',
    arr=embeddings
)

# # TSNE
# tsne_model = TSNE(
#     n_components=3,
#     perplexity=30,
#     learning_rate=200
# )
# embeddings = tsne_model.fit_transform(
#     pd.DataFrame(map_data.line_intensities)
# )

# # UMAP
# umap_model = UMAP(n_components=3, n_neighbors=30, min_dist=0.5)
# embeddings = umap_model.fit_transform(
#     pd.DataFrame(map_data.line_intensities)
# )