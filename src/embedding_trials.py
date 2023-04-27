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

map_data = MapData(f'./data/Rakoviny/P56B/{file_name}.libsdata')
map_data.get_metadata()
map_data.load_wavelenths()
print('loading data')
map_data.load_all_data()
##################
# map_data.spectra = map_data.spectra[:200,:]
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
    Path('D:/OneDrive - Vysoké učení technické v Brně/projects/marsData/inventory/nistTables')
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

def process(
    model,
    model_type: str,
    model_id: str,
    data
):
    model_id = f'{model_type.lower()}{model_id}'
    print(f'{model_type} - embedding')
    embeddings = model.fit_transform(data)
    ### plotting
    fig = plot_embedding(
        embeddings,
        # explained_variances=model.explained_variance_ratio_.copy(),
        # colors=predicted_labels[clustering_method],
        marker_size=8,
        return_figure=True
    )
    fig.suptitle(
        f'{model_type} {LATENT_SPACE_DIM} comp.; emission lines'
    )
    fig.patch.set_alpha(0)
    fig.tight_layout()
    fig.savefig(
        f'./temp/{model_id}.png',
        transparent=True
    )
    ### saving model and embeddings
    print(f'{model_type} - saving model')
    dump(
        model,
        f'./temp/{model_id}.joblib'
    )
    print(f'{model_type} - saving embeddings')
    np.save(
        file=f'./temp/embeddings_{model_id}.npy',
        arr=embeddings
    )

    plt.close()

    return None

# Embeddings
LATENT_SPACE_DIM = 3
## PCA from emission lines ----------------------------------------------
process(
    model=PCA(LATENT_SPACE_DIM),
    model_type='PCA',
    model_id=f'_{file_name}_from_lines_{time():.0f}',
    data=pd.DataFrame(map_data.line_intensities)
)
## PCA from whole spectra ----------------------------------------------
process(
    model=PCA(LATENT_SPACE_DIM),
    model_type='PCA',
    model_id=f'_{file_name}_from_spectra_{time():.0f}',
    data=pd.DataFrame(map_data.spectra)
)
## tSNE from emission lines ----------------------------------------------
for perp in [5,10,20,30,50]:
    process(
        model=TSNE(
            n_components=LATENT_SPACE_DIM,
            perplexity=perp,
            learning_rate='auto',
            random_state=97481
        ),
        model_type='tSNE',
        model_id=f'_{file_name}_from_lines_{time():.0f}',
        data=pd.DataFrame(map_data.line_intensities)
    )
    ## tSNE from whole spectra ----------------------------------------------
    process(
        model=TSNE(
            n_components=LATENT_SPACE_DIM,
            perplexity=perp,
            learning_rate='auto',
            random_state=97481
        ),
        model_type='tSNE',
        model_id=f'_{file_name}_from_spectra_{time():.0f}',
        data=pd.DataFrame(map_data.spectra)
    )
## UMAP from emission lines ----------------------------------------------
for metric in ['cosine','euclidean']:
    for nn in [5,15,30,50]:
        process(
            model=UMAP(
                n_components=LATENT_SPACE_DIM,
                n_neighbors=nn,
                min_dist=0.1, #default
                metric=metric,
                output_metric='euclidean', #default
                n_epochs=200,
                learning_rate=1, #default
                random_state=97481
            ),
            model_type='UMAP',
            model_id=f'_{file_name}_from_lines_{time():.0f}',
            data=pd.DataFrame(map_data.line_intensities)
        )
        ## UMAP from whole spectra ----------------------------------------------
        process(
            model=UMAP(
                n_components=LATENT_SPACE_DIM,
                n_neighbors=nn,
                min_dist=0.1, #default
                metric=metric,
                output_metric='euclidean', #default
                n_epochs=200,
                learning_rate=1, #default
                random_state=97481
            ),
            model_type='UMAP',
            model_id=f'_{file_name}_from_spectra_{time():.0f}',
            data=pd.DataFrame(map_data.spectra)
        )