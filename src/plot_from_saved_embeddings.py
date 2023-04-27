from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from src.MapData import MapData
    from src.LineFinder import LineFinder
    from src.plotting_functions import plot_embedding, plot_single_variable_map
except:
    from MapData import MapData
    from LineFinder import LineFinder
    from plotting_functions import plot_single_variable_map

file_name = '2022_03_22_P56B_307x532'
map_data = MapData(f'./data/Rakoviny/P56B/{file_name}.libsdata')
map_data.get_metadata()
map_data.get_map_dimensions()

for file_name in Path('./temp').glob('*.npy'):
    print(file_name)
    embedding = np.load(file=file_name)

    image_data = np.concatenate([
        map_data.vector_to_array(
            embedding[:,0]
        )[...,np.newaxis],
        map_data.vector_to_array(
            embedding[:,1]
        )[...,np.newaxis],
        map_data.vector_to_array(
            embedding[:,2]
        )[...,np.newaxis]],
        axis=2
    )

    image_data -= image_data.min()
    image_data /= image_data.max()

    fig, ax = plt.subplots(
        figsize=(9,9),
        nrows=1,
        ncols=1
    )
    ax.imshow(image_data)
    fig.tight_layout()

    fig.savefig(
        file_name.with_suffix('.png'),
        transparent=True,
        dpi=300
    )
