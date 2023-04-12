
import numpy as np
import matplotlib.pyplot as plt

def plot_embedding(
    embedded_data,
    explained_variances=None,
    colors=None
):
    if explained_variances is not None: explained_variances *= 100
    pc_count = embedded_data.shape[1]

    if colors is None: colors = [1] * len(embedded_data)

    col_count = np.min([
        3,
        int(np.sqrt(pc_count // 2))
    ])
    row_count = np.min([
        3, 
        int(np.sqrt(pc_count // 2))
    ])

    fig, ax = plt.subplots(
        ncols=col_count,
        nrows=row_count
    )

    if col_count * row_count == 1:
        ax.scatter(
            x=embedded_data[:,0],
            y=embedded_data[:,1],
            s=.5,
            alpha=.2,
            c=colors
        )

        x_lab = f'PC1 ({explained_variances[0]:.2f} %)' if explained_variances is not None else f'latent dim. 1'
        y_lab = f'PC2 ({explained_variances[1]:.2f} %)' if explained_variances is not None else f'latent dim. 2'

        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.set_xticks([])
        ax.set_yticks([])

    elif col_count * row_count >= 2:
        for pc_ndx in range(col_count * row_count):
            ax[pc_ndx // col_count,pc_ndx % row_count].scatter(
                x=embedded_data[:,pc_ndx*2],
                y=embedded_data[:,pc_ndx*2+1],
                s=.5,
                alpha=.2,
                c=colors
            )

            x_lab = f'PC{pc_ndx*2 + 1} ({explained_variances[pc_ndx*2]:.2f} %)' if explained_variances is not None else f'latent dim. {pc_ndx*2 + 1}'
            y_lab = f'PC{pc_ndx*2 + 2} ({explained_variances[pc_ndx*2 + 1]:.2f} %)' if explained_variances is not None else f'latent dim. {pc_ndx*2 + 2}'

            ax[pc_ndx // col_count,pc_ndx % row_count].set_xlabel(x_lab)
            ax[pc_ndx // col_count,pc_ndx % row_count].set_ylabel(y_lab)
            ax[pc_ndx // col_count,pc_ndx % row_count].set_xticks([])
            ax[pc_ndx // col_count,pc_ndx % row_count].set_yticks([])

    fig.tight_layout()
    fig.show()
