
import numpy as np
import matplotlib.pyplot as plt

def plot_embedding(
    embedded_data,
    explained_variances=None,
    colors=None,
    marker_size:float=2.,
    return_figure:bool=False
):
    PERCENTAGE = 100
    if explained_variances is not None: explained_variances *= PERCENTAGE

    embedding_dimension = embedded_data.shape[1]
    plot_combination_count = embedding_dimension * (embedding_dimension-1) // 2
    db_logger.debug(f'embedding combinations: {plot_combination_count}')
    
    if plot_combination_count <= 9:
        plot_count = plot_combination_count 
        col_count = np.min([3,plot_count])
        row_count = np.min([3,plot_count // col_count])
        embedding_indices_to_plot = [
            x 
            for x 
            in np.array(np.meshgrid(
                np.arange(embedding_dimension - 1),
                np.arange(embedding_dimension - 1) + 1
            ))\
              .T\
              .reshape(-1,2) 
            if len(np.unique(x)) > 1
        ]
    else:
        plot_count = embedding_dimension
        col_count = np.min([3,int(np.sqrt(plot_count))])
        row_count = np.min([3,int(np.sqrt(plot_count))])
        embedding_indices_to_plot = np.arange(0,col_count*row_count*2).reshape(-1,2)

    db_logger.debug(f'plot_count: {plot_count}')
    db_logger.debug(f'cols, rows: {col_count}, {row_count}')

    if colors is None: colors = [1] * len(embedded_data)

    fig, ax = plt.subplots(
        ncols=col_count,
        nrows=row_count
    )

    if col_count == 1:
        ax.scatter(
            x=embedded_data[:,0],
            y=embedded_data[:,1],
            s=marker_size,
            alpha=.25,
            c=colors,
            linewidth=0
        )

        x_lab = f'PC1 ({explained_variances[0]:.2f} %)' if explained_variances is not None else f'latent dim. 1'
        y_lab = f'PC2 ({explained_variances[1]:.2f} %)' if explained_variances is not None else f'latent dim. 2'

        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.set_xticks([])
        ax.set_yticks([])

    elif col_count >= 2 and row_count == 1:
        for ndx,plot_ndx in enumerate(range(col_count)):
            ax[plot_ndx].scatter(
                x=embedded_data[:,embedding_indices_to_plot[ndx][0]],
                y=embedded_data[:,embedding_indices_to_plot[ndx][1]],
                s=marker_size,
                alpha=.25,
                c=colors,
                linewidth=0
            )

            x_lab = 'PC{} ({:.2f} %)'.format(
                embedding_indices_to_plot[ndx][0] +1,
                explained_variances[embedding_indices_to_plot[ndx][0]]
            ) if explained_variances is not None else 'latent dim. {}'.format(
                embedding_indices_to_plot[ndx][0] +1
            )
            y_lab = 'PC{} ({:.2f} %)'.format(
                embedding_indices_to_plot[ndx][1] +1,
                explained_variances[embedding_indices_to_plot[ndx][1]]
            ) if explained_variances is not None else 'latent dim. {}'.format(
                embedding_indices_to_plot[ndx][1] +1
            )            

            ax[plot_ndx].set_xlabel(x_lab)
            ax[plot_ndx].set_ylabel(y_lab)
            ax[plot_ndx].set_xticks([])
            ax[plot_ndx].set_yticks([])
    
    elif row_count >= 2:
        for ndx,plot_ndx in enumerate(range(col_count * row_count)):
            ax[plot_ndx // col_count,plot_ndx % row_count].scatter(
                x=embedded_data[:,embedding_indices_to_plot[ndx][0]],
                y=embedded_data[:,embedding_indices_to_plot[ndx][1]],
                s=marker_size,
                alpha=.25,
                c=colors,
                linewidth=0
            )

            x_lab = 'PC{} ({:.2f} %)'.format(
                embedding_indices_to_plot[ndx][0] +1,
                explained_variances[embedding_indices_to_plot[ndx][0]]
            ) if explained_variances is not None else 'latent dim. {}'.format(
                embedding_indices_to_plot[ndx][0] +1
            )
            y_lab = 'PC{} ({:.2f} %)'.format(
                embedding_indices_to_plot[ndx][1] +1,
                explained_variances[embedding_indices_to_plot[ndx][1]]
            ) if explained_variances is not None else 'latent dim. {}'.format(
                embedding_indices_to_plot[ndx][1] +1
            )

            ax[plot_ndx // col_count,plot_ndx % row_count].set_xlabel(x_lab)
            ax[plot_ndx // col_count,plot_ndx % row_count].set_ylabel(y_lab)
            ax[plot_ndx // col_count,plot_ndx % row_count].set_xticks([])
            ax[plot_ndx // col_count,plot_ndx % row_count].set_yticks([])

    fig.tight_layout()
    fig.show()

    if return_figure: return(fig)
    else: return(None)
