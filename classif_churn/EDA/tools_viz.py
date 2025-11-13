import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import set_config
import matplotlib.pyplot as plt
from itertools import chain, cycle
from IPython.display import display_html
from sklearn.feature_selection import mutual_info_classif

def display_df(*args, titles=cycle([''])):
    """Shows dataframes side by side."""
    html_str = ''
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += '<br>'
        html_str += f'<p><strong>{title}</strong></p>'
        html_str += df.to_html().replace('table',
                                         'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)


def rank_features(dataset, target, features, rnd_state=1234):
    """Returns features scores.
    
    Parameters
    ----------
    dataset: pd.DataFrame. Dataset with all data.
    target: array. Ground truth of target.
    features: array. Names of the features.
    rnd_state: integer. Random state.
    
    Returns
    -------
    ranked_features: dictionary. Features and their scores.
    """
    
    features_scores = dict()
    
    for feature in features:
        mask = ~dataset[feature].isna()
        X = dataset[mask][[feature]]
        y = target[mask]
        discr_fts_mask = X.dtypes.values == 'int64'

        score = mutual_info_classif(
            X=X,
            y=y,
            discrete_features=discr_fts_mask,
            n_neighbors=5,
            random_state=rnd_state
        )
        features_scores[feature] = score[0]

    # Sort features according their scores (descending order)
    features_scores = dict(
        sorted(features_scores.items(), key=lambda x: x[1], reverse=True)
    )
    return features_scores

def set_seed(seed=1234):
    """Set seed for reproducibility."""
    np.random.seed(seed)


def set_display():
    """Set adjustment for visualization."""
    set_config(display='diagram')
    sns.set_palette('Dark2')  # Dark2 Set1 Paired Accent hot

    plt.style.use('fivethirtyeight')
    sns.set_context(
        'paper', 
        rc={
            'axes.titlesize': 12, 
            'axes.labelsize': 11, 
            'xtick.labelsize': 10, 
            'ytick.labelsize': 10,
            'legend.loc': 'upper right'
        }
    )

    # Set global settings for pandas
    pd.set_option('display.precision', 8)  # decimal number precision
    pd.set_option('display.max_columns', None)  # display all columns
    pd.set_option('display.max_rows', 100)  # display max=100 rows

def show_hist_cat(data, name, 
                  target, order=None, 
                  palette=None,
                  hue_order=['Active', 'PreLost'],
                  nbins=None, figsize=(14, 5),
                  legend_title='Статус клиента',
                  annotate=True, xlabel=None,
                  stat_norm='probability',
                  bbox=(1, 1)):
    """Shows histograms for categorical data."""
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    sns.countplot(
        data=data,
        x=name, 
        hue=target,
        hue_order=hue_order,
        order=order,
        palette=palette,
        ax=ax[0]
    )
    plt.setp(ax[0].get_xticklabels(), rotation=45, 
             rotation_mode='anchor', ha='right')
    ax[0].set_title('Диаграмма распределения клиентов', pad=15)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel('Количество')
#     ax[0].get_legend().set_bbox_to_anchor(bbox)
    ax[0].get_legend().set_title(legend_title)
    
    sns.histplot(
        data=data,
        x=name,
        hue=target,
        hue_order=hue_order,
        stat=stat_norm,
        multiple='fill',
        palette=palette,
        ax=ax[1]
    )
    
    plt.setp(ax[1].get_xticklabels(), rotation=45, 
             rotation_mode='anchor', ha='right')
    ax[1].set_title('Нормированная диаграмма распределения клиентов', pad=15)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Доля')
#     ax[1].get_legend().set_bbox_to_anchor(bbox)
    ax[1].get_legend().set_title(legend_title)
    
    # Set annotation
    if annotate:
        for i, p in enumerate(ax[0].patches):
            ax[0].annotate(
                f"{p.get_height():.0f}", 
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center',
                va='bottom',
                xytext=(0, 5), 
                textcoords='offset points',
                fontsize=8,
                rotation=90
            )

        for i, p in enumerate(ax[1].patches):
            ax[1].annotate(
                f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width() / 2, p.get_y() + p.get_height() * 0.5),
                ha='center',
                va='center',
                xytext=(0, 0), 
                textcoords='offset points',
            )

#     plt.tight_layout()
    plt.show()

def show_histograms(feature, target, bins=100, bins2=100, 
                    hue_order=[0, 1],
                    palette=None, figsize=(14, 5),
                    annotate=True, 
                    xlabel=None, ylabel=None,
                    legend_title='Статус клиента',
                    stat='count', stat_norm='probability',
                    binrange=None, binrange_norm=None, 
                    bbox=(1, 1)):
    """Plots histogram and normalized histogram
    for feature's values in context of target.

    Parameters
    ----------
    feature: array. Names of the feature.
    target: array. Name of the target.
    bins: int. Number of bins for histogram.
    bins2: int. Number of bins for normalized histogram.
    figsize: tuple. Size of figure.

    Returns
    -------
    None.
    """

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax = ax.reshape(-1)
    sns.histplot(
        x=feature,
        hue=target,
        hue_order=hue_order,
        stat=stat,
        common_norm=False,
        element='step',
        bins=bins,
        binrange=binrange,
        palette=palette,
        ax=ax[0]
    )
    ax[0].set_title('Гистограмма')
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
#     ax[0].get_legend().set_bbox_to_anchor(bbox)
    ax[0].get_legend().set_title(legend_title)

    sns.histplot(
        x=feature,
        hue=target,
        hue_order=hue_order,
        stat=stat_norm,
        element='bars',
        multiple='fill',
        bins=bins2,
        binrange=binrange_norm,
        palette=palette,
        ax=ax[1]
    )
    ax[1].set_title('Нормированная гистограмма')
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Доля')
#     ax[1].get_legend().set_bbox_to_anchor(bbox)
    ax[1].get_legend().set_title(legend_title)

    # Set annotation
    if annotate:
        for i, p in enumerate(ax[1].patches):
            ax[1].annotate(
                f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width() / 2, p.get_y() + p.get_height() * 0.5),
                ha='center',
                va='center',
                xytext=(0, 0), 
                textcoords='offset points',
            )
    
#     plt.tight_layout()
    plt.show()

def show_corr_matrix(dataset, features, corr_method='pearson', figsize=(10, 8)):
    """Plots correlation matrix.

    Parameters
    ----------
    dataset: pd.DataFrame. Dataset with all data.
    features: array. Names of the features.
    corr_method: {'pearson', 'kendall', 'spearman'}. Method of correlation.
    figsize: tuple. Size of figure.

    Returns
    -------
    None.
    """

    # Get correlation matrix
    corr_matrix = dataset[features].corr(method=corr_method)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        vmin=-1.,
        vmax=1.,
        annot=True,
        cmap='coolwarm',
        cbar=False
    )

    plt.show()
