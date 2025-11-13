# Import dependencies
from IPython.display import display_html
from itertools import chain, cycle

from sklearn import set_config
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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