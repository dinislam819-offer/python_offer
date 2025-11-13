from lightgbm import plot_importance
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    class_likelihood_ratios,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constant
METRICS_NAMES = [
    'AuPR', 
    'AuROC', 
    'Balanced Accuracy',
    'Brier loss',
    'F1',
    'Log loss'
]

def aggregate_cv_metrics(scores, model_name, columns=METRICS_NAMES):
    """Aggregates metrics for function cross_validate."""
    
    names = list(scores.keys())[2:]
    metric_val, metric_names = [], []
    for name in names:
        metric_val.append(f"{np.nanmean(scores[name]):.8f} +/- {np.nanstd(scores[name]):.8f}")
        metric_names.append(name.split('_')[1])

    cv_metrics = pd.DataFrame(
        {'Model': [model_name] * len(metric_names), 'Metric': metric_names, 'Value': metric_val}
    )
    cv_metrics = pd.pivot_table(cv_metrics, values='Value', columns='Metric', 
                                index='Model', aggfunc='first')
    cv_metrics = cv_metrics.reindex(columns=columns)
    
    return cv_metrics


# def calc_lift(y_true, y_pred_pbs):
#     overall_rate = sum(y_true) / len(y_true)
#     sort_by_pred = [(p, t) for p, t in sorted(zip(y_pred_pbs, y_true))]
#     i90 = int(round(len(y_true) * 0.9))
#     top_decile_count = sum([p[1] for p in sort_by_pred[i90:]])
#     top_decile = top_decile_count / (len(y_true) - i90)
#     lift = top_decile / overall_rate
#     return lift


def calc_log_loss_and_brier(y_true, y_pred_pbs, eps=0.05, weights=None):

    ll = np.mean(-(1 - y_true) * np.log(1 - y_pred_pbs) - y_true * np.log(y_pred_pbs))

    mask = y_true == 1
    br = np.mean(np.square(y_true[mask] - y_pred_pbs[mask]))
    
    alpha = np.log((1 - eps) / eps)
    result = (ll + alpha * br) / (1 + alpha)

    return result



def get_bucket_stats(y_true,
                     y_pred,
                     y_pred_pbs, 
                     num_points=21, 
                     class_names=['Класс 0', 'Класс 1'], 
                     samples_weights=None,
                     gain_matrix=np.array([[0, 0], [0, 0]])):
    """Shows precision, recall, f1 for different threshold.

    Parameters
    ----------
    y_true: array. Ground truth.
    prob: array. Predicted scores (0-1).

    Returns
    -------
    metrcis: pd.DataFrame. Threshold, precision, recall, f1, true negative,
        false positive, false negative, true positive.
    """
    # business summary
    bs = pd.DataFrame({
        'Класс': y_true,
        'pbs': y_pred_pbs
    })
    
    bs['Интервал скора'] = pd.cut(
        bs['pbs'], 
        bins=np.linspace(0, 1, num=num_points, endpoint=True),
        right=True,
        include_lowest=True
    )
    
    bs = pd.get_dummies(bs, columns=['Класс'])

    bs = bs[['Интервал скора', 'Класс_0', 'Класс_1']].groupby(by='Интервал скора').sum()
    bs = bs.rename(columns={
        'Класс_0': class_names[0],
        'Класс_1': class_names[1]
    })
    bs = bs.astype(dtype={
        class_names[0]: 'int',
        class_names[1]: 'int'
    })

    bs.loc['Общий итог'] = pd.Series(bs[class_names].sum(), index=class_names)
    
    scores = []
    for thr in np.linspace(0, 1, num=num_points, endpoint=True)[1:]:
        thr = round(thr, 2)
        y = (y_pred_pbs >= thr).astype('float')
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y, 
            average='binary',
            zero_division=0,
            sample_weight=samples_weights
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y, sample_weight=samples_weights).ravel()
        pos_lr, neg_lr = class_likelihood_ratios(y_true, y, sample_weight=samples_weights, raise_warning=False)
        gain = tn * gain_matrix[0, 0] + fp * gain_matrix[0, 1] + fn * gain_matrix[1, 0] + tp * gain_matrix[1, 1]
        scores.append((thr, p, r, f, tn, fp, fn, tp, pos_lr, neg_lr, gain))

    metrics = pd.DataFrame(
        data=scores,
        columns=['threshold', 'precision', 'recall',
                 'f1', 'tn', 'fp', 'fn', 'tp', 'pos_lr', 
                 'neg_lr', 'gain']
    )
    
    metrics = metrics.astype(dtype={
        'tn': 'int', 'fp': 'int',
        'fn': 'int', 'tp': 'int'
    })
    
    result = pd.concat([bs, metrics.set_index(keys=bs.index[:-1])], 
                       axis=1, join='outer').reset_index(names='Интервал скора')
    return result

# def gini_and_rate_top(y_true, y_pred, k=10, weights=None):
#     """Вычисляет Gini и долю положительных примеров в топ-k% по скору."""
#     # Приводим к numpy массивам (иначе Pandas индексы вызовут KeyError)
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)

#     n_pos = np.sum(y_true)
#     n_neg = y_true.shape[0] - n_pos

#     # Сортируем по предсказанию
#     indices = np.argsort(y_pred)[::-1]
#     y_true, y_pred = y_true[indices], y_pred[indices]

#     # Веса
#     if weights is None:
#         weights = np.ones(len(y_true))
#     cum_norm_weight = (weights * (1 / weights.sum())).cumsum()

#     # top-k% mask
#     pct_mask = cum_norm_weight <= k / 100
#     k_rate = np.sum(y_true[pct_mask]) / n_pos

#     # Gini
#     g = 2 * roc_auc_score(y_true, y_pred, sample_weight=weights) - 1

#     return 0.5 * (g + k_rate)


def get_metrics(y_true, y_pred, y_pred_pbs):    
    """Show performance on test dataset."""
    result = pd.DataFrame({
        'Metric': METRICS_NAMES,
        'Value': [average_precision_score(y_true, y_pred_pbs),
                  roc_auc_score(y_true, y_pred_pbs),
                  balanced_accuracy_score(y_true, y_pred),
                  brier_score_loss(y_true, y_pred_pbs),
                  f1_score(y_true, y_pred),
                  log_loss(y_true, y_pred_pbs)]
    })  #.set_index('Metric')
    
    return result

def grid_search_best_scores(model_cv, model_name):
    """Metrics from best model from grid search cv."""
    
    # model_name = model_cv.best_estimator_['classifier'].__class__.__name__
    bi = model_cv.best_index_
    metrics = {}
    mean_metric = None
    std_metric = None
    for key, val in model_cv.cv_results_.items():
        if 'mean_test' in key:
            mean_metric = val[bi]
            mtc = key.split('_')[-1]
        if 'std_test' in key and mtc == key.split('_')[-1]:
            std_metric = val[bi]

        if mean_metric is not None and std_metric is not None:
            metrics[mtc] = f"{mean_metric:.8f} +/- {std_metric:.8f}"
            mean_metric = None
            std_metric = None
    
    return pd.DataFrame(metrics, index=[model_name]).reset_index(names='Model')


def grid_search_scores(cv_results, rank_name='rank_test_Gini_RateTop'):
    """Make table with cv metrics from GridSearchCV."""
    cv_metrics = pd.DataFrame(cv_results).sort_values(by=rank_name)
    cv_metrics = cv_metrics.set_index(rank_name)
    cls = [cl for cl in cv_metrics.columns \
           if 'param_' in cl or 'mean_test_' in cl or 'std_test_' in cl]

    param_rename = {cl: cl.split('__')[1] for cl in cls if 'param_' in cl}
    cv_metrics = cv_metrics[cls].rename(columns=param_rename)

    test_rename = {cl: ' '.join(cl.split('_')[:1] + cl.split('_')[2:]) for cl in cv_metrics.columns \
                   if '_test_' in cl}
    cv_metrics = cv_metrics.rename(columns=test_rename)
    return cv_metrics.reset_index()


# def k_s_stat(y, y_pred):
#     return ks_2samp(y_pred[y == 0], y_pred[y == 1]).statistic


def show_top_features(model, n_top=30, figsize=(14, 15)):
    """Show features importances."""
    
    columns = [*model['preprocessor'].transformers_[0][2],
               *model['preprocessor'].transformers_[1][2]]
    columns_map = {f"Column_{i}": cl for i, cl in enumerate(columns)}
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax = ax.ravel()
    for i, imp_type in enumerate(['split', 'gain']):
        plot_importance(
            booster=model['classifier'],
            ax=ax[i],
            importance_type=imp_type,
            precision=1,
            max_num_features=n_top
        )
        labels = [item.get_text() for item in ax[i].get_yticklabels()]
        ax[i].set_yticklabels([columns_map[item] for item in labels])
        ax[i].set_title(f"{imp_type}")
    plt.tight_layout()
    plt.show()