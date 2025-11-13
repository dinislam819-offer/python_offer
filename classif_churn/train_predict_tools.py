# Import dependencies
import logging
import numpy as np
import pandas as pd
import warnings
import os
import glob
import joblib
import json

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    make_scorer,
    roc_auc_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split

from collections import Counter, defaultdict
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.model_selection import RandomizedSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, RobustScaler

from xgboost import XGBClassifier


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

# ----------------- Константы ---------------------------
SUR = 'surname'
TARGET = 'target'
# ----------------- Метрики ----------------------------

def calc_lift(y_true: np.ndarray, y_pred_pbs: np.ndarray) -> float:
    """Вычисляет Lift для верхнего дециля предсказаний."""
    overall_rate = y_true.mean()
    sorted_idx = np.argsort(y_pred_pbs)
    i90 = int(round(len(y_true) * 0.9))
    top_decile_count = y_true[sorted_idx[i90:]].sum()
    top_decile = top_decile_count / (len(y_true) - i90)
    return top_decile / overall_rate


def calc_log_loss_and_brier(y_true: np.ndarray, y_pred_pbs: np.ndarray) -> float:
    """Смешанная метрика: log loss и brier score с учетом редкого класса."""
    y_pred_pbs = np.clip(y_pred_pbs, 1e-15, 1 - 1e-15)
    ll = log_loss(y_true, y_pred_pbs)
    mask = y_true == 1
    br = np.mean((y_true[mask] - y_pred_pbs[mask])**2)
    scale_pos_weight = (y_true == 0).sum() / (y_true == 1).sum()
    alpha = 0.5 * np.log(scale_pos_weight)
    return (ll + alpha * br) / (1 + alpha)


def calc_brier_loss_weighted(y_true: np.ndarray, y_pred_pbs: np.ndarray) -> float:
    """Взвешенный Brier loss по классам."""
    mask0, mask1 = y_true == 0, y_true == 1
    return np.mean([
        brier_score_loss(y_true[mask0], y_pred_pbs[mask0]),
        brier_score_loss(y_true[mask1], y_pred_pbs[mask1])
    ])


def calc_log_loss_weighted(y_true: np.ndarray, y_pred_pbs: np.ndarray) -> float:
    """Взвешенный log loss для несбалансированных классов."""
    sample_weight = np.where(
        y_true == 1, 
        (y_true == 0).sum() / (y_true == 1).sum(), 
        1.0
    )
    return log_loss(y_true, y_pred_pbs, sample_weight=sample_weight)


def calc_focal_loss(y_true: np.ndarray, y_pred_pbs: np.ndarray, gamma: float = 0.01) -> float:
    """Focal loss для бинарной классификации."""
    y_pred_pbs = np.clip(y_pred_pbs, 1e-15, 1 - 1e-15)
    alpha = (y_true == 0).sum() / len(y_true)
    fl = -y_true * np.log(y_pred_pbs) * alpha * (1 - y_pred_pbs)**gamma \
         - (1 - y_true) * np.log(1 - y_pred_pbs) * (1 - alpha) * y_pred_pbs**gamma
    return fl.mean()

# ----------------- Словарь scorer'ов -------------------
SEARCH_METRIC = 'Log loss weight'
SCORE_DICT = {
    'AuPR': 'average_precision',
    'AuROC': 'roc_auc',
    'Brier loss': 'neg_brier_score',
    'Brier loss weight': make_scorer(calc_brier_loss_weighted, greater_is_better=False, response_method='predict_proba'), 
    'Lift': make_scorer(calc_lift, greater_is_better=True, response_method='predict_proba'),
    'Lloss&Brier': make_scorer(calc_log_loss_and_brier, greater_is_better=False, response_method='predict_proba'),
    'Log loss': 'neg_log_loss',
    'Log loss weight': make_scorer(calc_log_loss_weighted, greater_is_better=False, response_method='predict_proba'),
    'Focal loss': make_scorer(calc_focal_loss, greater_is_better=False, response_method='predict_proba')
}

# ----------------- Функция для расчета всех метрик -------------
def calc_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_pred_pbs: np.ndarray
) -> dict[str, float]:
    """Вычисляет различные метрики для оценки качества модели.
    
    Parameters
    ----------
    y_true: array-like. Таргет.
    y_pred: array-like. Спрогнозированный таргет. Значения 0 или 1.
    y_pred_pbs: array-like. Спрогнозированный скор. Значения от 0 до 1.
    
    Returns
    -------
    metrics: dict. Метрики.
    """
    return {
        'AuPR': average_precision_score(y_true, y_pred_pbs),
        'AuROC': roc_auc_score(y_true, y_pred_pbs),
        'Brier loss': brier_score_loss(y_true, y_pred_pbs),
        'Brier loss weight': calc_brier_loss_weighted(y_true, y_pred_pbs),
        'Lift': calc_lift(y_true, y_pred_pbs),
        'Lloss&Brier': calc_log_loss_and_brier(y_true, y_pred_pbs),
        'Log loss': log_loss(y_true, y_pred_pbs),
        'Log loss weight': calc_log_loss_weighted(y_true, y_pred_pbs),
        'Focal loss': calc_focal_loss(y_true, y_pred_pbs),
    }

# ----------------- Анализ по бакетам -----------------------
def get_bucket_stats(
    y_true: np.ndarray,
    y_pred_pbs: np.ndarray,
    num_points: int = 21,
    class_names: list[str] = ['Класс 0', 'Класс 1'],
    sample_weights: np.ndarray = None,
    gain_matrix: np.ndarray = None
) -> pd.DataFrame:
    """Shows precision, recall, f1 for different threshold.

    Parameters
    ----------
    y_true: array-like. Ground truth.
    y_pred_pbs: array-like. Predicted scores [0-1].
    num_points: int. Кол-во бакетов.
    class_names: list. Наименования классов.
    samples_weights: array-like. Веса для наблюдений.
    gain_matrix: array-like. Матрица для расчета стоимости.

    Returns
    -------
    result: pd.DataFrame. Threshold, precision, recall, f1, true negative,
        false positive, false negative, true positive, pos_lr, neg_lr, gain
    """
    if gain_matrix is None:
        gain_matrix = np.zeros((2, 2))

    # бизнес-статистика
    df = pd.DataFrame({'Класс': y_true, 'pbs': y_pred_pbs})
    df['Интервал скора'] = pd.cut(df['pbs'], bins=np.linspace(0, 1, num_points), include_lowest=True)
    df = pd.get_dummies(df, columns=['Класс'])
    df = df.groupby('Интервал скора', observed=False).sum()
    df = df.rename(columns={0: class_names[0], 1: class_names[1]}).astype(int)
    df.loc['Общий итог'] = df.sum()

    scores = []
    thresholds = np.linspace(0, 1, num_points)[1:]
    for thr in thresholds:
        y_bin = (y_pred_pbs >= thr).astype(float)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_bin, average='binary', zero_division=0, sample_weight=sample_weights
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin, sample_weight=sample_weights).ravel()
        gain = tn * gain_matrix[0, 0] + fp * gain_matrix[0, 1] + fn * gain_matrix[1, 0] + tp * gain_matrix[1, 1]
        scores.append((thr, p, r, f, tn, fp, fn, tp, gain))

    # Создаем DataFrame по порогам
    metrics = pd.DataFrame(
        scores,
        columns=['threshold', 'precision', 'recall', 'f1', 'tn', 'fp', 'fn', 'tp', 'gain']
    )

    # Для объединения создаем индекс из интервалов (без "Общий итог")
    interval_index = df.index[:-1]
    metrics.index = interval_index

    # Объединяем
    result = pd.concat([df, metrics], axis=1, join='outer').reset_index(names='Интервал скора')
    return result

# ----------------- Чтение данных -----------------------------
def load_data(
        file_path: str
    ) -> pd.DataFrame:
    """Загружает CSV файл и возвращает DataFrame.
    
    Parameters
    ----------
    file_path : str Путь к CSV файлу.    
    
    Returns
    -------
    pd.DataFrame Загруженные данные.
    """
    return pd.read_csv(file_path)


# ----------------- Преобразование данных -----------------------
def transform_data(
        data: pd.DataFrame, 
        age_col: str = 'age'
    ) -> pd.DataFrame:
    """Преобразование данных и генерация новых атрибутов.
    
    Parameters
    ----------
    data: pd.Dataframe. Исходный датасет.

    Returns
    -------
    data: pd.Dataframe. Датасет, с трансформированными атрибутами.
    """
    data = data.copy()
    data.columns = data.columns.str.lower()
    
    # Создаем уникальный идентификатор
    data['ids'] = data['id'].astype(str) + '-' + data['customerid'].astype(str)
    
    data = data.drop(columns=['id', 'customerid'])
    data = data.rename(columns={'exited': 'target'})
    
    # Приведение возраста к int
    data[age_col] = data[age_col].astype(int)
    
    return data

# ----------------- Деление данных на train, val и test -----------------------------
def split_data(
        data: pd.DataFrame, 
        features: list[str], 
        rnd_state: int
    ):
    """Формирует различные виды датасетов: обучающий, валидационный и тестовый.
    
    Parameters
    ----------
    data: pd.Dataframe. Датасет, который будет разделяться на train и test.
    features: array-like. Список атрибутов.
    rnd_state: int. Random state.
    
    Returns
    -------
    X_test: pd.Dataframe. Датасет для тестирования модели.
    y_test: array-like. Таргет для тестового датасета.
    X_train: pd.Dataframe. Датасет для обучения модели. 
    y_train: array-like. Таргет для обучающего датасета. 
    X_val: pd.Dataframe. Валидационный датасет для обучения и поиска гиперпараметров с учетом 
        early_stopping (там, где early_stopping доступен)
    y_val: array-like. Таргет для валидационного датасета.
    """
    X = data[['surname', 'ids'] + features].copy()
    y = data['target'].to_numpy().ravel()

    # Простое stratify только по таргету
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        shuffle=True,
        random_state=rnd_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        shuffle=True,
        random_state=rnd_state
    )

    # Логирование распределения классов
    for name, y_ in zip(['X_train', 'X_val', 'X_test'], [y_train, y_val, y_test]):
        logging.info(f"{name} размер: {len(y_)}")
        for k, v in Counter(y_).items():
            logging.info(f"Класс {k}: {v} ({v / len(y_) * 100:.2f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test

# ----------------- Создание препроцессора -----------------------------

def create_age_bin(df: pd.DataFrame, age_col: str = 'age') -> pd.DataFrame:
    bins = [0, 13, 17, 25, 35, 45, 60, np.inf]
    labels = ['0-13', '14-17', '18-25', '26-35', '36-45', '46-60', '60+']
    df['age_bin'] = pd.cut(df[age_col], bins=bins, labels=labels, right=True, include_lowest=True)
    return df


def create_preprocessor(
    cat_fts: list[str], 
    num_fts: list[str]
) -> ColumnTransformer:
    """"Создает препроцессор для обработки данных.
    
    Parameters
    ----------
    cat_fts: array-like. Категориальные атрибуты.
    num_fts: array-like. Числовые атрибуты.

    Returns
    -------
    preprocessor: ColumnTransformer. Преобразователь данных.
    """
    preprocessor = ColumnTransformer(
        [
            # gender → 0/1
            ('gender', Pipeline([
                ('map_gender', FunctionTransformer(lambda x: x.replace({'Female': 0, 'Male': 1})))
            ]), ['gender']),
            
            # age_bin → порядковое кодирование
            ('age_bin', Pipeline([
                ('ordinal_enc', OrdinalEncoder(
                    categories=[['0-13', '14-17', '18-25', '26-35', '36-45', '46-60', '60+']],
                    handle_unknown='use_encoded_value', unknown_value=-1,
                    dtype=np.int16
                ))
            ]), ['age_bin']),
            
            # geography → порядковое кодирование
            ('geography', Pipeline([
                ('ordinal_enc', OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=-1,
                    dtype=np.int16
                ))
            ]), ['geography']),
            
            # числовые признаки
            ('num_fts', RobustScaler(), num_fts)
        ],
        remainder='drop'
    )
    return preprocessor

# ----------------- Обучение модели с подбором гиперпараметров -----------------

def train_model(
    preprocessor,
    classifier,
    param_grid,
    X_train,
    y_train,
    groups=None,
    scoring=SCORE_DICT,
    n_iter=50,
    random_state=42
):
    """
    Полная обертка для обучения одной модели с подбором гиперпараметров.
    Используется StratifiedGroupKFold для несбалансированных данных.

    Parameters
    ----------
    preprocessor: ColumnTransformer Препроцессор для обработки данных.
    classifier: sklearn-compatible estimator Классификатор (например, XGBClassifier).
    param_grid: dict Сетка параметров для RandomizedSearchCV.
    X_train: pd.DataFrame Обучающий датасет.
    y_train: np.array Таргет обучающего датасета.
    groups: array-like, optional руппы для StratifiedGroupKFold кросс-валидации.
    scoring: dict Метрики для оценки модели.
    n_iter: int Количество итераций RandomizedSearchCV.
    random_state: int Random state.

    Returns
    -------
    model_cv: RandomizedSearchCV
        Обученная модель с найденными лучшими гиперпараметрами.
    train_metrics: dict
        Метрики модели на обучающей выборке.
    """

    # Создаем pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Настройка кросс-валидации
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # RandomizedSearchCV
    model_cv = RandomizedSearchCV(
        estimator=model_pipeline,
        param_distributions=param_grid,
        scoring=scoring,
        cv=cv,
        refit=list(scoring.keys())[0],
        n_iter=n_iter,
        n_jobs=-1,
        random_state=random_state
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        if groups is not None:
            model_cv.fit(X_train, y_train, groups=groups)
        else:
            model_cv.fit(X_train, y_train)

    # Предсказания на обучающей выборке
    y_pred = model_cv.predict(X_train)
    y_pred_proba = model_cv.predict_proba(X_train)[:, 1]

    # Вычисляем метрики
    train_metrics = calc_metrics(y_train, y_pred, y_pred_proba)

    return model_cv, train_metrics


# ----------------- Сохранение модели -----------------
def save_model(
    trained_model,
    model_folder: str,
    model_prefix_name: str,
    model_version: str,
    most_avail_dt_rep: str
) -> None:
    """Сохраняет обученную модель и удаляет старые файлы, если их больше 6."""
    os.makedirs(model_folder, exist_ok=True)
    year_month = ''.join(most_avail_dt_rep.split('-')[:-1][::-1])
    file_name = f"{model_prefix_name}_xgb_v{model_version}_{year_month}.pickle"
    file_path = os.path.join(model_folder, file_name)
    
    if isinstance(trained_model, RandomizedSearchCV):
        model_to_save = trained_model.best_estimator_.named_steps['classifier']
    elif hasattr(trained_model, 'named_steps') and 'classifier' in trained_model.named_steps:
        model_to_save = trained_model.named_steps['classifier']
    else:
        model_to_save = trained_model

    joblib.dump(model_to_save, file_path)
    
    # Удаляем самый старый файл, если больше 6
    all_files = sorted(glob.glob(os.path.join(model_folder, "*.pickle")), key=os.path.getmtime)
    if len(all_files) > 6:
        os.remove(all_files[0])

# ----------------- Скоринг -----------------------------
def scoring_unseen_data(
    trained_model,  # одна обученная модель
    X: pd.DataFrame, 
    y: np.ndarray, 
    score_mode: str, 
    gain_matrix: np.ndarray = None
):
    """Скоринг на hold-out данных для одной модели.
    
    Parameters
    ----------
    trained_model: sklearn estimator. Обученная модель.
    X: pd.DataFrame. Тестовая выборка.
    y: np.array. Таргет.
    score_mode: str, {'test', 'predict'}.
    gain_matrix: np.ndarray. Матрица затрат для стоимостного анализа (необязательно).

    Returns
    -------
    scores: pd.DataFrame. Наблюдения и предсказания.
    test_metrics: dict. Метрики модели.
    models_buckets: pd.DataFrame. Разбивка по бакетам (если score_mode='test').
    """
    y_pred = trained_model.predict(X)
    y_pred_proba = trained_model.predict_proba(X)[:, 1]

    # Создаем DataFrame с предсказаниями
    scores = X.copy()
    scores['churn_prediction'] = y_pred
    scores['churn_score'] = y_pred_proba

    test_metrics = {}
    models_buckets = {}

    if score_mode == 'test':
        test_metrics = calc_metrics(y, y_pred, y_pred_proba)
        models_buckets = get_bucket_stats(y, y_pred_proba, gain_matrix=gain_matrix)

    return scores, test_metrics, models_buckets


# ----------- Сохранение скоров -----------------
def save_scores(
    scores: pd.DataFrame, 
    model_prefix_name: str,
    model_version: str,
    most_avail_dt_rep: str
) -> None:
    """Сохранение скоров в файл.

    Parameters
    ----------
    scores: pd.DataFrame. Наблюдения и предсказания.
    model_prefix_name: str. Префикс в наименовании модели.
    model_version: str. Версия модели.
    most_avail_dt_rep: str. Последняя доступная дата 'YYYY-MM-DD'.

    Returns
    -------
    None
    """
    os.makedirs("scores", exist_ok=True)
    year_month = ''.join(most_avail_dt_rep.split('-')[:2])  # 'YYYYMM'
    file_name = f"{model_prefix_name}_v{model_version}_{year_month}_scores.csv"
    file_path = os.path.join("scores", file_name)

    scores.to_csv(file_path, index=False)
    logging.info(f"Скоры сохранены в {file_path}")

# ----------- Общая функция -----------------

def train_predict_churn(
    score_mode: str,
    model_folder: str,
    model_prefix_name: str,
    model_version: str,
    rnd_state: int,
    gain_matrix: np.array,
    most_avail_dt_rep: str, 
    file_path: str = 'train.csv'
) -> None:
    """Обертка для функций, обозначенных выше.

    Parameters
    ----------
    score_mode: str, {'test', 'predict'} Режим скоринга.
    model_folder: str Папка для сохранения модели.
    model_prefix_name: str Префикс имени модели.
    model_version: str Версия модели.
    rnd_state: int Random state.
    gain_matrix: np.array Матрица затрат.
    obs_target_period: int Период наблюдения таргета.
    file_path: str Путь к CSV файлу.
    """

    # ---------- Шаг 1. Чтение данных ----------
    logging.info("Загрузка данных...")
    data = load_data(file_path)
    logging.info(f"Загружено {len(data)} строк.")

    # ---------- Шаг 2. Преобразование данных ----------
    logging.info("Трансформация данных...")
    data = transform_data(data)
    data = create_age_bin(data, age_col='age')

    # Категориальные и числовые признаки
    cat_fts = ['gender', 'age_bin', 'geography']  # gender: 0/1, age_bin и geography: ordinal
    num_fts = [col for col in data.select_dtypes(include=['int', 'float']).columns 
               if col not in [TARGET, 'ids']]  

    # ---------- Шаг 3. Деление на train/val/test ----------
    logging.info("Разделение на train, val, test...")
    features = cat_fts + num_fts
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(data, features, rnd_state=rnd_state)

    # ---------- Шаг 4. Создание препроцессора ----------
    logging.info("Создание препроцессора...")
    preprocessor = create_preprocessor(cat_fts=cat_fts, num_fts=num_fts)
    preprocessor.fit(X_train)
    logging.info(f"Препроцессор обучен. Размерность после трансформации: {preprocessor.transform(X_train).shape}")

    # ---------- Шаг 5. Определение модели и сетки гиперпараметров ----------
    logging.info("Инициализация модели...")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    clf = XGBClassifier(
        n_estimators=750,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=rnd_state,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )

    param_grid = {
    'classifier__max_depth': [3, 4, 5, 6, 8],                # глубина дерева
    'classifier__min_child_weight': [1, 3, 5, 7],            # минимальный вес узла (контролирует переобучение)
    'classifier__gamma': [0, 0.1, 0.3, 0.5, 1],              # минимальное снижение потерь для сплита
    'classifier__subsample': [0.6, 0.8, 1.0],                # доля выборки для каждого дерева
    'classifier__colsample_bytree': [0.6, 0.8, 1.0],         # доля признаков для каждого дерева
    'classifier__learning_rate': [0.01, 0.05, 0.1],          # скорость обучения (eta)
    'classifier__n_estimators': [200, 400, 800],             # число деревьев (больше — дольше)
    # 'classifier__booster': ['gbtree', 'dart'],               # dart может улучшить результаты (Dropout)
    'classifier__scale_pos_weight': [1, 3, 5, 10, 20],       # важен при дисбалансе классов
    # 'classifier__max_delta_step': [0, 1, 5],                 # помогает при дисбалансе
    }

    # ---------- Шаг 6. Поиск гиперпараметров и обучение ----------
    logging.info("Обучение модели с подбором гиперпараметров...")

    model_cv, train_metrics = train_model(
        preprocessor=preprocessor,
        classifier=clf,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
        groups=None,
        scoring=SCORE_DICT,
        n_iter=50,
        random_state=rnd_state
    )

    best_params = model_cv.best_params_
    logging.info(f"Лучшие параметры: {best_params}")
    logging.info(f"Метрики на обучении: {train_metrics}")

    trained_models = {'xgb_custom': model_cv}

    # ---------- Сохранение гиперпараметров ----------
    params_folder = os.path.join(model_folder, "params")
    os.makedirs(params_folder, exist_ok=True)

    params_file = os.path.join(params_folder, f"{model_prefix_name}_xgb_v{model_version}_params.json")
    with open(params_file, "w") as f:
        json.dump(best_params, f, indent=4)

    logging.info(f"Гиперпараметры сохранены в {params_file}")

    # ---------- Шаг 7. Сохранение модели ----------
    logging.info("Сохранение модели...")

    best_model = trained_models['xgb_custom'].best_estimator_.named_steps['classifier']

    save_model(
        trained_model=trained_models['xgb_custom'],
        model_folder=model_folder,
        model_prefix_name=model_prefix_name,
        model_version=model_version,
        most_avail_dt_rep=most_avail_dt_rep
    )
    logging.info("Модель сохранена.")

    # ---------- Шаг 8. Скоринг ----------
    logging.info("Скоринг тестовой выборки...")
    X_test_proc = X_test.copy()
    y_test_proc = y_test.copy()

    scores, test_metrics, models_buckets = scoring_unseen_data(
        trained_model=trained_models['xgb_custom'],
        X=X_test_proc,
        y=y_test_proc,
        score_mode=score_mode,
        gain_matrix=gain_matrix
    )

    if score_mode == 'test':
        logging.info(f"Метрики на тесте: {test_metrics}")

    logging.info("Скоринг завершен.")

    # ----------- Шаг 9. Сохранение скоров -----------------
    logging.info("Сохранение скоров...")
    save_scores(
        scores,
        model_prefix_name=model_prefix_name,
        model_version=model_version,
        most_avail_dt_rep=most_avail_dt_rep
    )
    logging.info("Сохранение скоров завершено.")


    # "classifier__subsample": 0.6,
    # "classifier__scale_pos_weight": 1,
    # "classifier__n_estimators": 800,
    # "classifier__min_child_weight": 3,
    # "classifier__max_depth": 6,
    # "classifier__learning_rate": 0.01,
    # "classifier__gamma": 0.1