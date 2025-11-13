import os
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from json import load
from pathlib import Path
from time import perf_counter
from datetime import timedelta, datetime
from train_predict_tools import train_predict_churn

# ----------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------

CORR_THR = 0.85
DATA_FOLDER = 'data'  # папка для локального хранения данных
GAIN_MATRIX = np.array([
    [0, -300],
    [-3000, 3000]
])

MODEL_FOLDER = 'model'
MODEL_PREFIX_NAME = 'mark_churn'
MODEL_VERSION = '1.0'
MOST_AVAIL_DT_REP = '2025-11-10'
NUM_OUTER_CV = 5

SCORE_MODE = 'predict'    # 'test', 'predict'
SEED = 1234
SUPPLEMENT_FOLDER = 'supplement'

# ----------------------------------------------------------------------
# Назначим alias для удобства
# ----------------------------------------------------------------------
CUS_ID = 'customerid'
SUR = 'surname'
TARGET = 'exited'

# ----------------------------------------------------------------------
# Настройки
# ----------------------------------------------------------------------
warnings.filterwarnings('ignore', category=DeprecationWarning)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%b-%d %H:%M:%S',
    level=logging.INFO
)

np.random.seed(SEED)
pd.set_option('display.precision', 8)  # decimal number precision

# ----------------------------------------------------------------------
# Основной запуск
# ----------------------------------------------------------------------
if __name__ == "__main__":
    t1 = perf_counter()

    train_predict_churn(
        most_avail_dt_rep=MOST_AVAIL_DT_REP,
        score_mode=SCORE_MODE,
        model_folder=MODEL_FOLDER,
        model_prefix_name=MODEL_PREFIX_NAME,
        model_version=MODEL_VERSION,
        rnd_state=SEED,
        gain_matrix=GAIN_MATRIX,
    )

    logging.info(
        f"Task completed in: {timedelta(seconds=perf_counter() - t1)}"
    )