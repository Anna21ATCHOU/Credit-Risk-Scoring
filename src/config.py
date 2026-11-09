"""Configuration centrale du projet : chemins, seed, chargement des données.

Usage (depuis un notebook ou un script) :
    from src.config import CFG, SEED, TARGET, ID_COL, PATHS, load_data
"""
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ------------------------------------------------------------------ racine et config
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SEED = CFG["project"]["seed"]
TARGET = CFG["data"]["target"]
ID_COL = CFG["data"]["id_col"]
PATHS = {key: ROOT / value for key, value in CFG["paths"].items()}

for key in ("raw", "interim", "processed", "models", "figures"):
    PATHS[key].mkdir(parents=True, exist_ok=True)
PATHS["submission"].parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ logging
log = logging.getLogger("home_credit")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    log.addHandler(handler)
log.setLevel(logging.INFO)


# ------------------------------------------------------------------ reproductibilité
def set_seed(seed: int = SEED) -> None:
    """Fixe les graines aléatoires pour obtenir des résultats reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ------------------------------------------------------------------ chargement
def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Réduit la mémoire en abaissant la précision des types numériques."""
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    end = df.memory_usage(deep=True).sum() / 1024**2
    log.info("Mémoire : %.1f Mo -> %.1f Mo", start, end)
    return df


def load_data(reduce_mem: bool = True):
    """Charge application_train et application_test depuis data/raw."""
    train = pd.read_csv(PATHS["raw"] / CFG["data"]["train_file"])
    test = pd.read_csv(PATHS["raw"] / CFG["data"]["test_file"])
    log.info("Train : %s | Test : %s", train.shape, test.shape)
    if reduce_mem:
        train, test = reduce_mem_usage(train), reduce_mem_usage(test)
    return train, test


def load_table(name: str, reduce_mem: bool = True) -> pd.DataFrame:
    """Charge une table annexe (bureau, previous_application, pos_cash, ...)."""
    files = CFG["data"]["extra_files"]
    if name not in files:
        raise KeyError(f"Table inconnue : {name}. Choix : {list(files)}")
    df = pd.read_csv(PATHS["raw"] / files[name])
    log.info("%s : %s", name, df.shape)
    return reduce_mem_usage(df) if reduce_mem else df


def load_columns_description() -> pd.DataFrame:
    """Charge le dictionnaire des variables (encodage latin-1 sur le fichier Kaggle)."""
    path = PATHS["raw"] / CFG["data"]["description_file"]
    return pd.read_csv(path, encoding="latin-1")


def sanity_checks(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Contrôles de base avant toute analyse."""
    assert TARGET in train.columns, f"{TARGET} absente du train"
    assert TARGET not in test.columns, f"{TARGET} ne doit pas être dans le test"
    assert train[ID_COL].is_unique, "Identifiants dupliqués dans le train"
    assert test[ID_COL].is_unique, "Identifiants dupliqués dans le test"
    assert set(test.columns) <= set(train.columns), "Colonnes du test absentes du train"
    log.info("Taux de défaut : %.2f %% | contrôles OK", 100 * train[TARGET].mean())
