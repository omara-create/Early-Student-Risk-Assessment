"""Shared utilities for the OULAD early-risk experiment."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
CONFIG = ROOT / "configs" / "hyperparameters.json"

REQUIRED_FILES = [
    "courses.csv",
    "assessments.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "vle.csv",
    "studentVle.csv",
]


def load_config() -> dict:
    with open(CONFIG, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_raw_data() -> None:
    missing = [f for f in REQUIRED_FILES if not (DATA_RAW / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing OULAD files: "
            + ", ".join(missing)
            + "\nDownload OULAD from https://analyse.kmi.open.ac.uk/open_dataset "
              "and place the seven CSV files in data/raw/."
        )


def load_oulad() -> dict[str, pd.DataFrame]:
    ensure_raw_data()
    return {f[:-4]: pd.read_csv(DATA_RAW / f) for f in REQUIRED_FILES}


def save_csv(df: pd.DataFrame, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    df.to_csv(path, index=False)
    return path


def numeric_clip(series: pd.Series, lo=None, hi=None) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if lo is not None:
        x = x.clip(lower=lo)
    if hi is not None:
        x = x.clip(upper=hi)
    return x


def safe_divide(a, b):
    return np.where(np.asarray(b) == 0, 0, np.asarray(a) / np.asarray(b))
