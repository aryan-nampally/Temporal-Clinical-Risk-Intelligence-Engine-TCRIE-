"""
TCRIE - Data Ingestion Layer
Parses CDISC SDTM .xpt (SAS Transport) files into structured DataFrames.
Author: Nampally Aryan
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Standard SDTM domain filenames
DOMAIN_FILES = {
    "DM": "dm.xpt",
    "AE": "ae.xpt",
    "LB": "lb.xpt",
}


def load_xpt(filepath: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load a single SAS Transport (.xpt) file into a pandas DataFrame.

    Args:
        filepath: Path to the .xpt file.
        encoding: Character encoding for string columns.

    Returns:
        Parsed DataFrame with standardized uppercase column names.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"XPT file not found: {filepath}")
    if not filepath.suffix.lower() == ".xpt":
        raise ValueError(f"Expected .xpt file, got: {filepath.suffix}")

    logger.info(f"Loading XPT file: {filepath}")
    df = pd.read_sas(str(filepath), format="xport", encoding=encoding)

    # Standardize column names to uppercase
    df.columns = [col.upper() for col in df.columns]

    logger.info(f"  Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def load_all_domains(
    data_dir: str,
    domains: Optional[Dict[str, str]] = None,
    encoding: str = "utf-8",
) -> Dict[str, pd.DataFrame]:
    """
    Load all SDTM domain .xpt files from a directory.

    Args:
        data_dir: Directory containing .xpt files.
        domains: Optional dict mapping domain name to filename.
                 Defaults to {"DM": "dm.xpt", "AE": "ae.xpt", "LB": "lb.xpt"}.
        encoding: Character encoding.

    Returns:
        Dictionary mapping domain names to their DataFrames.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    if domains is None:
        domains = DOMAIN_FILES

    loaded = {}
    for domain_name, filename in domains.items():
        filepath = data_dir / filename
        if filepath.exists():
            loaded[domain_name] = load_xpt(str(filepath), encoding=encoding)
        else:
            logger.warning(f"  Domain file not found, skipping: {filepath}")

    if not loaded:
        raise RuntimeError(f"No domain files loaded from {data_dir}")

    logger.info(f"Successfully loaded {len(loaded)} domains: {list(loaded.keys())}")
    return loaded
