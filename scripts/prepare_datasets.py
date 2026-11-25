"""Convert the downloaded UCI archives into the CSV files bundled with ImputeAudit.

Run once from the project root:  python scripts/prepare_datasets.py
Every output file is complete (no missing cells); official train/test partitions,
where the dataset donors defined one, are kept in an ``official_split`` column.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "samples"


def prepare_dry_bean() -> pd.DataFrame:
    return pd.read_excel(RAW / "drybean" / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx")


def prepare_pendigits() -> pd.DataFrame:
    columns = [f"{axis}{i}" for i in range(1, 9) for axis in ("x", "y")] + ["digit"]
    train = pd.read_csv(RAW / "pendigits" / "pendigits.tra", header=None, names=columns)
    test = pd.read_csv(RAW / "pendigits" / "pendigits.tes", header=None, names=columns)
    train["official_split"] = "train"
    test["official_split"] = "test"
    return pd.concat([train, test], ignore_index=True)


def prepare_satimage() -> pd.DataFrame:
    columns = [f"px{p}_band{b}" for p in range(1, 10) for b in range(1, 5)] + ["land_cover"]
    train = pd.read_csv(RAW / "satimage" / "sat.trn", sep=r"\s+", header=None, names=columns)
    test = pd.read_csv(RAW / "satimage" / "sat.tst", sep=r"\s+", header=None, names=columns)
    train["official_split"] = "train"
    test["official_split"] = "test"
    return pd.concat([train, test], ignore_index=True)


def prepare_letter() -> pd.DataFrame:
    columns = ["letter", "x_box", "y_box", "width", "high", "onpix", "x_bar", "y_bar", "x2bar",
               "y2bar", "xybar", "x2ybr", "xy2br", "x_ege", "xegvy", "y_ege", "yegvx"]
    frame = pd.read_csv(RAW / "letter" / "letter-recognition.data", header=None, names=columns)
    frame = frame[columns[1:] + ["letter"]]
    frame["official_split"] = ["train"] * 16000 + ["test"] * (len(frame) - 16000)
    return frame


def prepare_htru2() -> pd.DataFrame:
    columns = [
        "profile_mean", "profile_std", "profile_kurtosis", "profile_skewness",
        "dmsnr_mean", "dmsnr_std", "dmsnr_kurtosis", "dmsnr_skewness", "is_pulsar",
    ]
    return pd.read_csv(RAW / "htru2" / "HTRU_2.csv", header=None, names=columns)


BUILDERS = {
    "dry_bean": prepare_dry_bean,
    "pendigits": prepare_pendigits,
    "satimage": prepare_satimage,
    "letter": prepare_letter,
    "htru2": prepare_htru2,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        frame = builder()
        frame.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: {frame.shape[0]} rows x {frame.shape[1]} columns, "
              f"missing cells = {int(frame.isna().sum().sum())}")


if __name__ == "__main__":
    main()
