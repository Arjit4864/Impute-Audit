# Datasets

`samples/` holds the five complete benchmark datasets used in the study, converted to CSV by
`scripts/prepare_datasets.py`. All five are published by the UCI Machine Learning Repository under the
Creative Commons Attribution 4.0 International licence (CC BY 4.0). The CSV files are unchanged apart
from column naming and, where the donors defined a train/test partition, an added `official_split` column.

| File | Dataset | Rows | Features | Classes | Split | Source |
|---|---|---|---|---|---|---|
| `dry_bean.csv` | Dry Bean | 13,611 | 16 | 7 | stratified 80/20 | Koklu and Ozkan (2020), https://doi.org/10.24432/C50S4B |
| `pendigits.csv` | Pen-Based Recognition of Handwritten Digits | 10,992 | 16 | 10 | official, writer-independent (7,494 / 3,498) | Alpaydin and Alimoglu (1998), https://doi.org/10.24432/C5MG6K |
| `satimage.csv` | Statlog (Landsat Satellite) | 6,435 | 36 | 6 | official (4,435 / 2,000) | Srinivasan (1993), https://doi.org/10.24432/C55887 |
| `letter.csv` | Letter Recognition | 20,000 | 16 | 26 | official (16,000 / 4,000) | Slate (1991), https://doi.org/10.24432/C5ZP40 |
| `htru2.csv` | HTRU2 | 17,898 | 8 | 2 | stratified 80/20 | Lyon et al. (2016), https://doi.org/10.24432/C5DK6R |

To rebuild the files from the original archives:

```bash
python scripts/download_datasets.py
```
