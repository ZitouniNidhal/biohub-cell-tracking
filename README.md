# BioHub Cell Tracking
The inference path reads Zarr v3 samples, detects cells with the
CPU-compatible DoG/blob fallback, links consecutive frames with Hungarian
matching, and writes node and edge rows.

## Local submission

Install the package and dependencies, then run:

```bash
pip install -e .
python submission.py --test-dir data/test --output submission.csv
```

The test directory must contain one directory per sample, named `*.zarr`.

## Kaggle notebook

Copy the repository into the notebook or attach it as a model/dataset, then
run:

```python
!pip install -q -r /kaggle/working/biohub-cell-tracking/requirements.txt
%cd /kaggle/working/biohub-cell-tracking
!python scripts/submit_kaggle.py
```

The generated file is `submission.csv` in the notebook working directory.
Submit it using the notebook's Kaggle **Submit** button. The exporter uses
the official voxel scale `(z, y, x) = (1.625, 0.40625, 0.40625)` micrometres.
