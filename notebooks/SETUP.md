# Notebook Setup Guide

## Prerequisites

Before running the notebooks, ensure you have completed these setup steps:

### 1. Install the Package

From the **project root directory** (not the notebooks folder), run:

```bash
pip install -e .
```

This installs the `biohub_tracking` package in development mode, making all modules available to the notebooks.

### 2. Install Additional Dependencies

```bash
pip install zarr matplotlib numpy scipy pandas
```

### 3. Prepare Data

Place your data in the following structure:

```
biohub-cell-tracking/
├── data/
│   ├── train/
│   │   ├── sample_001.zarr/
│   │   ├── sample_002.zarr/
│   │   └── ...
│   └── test/
│       ├── test_001.zarr/
│       ├── test_002.zarr/
│       └── ...
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_segmentation_training.ipynb
│   ├── 03_tracking_pipeline.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_submission.ipynb
│   └── SETUP.md (this file)
└── ...
```

## Running the Notebooks

### Option 1: Using Jupyter Lab

```bash
jupyter lab notebooks/
```

### Option 2: Using Jupyter Notebook

```bash
jupyter notebook notebooks/
```

### Option 3: Using VS Code

1. Open VS Code
2. Install the Jupyter extension if not already installed
3. Open the notebook file
4. Select Python kernel from the dropdown
5. Run cells with Shift+Enter

## Troubleshooting

### ModuleNotFoundError: No module named 'biohub_tracking'

**Solution:**
1. Ensure you ran `pip install -e .` from the project root
2. Restart the Jupyter kernel (use the kernel menu or Ctrl+Shift+P in VS Code)
3. Re-run the cells

### Path Issues

The notebooks assume they are run from within the `notebooks/` directory. If you get path errors:

1. Make sure you're running notebooks with the correct working directory
2. Check that `../data/` exists relative to the notebooks

### Data Not Found

- Verify your data is in the correct directory structure (see "Prepare Data" above)
- Data should be in Zarr v3 format with `.zarr` file extensions

## Notebook Overview

| Notebook | Purpose | Duration |
|----------|---------|----------|
| 01_data_exploration | Explore dataset structure and visualize frames | ~5-10 min |
| 02_segmentation_training | Demonstrate cell segmentation methods | ~10-15 min |
| 03_tracking_pipeline | Show cell linking and trajectory building | ~10-15 min |
| 04_evaluation | Analyze tracking performance and errors | ~10-15 min |
| 05_submission | Generate final competition submission | ~15-30 min |

## Quick Start

After setup, run the notebooks in order:

```bash
cd notebooks/
jupyter notebook 01_data_exploration.ipynb
```

Each notebook builds on the concepts from previous ones. You can use them as:
- **Learning material** - Understand the BioHub cell tracking pipeline
- **Baseline** - Use as starting point for your competition submission
- **Development** - Experiment with different parameters and methods

## Additional Resources

- See the `docs/` folder for detailed guides:
  - `ARCHITECTURE.md` - System design and module overview
  - `DATA_FORMAT.md` - Data structure details
  - `TRAINING_GUIDE.md` - Training workflows
  - `INFERENCE_GUIDE.md` - Running inference

