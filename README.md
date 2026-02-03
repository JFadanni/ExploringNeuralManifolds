# Notebook Documentation: Intrinsic Dimension Estimation with lFCI

## Overview
This documentation describes two Jupyter notebooks designed for **Intrinsic Dimension (ID) estimation** using the **Local Full Correlation Integral (lFCI)** method, as proposed in the paper *"Exploring neural manifolds across a wide range of intrinsic dimensions."* The notebooks are modular, user-friendly, and can be applied to custom datasets.

## Requirements
- Python 3.10+
- Libraries: `numpy`, `matplotlib`, `h5py`, `scipy`, `scikit-learn`, 'pyFCI' 
- Custom module: `lfci_functions.py`

## ID estimation workflow
1. **Data Loading** → Use dataset from the original reference or load custom data.
2. **Parameter Setup** → Adjust `lfci_params` based on data size and quality requirements.
3. **Estimation** → Run `compute_local_fci()` to obtain local ID estimates.
4. **Filtering** → Apply quality thresholds (`gof_thr`, `delta_thr`) to extract the final estimate.
5. **Visualization** → Use built-in plotting functions to inspect results.
   
## 1. ID_pipeline 
This folder contains a notebook (`ID_estimation_simple.ipynb`), which implements a **modular pipeline** for estimating the intrinsic dimension of **any custom dataset** using the lFCI method.

### Features:
1. **Imports and Configuration**  
   - Imports necessary libraries and loads custom functions from `lfci_functions.py`.  

2. **Pipeline for Custom Data**  
   - Step-by-step process:
     1. Load your dataset (replace placeholder with real data).
     2. Set lFCI parameters (neighbors, centers, thresholds, etc.).
     3. Compute local FCI estimates.
     4. Visualize raw results (histograms and delta distributions).
     5. Apply quality filtering and extract final ID estimate.
   - The notebook with placeholder data and small statistics (small parameters) runs in less than one minute on a standard computer.
   - The pipeline is **dataset-agnostic** and can be applied to neural data, images, embeddings, or any high-dimensional dataset.

3. **Plotting Function**  
   - `plot_id_results_simple()`: Visualizes ID estimation results with three subplots:
     - Filtered vs. unfiltered ID estimates
     - Scatter plot: GoF vs. Delta (quality metrics)
     - Violin plot: GoF distribution across different k-values

4. **Tips for Parameter Tuning**  
   - `n_neighbors`: Use logarithmic spacing for small to large neighborhoods.
   - `n_centers`: More centers improve statistics but increase computation.
   - `delta_thr`: For isotropic and flat data, the 99th percentile may be < 2 (see paper).
   - `n_jobs`: Set to -1 for parallel processing.

## 2. demo 
This folder contains notebook (`ID_estimation_simple.ipynb`), which is a **demonstration** of how to replicate results from the original paper using **smaller parameters** for user-friendly computation.

### Features:
- Follows the same pipeline as in the previous sections.
- Uses **reduced parameters** compared to the paper for faster execution.
- With sufficient computer power, allows replication of results in the original reference. Complete parameters are provided in the paper's supplementary material for full reproducibility.


For detailed methodology and theoretical background, refer to:  
*"Exploring neural manifolds across a wide range of intrinsic dimensions."*
