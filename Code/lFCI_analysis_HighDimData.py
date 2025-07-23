#! /usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

from lfci_functions import *
from utils import makedirs

def compute_all_deltas(dataset, n_centers, neighborhoods,neighbors_indices):
    """
    Compute the deltas for local FCI analysis by calculating the ratio of the minimum distance 
    from data points to their barycentric coordinate to the mean distance of neighbors.

    Parameters
    ----------
    dataset : array-like
        The data array containing the observations.
    n_centers : int
        The number of centers to be considered.
    neighborhoods : array-like
        An array containing the sizes of neighborhoods for each center.
    neighbors_indices : array-like
        An array of indices indicating the neighbors for each center.

    Returns
    -------
    deltas : ndarray
        An array of shape (n_centers, n_neighborhoods) containing the computed deltas for each center and neighborhood size.
    """

    n_neighborhoods = len(neighborhoods)
    deltas = np.zeros((n_centers,n_neighborhoods))
    distances,_ = KDTree_neighbors_dist(dataset)
    for i in range(n_centers):
        for j,k in enumerate(neighborhoods):
            sample = dataset[neighbors_indices[i][:k]]
            bar_sample = sample.mean(axis=0)
            distances_selected = distances[neighbors_indices[i][:k]]
            mean_distances = distances_selected.mean()
            deltas[i,j] = compute_deltas(sample,bar_sample, mean_distances)
    return deltas

def compute_lfci_high_dim_data(**kwargs):

    """
    Compute local FCI for high-dimensional data with different embeddings and neighborhood sizes.

    Parameters
    ----------
    analysis_folder : str, optional
        The folder where the results of the analysis are stored. Defaults to "HighDimData".
    ncents : int, optional
        The number of centers to be considered. Defaults to 100.
    delta_thr : float, optional
        The threshold above which the local FCI values are considered significant. Defaults to 3.
    gof_thr : float, optional
        The threshold below which the local FCI values are considered significant. Defaults to 0.02.
    embeddings : tuple of str, optional
        The embeddings to be considered. Defaults to ("exp", "lin").
    dims : tuple of int, optional
        The dimensionality of the data to be considered. Defaults to (3, 6, 10, 20, 40).

    Notes
    -----
    The results of the local FCI analysis are stored in a folder named `analysis_folder/ID_results`.
    The results are stored in HDF5 files named `lfci_<dim>d_<embedding>.h5`.
    """

    data_folder = "Manuscript_Data_and_Code/fig4_data"
    analysis_folder =   kwargs.get("analysis_folder","HighDimData")
    fci_folder = f"{analysis_folder}/ID_results"

    makedirs(fci_folder)


    with h5py.File(f"{data_folder}/data_effect_of_changing_true_d.h5", "r") as f: 
        X_40d_exp = f["X_40d_seed0_100dB_exp"][:]
        X_20d_exp = f["X_20d_seed0_100dB_exp"][:]
        X_10d_exp = f["X_10d_seed0_100dB_exp"][:]
        X_6d_exp = f["X_6d_seed0_100dB_exp"][:]
        X_3d_exp = f["X_3d_seed0_100dB_exp"][:]
        X_40d_lin = f["X_40d_seed0_100dB_lin"][:]
        X_20d_lin = f["X_20d_seed0_100dB_lin"][:]
        X_10d_lin = f["X_10d_seed0_100dB_lin"][:]
        X_6d_lin = f["X_6d_seed0_100dB_lin"][:]
        X_3d_lin = f["X_3d_seed0_100dB_lin"][:]


    neighborhoods = np.arange(20, 12000, 900)
    ncents = kwargs.get("ncents", 100)
    delta_thr = kwargs.get("delta_thr", 3)
    gof_thr = kwargs.get("gof_thr", 0.02)

    n_centers = np.tile(ncents, len(neighborhoods))
    embeddings = kwargs.get("embeddings", ("exp", "lin"))
    if type(embeddings) == str:
        embeddings = (embeddings)
    

    all_dims = (3, 6, 10, 20, 40)

    dims = kwargs.get("dims", all_dims)
    if type(dims) == int:
        dims = (dims, )

    for embedding in embeddings:
        for d in dims:
            data = eval(f"X_{d}d_{embedding}")
            indexes_centers = np.random.choice(data.shape[0], ncents, replace=False)
            ID_FCI, GoF = compute_id(data, return_gof=True)
            np.save(f"{fci_folder}/ID_FCI_{d}d_{embedding}", np.array([ID_FCI, GoF]))

            ids_lfci, distances, indices_neighbors = local_fci_distance_fast(data, indexes_centers, n_centers, neighborhoods, fci_method="full")

            deltas = compute_all_deltas(data, ncents, neighborhoods, indices_neighbors)

            file_name = f"lfci_{d}d_{embedding}.h5"
            with h5py.File(f"{fci_folder}/{file_name}", "w") as f:
                f["local_ids"] = ids_lfci
                f["selected_distances"] = distances
                f["neighbor_indices"] = indices_neighbors
                f["centers"] = indexes_centers
                f["neighborhoods"] = neighborhoods
                f["deltas"] = deltas

if __name__ == "__main__":
    compute_lfci_high_dim_data()