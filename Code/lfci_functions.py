import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import  cdist
from pyFCI import pyFCI
from joblib import Parallel, delayed

def compute_id(coords, method="full", return_gof=False, **kwargs):
    """
    Computes the intrinsic dimension using FCI.

    Parameters
    ----------
    coords : array (n_samples, n_features)
        The data points.
    method : str, optional
        The method used to compute the intrinsic dimension. Default is "full",
        the other option is "montecarlo" or "mc".

    Returns
    -------
    id_fci : float
        The intrinsic dimension using FCI.
    """
    norm_coords = pyFCI.center_and_normalize(coords)
    if method == "full":
        fci = pyFCI.FCI(norm_coords)
    elif method in ("montecarlo", "mc"):
        fci = pyFCI.FCI_MC(norm_coords)
    else:
        raise ValueError("unknown method")

    n_fit_points = kwargs.get("n_fit_points", 500)
    try:
        id_fci, _, gof = pyFCI.fit_FCI(fci, n_fit_points)
    except:
        id_fci = np.nan
        gof = 1
    if return_gof:
        return id_fci, gof
    else:
        return id_fci


def local_fci_distance(dataset, center_indices, n_centers_neighborhood, k_values, fci_method="full"):
    """
    Compute the local intrinsic dimension using FCI for a set of centers with different neighborhood sizes.

    Parameters
    ----------
    dataset : array (n_samples, n_features)
        The data points.
    center_indices : array (n_centers)
        The indices of the centers.
    n_centers_neighborhood : array (n_centers)
        The number of centers to consider for each neighborhood size.
    k_values : array (n_k_values)
        The neighborhood sizes.
    fci_method : str, optional
        The method used to compute the intrinsic dimension. Default is "full",
        the other option is "montecarlo" or "mc".

    Returns
    -------
    local_ids : array (n_centers, n_k_values, 3)
        The local intrinsic dimensions, the corresponding k value and the goodness of fit.
    selected_distances : array (n_centers, max_k+1)
        The distances to the k-th neighbors.
    neighbor_indices : array (n_centers, max_k+1)
        The indices of the k-th neighbors.
    """
    tree = KDTree(dataset)
    local_ids = np.empty((len(center_indices), len(k_values), 3))
    max_k = k_values[-1]
    selected_distances, neighbor_indices = tree.query(dataset[center_indices], max_k + 1, workers=-1)

    for j, (k, n_c_n) in enumerate(zip(k_values, n_centers_neighborhood)):
        print(f"{j = :2d}, {k = :6d}, {n_c_n = :3d}",end="\r")
        for i in range(n_c_n):
            center = center_indices[i]
            neighbors = dataset[neighbor_indices[i][:k]]
            fitted_dimension, gof = compute_id(neighbors, fci_method, return_gof=True)
            local_ids[i, j] = [k, fitted_dimension, gof]

    return local_ids, selected_distances, neighbor_indices

def unwrap_local_ids(local_ids):
    local_ids_2d = local_ids[:,:,1]
    GoF = local_ids[:,:,2]
    k = local_ids[:,:,0]
    return local_ids_2d, GoF, k

def compute_weights(n_centers_for_neighborhoods):
    weights = np.repeat(1/n_centers_for_neighborhoods,n_centers_for_neighborhoods[0]).reshape((len(n_centers_for_neighborhoods),n_centers_for_neighborhoods[0])).T
    return weights

def flatten_variable(variable_2d,n_centers_for_neighborhoods, n_total_centers, **kwargs):
    """
    Flatten a 2d array into a 1d array given the number of centers per neighborhood.

    Parameters
    ----------
    variable_2d : array (n_centers, n_k_values)
        The variable to flatten.
    n_centers_for_neighborhoods : array (n_k_values)
        The number of centers per neighborhood.
    n_total_centers : int
        The total number of centers.

    Returns
    -------
    variable_flat : array (n_total_centers)
        The flattened variable.
    """
    dtype = kwargs.get("dtype", variable_2d.dtype)
    variable_flat = np.zeros(n_total_centers, dtype=dtype)
    ncn_0 = n_centers_for_neighborhoods[0]
    variable_flat[:ncn_0] = variable_2d[:ncn_0,0]
    for i, ncn in enumerate(n_centers_for_neighborhoods[1:]):
        variable_flat[ncn_0:ncn_0+ncn] = variable_2d[:ncn,i+1]
        ncn_0 += ncn
    return variable_flat

def local_fci_distance_fast(dataset, center_indices, n_centers_neighborhood, k_values, fci_method="full", **kwargs):
    """
    Compute the local intrinsic dimension using FCI for a set of centers with different neighborhood sizes using joblib for parallel computation.
    This functions uses parallelization to speed up the computation.

    Parameters
    ----------
    dataset : array (n_samples, n_features)
        The data points.
    center_indices : array (n_centers)
        The indices of the centers.
    n_centers_neighborhood : array (n_centers)
        The number of centers to consider for each neighborhood size.
    k_values : array (n_k_values)
        The neighborhood sizes.
    fci_method : str, optional
        The method used to compute the intrinsic dimension. Default is "full",
        the other option is "montecarlo" or "mc".
    **kwargs : dict
        Additional keyword arguments to be passed to the internal functions.

    Returns
    -------
    local_ids : array (n_centers, n_k_values, 3)
        The local intrinsic dimensions, the corresponding k value and the goodness of fit.
    selected_distances : array (n_centers, max_k+1)
        The distances to the k-th neighbors.
    neighbor_indices : array (n_centers, max_k+1)
        The indices of the k-th neighbors.
    """
    n_jobs = kwargs.get("n_jobs", -1)
    tree = KDTree(dataset)
    local_ids = np.full((len(center_indices), len(k_values), 3),-1,dtype=np.float32)
    max_k = k_values[-1]
    selected_distances, neighbor_indices = tree.query(dataset[center_indices], max_k + 1, workers=n_jobs)

    def compute_center(i, k, n_c_n):
        center = center_indices[i]
        neighbors = dataset[neighbor_indices[i][:k]]
        fitted_dimension, gof = compute_id(neighbors, fci_method, return_gof=True, **kwargs)
        return (k, fitted_dimension, gof)

    for j, (k, n_c_n) in enumerate(zip(k_values, n_centers_neighborhood)):
        print(f"{j = :2d}, {k = :6d}, {n_c_n = :3d}",end="\r")
        local_ids_out = Parallel(n_jobs=n_jobs)(delayed(compute_center)(i, k, n_c_n) for i in range(n_c_n))
        local_ids[:n_c_n,j] = local_ids_out

    return local_ids, selected_distances, neighbor_indices

def KDTree_neighbors_dist(data,n_neighbors=2, workers=4,return_indexes=True):
    """
    Compute the distances to the n_neighbors nearest neighbors.

    Parameters
    ----------
    data : array-like
        The data points for which to compute the nearest neighbor distances.
    n_neighbors : int, optional
        The number of nearest neighbors to consider. Default is 2.
    workers : int, optional
        The number of parallel jobs to run for neighbors search. Default is 4.

    Returns
    -------
    ndarray
        An array containing the distances to the n_neighbors nearest neighbors for each data point.
    """
    kdtree = KDTree(data)
    distances, indexes = kdtree.query(data, k=n_neighbors, workers=workers)
    neighbor_distances = distances[:, 1:]
    if return_indexes:
        return neighbor_distances, indexes
    else:
        return neighbor_distances


def compute_deltas(coordinates, barycentric_coordinates, mean_distance=None, neighbor_threshold=2):
    """
    Compute the ratio of the minimum distance to the mean distance of neighbors.

    Parameters
    ----------
    coordinates : array-like
        Coordinates of the data points.
    barycentric_coordinates : array-like
        Barycentric coordinates used for distance computation.
    mean_distance : float, optional
        Precomputed mean distance. If None, it will be calculated using KDTree neighbors.
    neighbor_threshold : int, optional
        Number of nearest neighbors to consider for mean distance calculation. Default is 2.

    Returns
    -------
    float
        The ratio of the minimum distance from coordinates to barycentric_coordinates to the mean distance of neighbors.

    Notes
    -----
    If `mean_distance` is not provided, the mean distance is computed using the KDTree with the given neighbor_threshold.
    """

    if mean_distance is None:
        neighbor_distances = KDTree_neighbors_dist(coordinates, neighbor_threshold)
        mean_distance = np.mean(neighbor_distances)
    min_distance = np.min(cdist(coordinates, barycentric_coordinates.reshape(1, -1)))
    return min_distance / mean_distance