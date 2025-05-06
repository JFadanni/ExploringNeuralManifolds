import numpy as np
from scipy.spatial import KDTree
import h5py
from pyFCI import pyFCI

from joblib import Parallel, delayed

from task_handler2 import Task

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

    try:
        id_fci, _, gof = pyFCI.fit_FCI(fci)
    except:
        id_fci = np.nan
        gof = 1
    if return_gof:
        return id_fci, gof
    else:
        return id_fci

def local_fci_distance_fast(dataset, center_indices, n_centers_neighborhood, k_values, fci_method="full"):
    # uses a decreasing number of centers at the increasing of neighborhood size

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

def local_fci_distance_ultrafast(dataset, center_indices, n_centers_neighborhood, k_values, fci_method="full", **kwargs):
    # uses a decreasing number of centers at the increasing of neighborhood size

    n_jobs = kwargs.get("n_jobs", -1)
    tree = KDTree(dataset)
    local_ids = np.empty((len(center_indices), len(k_values), 3))
    max_k = k_values[-1]
    selected_distances, neighbor_indices = tree.query(dataset[center_indices], max_k + 1, workers=-1)

    def compute_center(i, k, n_c_n):
        center = center_indices[i]
        neighbors = dataset[neighbor_indices[i][:k]]
        fitted_dimension, gof = compute_id(neighbors, fci_method, return_gof=True)
        return (k, fitted_dimension, gof)

    for j, (k, n_c_n) in enumerate(zip(k_values, n_centers_neighborhood)):
        print(f"{j = :2d}, {k = :6d}, {n_c_n = :3d}",end="\r")
        local_ids_out = Parallel(n_jobs=n_jobs)(delayed(compute_center)(i, k, n_c_n) for i in range(n_c_n))
        local_ids[:n_c_n,j] = local_ids_out

    return local_ids, selected_distances, neighbor_indices
def smooth(x, window_len=20 ):
    Xs = np.zeros_like(x)
    tree = KDTree(x)
    distances, neighbors = tree.query(x, k=window_len, workers=-1)
    for i in range(len(x)):
        rn = np.random.rand(window_len)
        rn = rn/np.sum(rn)
        Xs[i] = np.matmul(rn, x[neighbors[i]])
    return Xs


class TaskExpanded(Task):
    def __init__(self,task_name, train_id="", *args, **kwargs):
        super().__init__(task_name,*args, **kwargs)
        self.train_id = train_id
    
    def set_neighborhoods(self, n_neighborhoods, min_neighborhood_size=50, max_neighborhood_size=10000):
        neighborhoods = np.logspace(np.log2(min_neighborhood_size), np.log2(max_neighborhood_size), n_neighborhoods,base=2,dtype=int)+1
        self.neighborhoods = neighborhoods
        return neighborhoods

    def compute_n_centers_neighborhood(self,l_dataset, n_neighborhoods, coefficient=50):
        neighborhoods = self.neighborhoods
        inverse_fraction = 1/(neighborhoods / l_dataset)
        n_centers_max = int(inverse_fraction.max())
        n_centers_min = int(inverse_fraction.min()*coefficient)
        n_centers_for_neighborhoods = np.floor(np.logspace(np.log2(n_centers_min), np.log2(n_centers_max), n_neighborhoods, base=2)).astype(int)
        n_centers_for_neighborhoods = np.flip(n_centers_for_neighborhoods)
        self.n_centers_for_neighborhoods = n_centers_for_neighborhoods
        self.n_total_centers = np.sum(n_centers_for_neighborhoods)
        return n_centers_for_neighborhoods, n_centers_min, n_centers_max

    def set_centers(self, centers):
        self.centers = centers

    def compute_localFCI(self, dataset, fci_method="full"):
        neighborhoods = self.neighborhoods
        n_centers_for_neighborhoods = self.n_centers_for_neighborhoods
        centers = self.centers
        local_ids, selected_distances, neighbor_indices = local_fci_distance_fast(
            dataset, centers, n_centers_for_neighborhoods, neighborhoods, fci_method=fci_method)

        self.local_ids = local_ids
        self.selected_distances = selected_distances
        self.neighbor_indices = neighbor_indices

        return local_ids, selected_distances, neighbor_indices

    def compute_localFCI_ultrafast(self, dataset, fci_method="full",**kwargs):
        neighborhoods = self.neighborhoods
        n_centers_for_neighborhoods = np.copy(self.n_centers_for_neighborhoods)
        centers = self.centers
        local_ids, selected_distances, neighbor_indices = local_fci_distance_ultrafast(
            dataset, centers, n_centers_for_neighborhoods, neighborhoods, fci_method=fci_method,**kwargs)

        self.local_ids = local_ids
        self.selected_distances = selected_distances
        self.neighbor_indices = neighbor_indices

        return local_ids, selected_distances, neighbor_indices

    def load_localFCI(self, base_folder=".", file_prefix="local_fci"):
        task_name = self.task_name
        train_id = self.train_id
        file_name = f"{file_prefix}_{task_name}_train{train_id}.h5"

        with h5py.File(f"{base_folder}/{file_name}", "r") as f:
            self.local_ids = f["local_ids"][:]
            self.selected_distances = f["selected_distances"][:]
            self.neighbor_indices = f["neighbor_indices"][:]
            self.centers =f["centers"][:]
            self.n_centers_for_neighborhoods = f["n_centers_neigh"][:]
            self.neighborhoods = f["neighborhoods"][:]

        self.n_total_centers = np.sum(self.n_centers_for_neighborhoods)
        self.n_neighborhoods = len(self.neighborhoods)

        return self.local_ids, self.selected_distances, self.neighbor_indices

    def save_localFCI(self, base_folder=".", file_prefix="local_fci"):
        task_name = self.task_name
        train_id = self.train_id
        file_name = f"{file_prefix}_{task_name}_train{train_id}.h5"

        with h5py.File(f"{base_folder}/{file_name}", "w") as f:
            f["local_ids"] = self.local_ids
            f["selected_distances"] = self.selected_distances
            f["neighbor_indices"] = self.neighbor_indices
            f["centers"] = self.centers
            f["n_centers_neigh"] = self.n_centers_for_neighborhoods
            f["neighborhoods"] = self.neighborhoods

    def unwrap_local_ids(self):
        local_ids = self.local_ids
        local_ids_2d = local_ids[:,:,1]
        GoF = local_ids[:,:,2]
        k = local_ids[:,:,0]
        return local_ids_2d, GoF, k

    def compute_weights(self):
        n_centers_for_neighborhoods = self.n_centers_for_neighborhoods
        weights = np.repeat(1/n_centers_for_neighborhoods,n_centers_for_neighborhoods[0]).reshape((len(n_centers_for_neighborhoods),n_centers_for_neighborhoods[0])).T
        self.weights = weights
        return weights

    def flatten_GoF(self):
        GoF = self.local_ids[:,:,2]
        n_centers_for_neighborhoods = self.n_centers_for_neighborhoods
        n_total_centers = self.n_total_centers
        gof_flat = flatten_variable(GoF,n_centers_for_neighborhoods, n_total_centers)
        return gof_flat

    def flatten_id(self):
        local_id_2d = self.local_ids[:,:,1]
        n_centers_for_neighborhoods = self.n_centers_for_neighborhoods
        n_total_centers = self.n_total_centers
        local_id_flat = flatten_variable(local_id_2d,n_centers_for_neighborhoods, n_total_centers)
        return local_id_flat

    def flatten_k(self):
        k_2d = self.local_ids[:,:,0]
        n_centers_for_neighborhoods = self.n_centers_for_neighborhoods
        n_total_centers = self.n_total_centers
        k_flat = flatten_variable(k_2d,n_centers_for_neighborhoods, n_total_centers)
        return k_flat

    def flatten_weights(self):
        weights_2d = self.weights
        n_centers_for_neighborhoods = self.n_centers_for_neighborhoods
        n_total_centers = self.n_total_centers
        weights_flat = flatten_variable(weights_2d,n_centers_for_neighborhoods, n_total_centers)
        return weights_flat

def flatten_variable(variable_2d,n_centers_for_neighborhoods, n_total_centers):
        variable_flat = np.zeros(n_total_centers)
        ncn_0 = n_centers_for_neighborhoods[0]
        variable_flat[:ncn_0] = variable_2d[:ncn_0,0]
        for i, ncn in enumerate(n_centers_for_neighborhoods[1:]):
            variable_flat[ncn_0:ncn_0+ncn] = variable_2d[:ncn,i+1]
            ncn_0 += ncn
        return variable_flat


    