import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import  cdist
from pyFCI import pyFCI
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity
from typing import Tuple, Dict, Any, Optional, Callable
import warnings
from dadapy import Data
import matplotlib.pyplot as plt

def compute_local_fci(dataset: np.ndarray,
                      n_neighbors: int = 9,
                      n_centers: int = 9,
                      fci_method: str = "full",
                      n_jobs: int = 1) -> Tuple[np.ndarray, ...]:
    """
    Compute local FCI (lFCI) estimates.
    
    Parameters
    ----------
    dataset : np.ndarray
        Input data of shape (n_samples, n_features)
    n_neighbors : int
        Number of neighborhood sizes to consider
    n_centers : int
        Number of center points to sample
    fci_method : str
        FCI computation method
    n_jobs : int
        Number of parallel jobs
    
    Returns
    -------
    Tuple[np.ndarray, ...]
        ids : ID estimates
        deltas : delta values
        k : neighborhood sizes
        gof : GoF values
    """
    n_samples = len(dataset)
    
    # Define neighborhood sizes (logarithmic spacing)
    neighborhoods = np.logspace(np.log(n_neighbors), np.log(n_samples), 
                               n_neighbors + 1, base=np.e, dtype=int)[:-1]
    
    # Tile n_centers for compatibility with local_fci_distance_fast
    n_centers_tiled = np.tile(n_centers, n_neighbors)
    
    # Randomly select centers
    centers = np.random.choice(n_samples, n_centers, replace=True)
    
    # Compute local FCI
    ids, distances, neighbor_indices = local_fci_distance_fast(
        dataset, centers, n_centers_tiled, neighborhoods,
        fci_method=fci_method, n_jobs=n_jobs
    )
    """
    ids : np.ndarray
        Array of shape (n_centers, n_neighbors, 3) containing:
        [:, :, 0] : k values (neighborhood sizes)
        [:, :, 1] : ID estimates
        [:, :, 2] : Goodness of Fit (GoF) values
    
    """
    ids_2d = ids[:, :, 1].flatten()
    GoF = ids[:, :, 2].flatten()
    k = ids[:, :, 0].flatten()

    """
        ids_2d : ID estimates (n_centers, n_neighbors)
        GoF : Goodness of Fit values (n_centers, n_neighbors)
        k : Neighborhood sizes (n_centers, n_neighbors)
    """

    # Compute delta values
    deltas = compute_deltas_tree(dataset, n_centers, neighborhoods, neighbor_indices).flatten()

    return ids_2d, deltas, k, GoF, neighborhoods


def extract_lfci_estimate(all_ids: np.ndarray,
                         all_deltas: np.ndarray,
                         all_k: np.ndarray,
                         all_gof: np.ndarray,
                         neighborhoods: np.ndarray,
                         delta_thr: float = 2.0,
                         ci_interval: Tuple[float, float] = (10, 90),
                         KDE_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract ID estimate from lFCI distribution using quality filters.
    
    Parameters
    ----------
    all_ids : np.ndarray
        Flattened ID estimates
    all_deltas : np.ndarray
        Flattened delta values
    all_k : np.ndarray
        Flattened neighborhood sizes
    all_gof : np.ndarray
        Flattened GoF values
    neighborhoods : np.ndarray
        Array of neighborhood sizes used
    delta_thr : float
        Threshold for delta values (filter out high deltas)
    ci_interval : Tuple[float, float]
        Percentile interval for confidence interval
    #bandwidth : float
    #    Bandwidth for KDE peak detection
    KDE_params : Optional[Dict[str, Any]]
        Dictionary of parameters for KDE peak detection
        - 'bandwidth': Bandwidth for KDE (default: 0.3)
        - 'x_min': Minimum value for KDE (default: 0)
        - 'x_max': Maximum value for KDE (default: 10)
        - 'n_points_peak': Number of points for KDE peak detection (default: 300)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'gof_thr': Goodness of Fit threshold
        - 'id_peak': Peak ID estimate from KDE
        - 'ci_id': Confidence interval
        - 'filtered_data': Indices of data points that passed filters
    """
    # Determine GoF threshold from percentiles for each neighborhood size
    percentiles_gof = np.zeros(len(neighborhoods))
    ks = np.zeros(len(neighborhoods))
    for i, k in enumerate(neighborhoods):
        idx_k = np.where(all_k == k)[0]
        if len(idx_k) > 0:
            percentiles_gof[i] = np.percentile(all_gof[idx_k], 99)
            ks[i] = k
        else:
            percentiles_gof[i] = np.inf
            ks[i] = k

    gof_thr = np.min(percentiles_gof)
    opt_k = ks[np.argmin(percentiles_gof)]
    print("   GoF threshold determined:", gof_thr, "as 99th percentile at k =", opt_k)
    
    # Apply filters: delta < threshold AND GoF < threshold
    cond = (all_deltas < delta_thr) & (all_gof < gof_thr)
    idx_good = np.where(cond)[0]
    
    # Extract filtered lFCI values
    lfci_good = all_ids[idx_good]

    # Compute peak estimate and confidence interval
    if len(lfci_good) > 0:
        #print(id_peak, density, x_range)
        id_peak, density, x_range = compute_peak_lfci(lfci_good, bandwidth=KDE_params['bandwidth'], x_min=KDE_params["x_min"],
                                    x_max=KDE_params["x_max"], n_points_peak=KDE_params["n_points_peak"])
        ci_id = np.percentile(lfci_good, ci_interval)
    else:
        id_peak = np.nan
        ci_id = [np.nan, np.nan]
        density = np.nan
        x_range = np.nan
        warnings.warn("No data points passed the filters")
    
    return {
        'gof_thr': gof_thr,
        'opt_k': opt_k, #optimal k at which GoF threshold was found
        'id_peak': id_peak,
        'ci_id': ci_id,
        'filtered_indices': idx_good,
        'n_filtered': len(idx_good),
        'kde_density': density,
        'kde_x_range': x_range, 
    }


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


def compute_deltas_tree(dataset, n_centers, neighborhoods,neighbors_indices):
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

def exponential_embedding(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply exponential embedding to data.
    
    Parameters
    ----------
    x : np.ndarray
        Input data of shape (n_samples, n_features)
    alpha : float
        Embedding strength parameter
        
    Returns
    -------
    np.ndarray
        Embedded data
    """
    return (np.exp(alpha * x) - 1) / (np.exp(alpha) - 1)



class DataGenerator:
    """
    Base class for data generators.
    Override generate() method to create custom datasets.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize data generator.
        
        Parameters
        ----------
        random_seed : int
            Random seed for reproducibility
        """
        self.rng = np.random.default_rng(random_seed)
        self.config = {}
    
    def generate(self, n_samples: int, **kwargs) -> np.ndarray:
        """
        Generate dataset (to be overridden by subclasses).
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        
        Returns
        -------
        np.ndarray
            Generated data of shape (n_samples, n_features)
        """
        raise NotImplementedError("Subclasses must implement generate() method")


class HypercubeGenerator(DataGenerator):
    """
    Generate hypercube data with optional exponential embedding.
    This replicates the original functionality as an example.
    """
    
    def __init__(self, intrinsic_dim: int, ambient_dim: int, 
                 alpha: float = 0.0, random_seed: int = 42):
        """
        Initialize hypercube generator.
        
        Parameters
        ----------
        intrinsic_dim : int
            True intrinsic dimension
        ambient_dim : int
            Ambient/embedding dimension
        alpha : float
            Exponential embedding parameter (0 = no embedding)
        random_seed : int
            Random seed for reproducibility
        """
        super().__init__(random_seed)
        self.intrinsic_dim = intrinsic_dim
        self.ambient_dim = ambient_dim
        self.alpha = alpha
        self.config = {
            'intrinsic_dim': intrinsic_dim,
            'ambient_dim': ambient_dim,
            'alpha': alpha
        }
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate hypercube data.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        
        Returns
        -------
        np.ndarray
            Generated data
        """
        # Generate uniform data in intrinsic dimensions
        data = np.zeros((n_samples, self.ambient_dim))
        data[:, :self.intrinsic_dim] = self.rng.uniform(0, 100, (n_samples, self.intrinsic_dim))
        
        # Add small Gaussian noise in extra dimensions
        if self.ambient_dim > self.intrinsic_dim:
            data[:, self.intrinsic_dim:] = self.rng.normal(1, 0.01, 
                                                         (n_samples, self.ambient_dim - self.intrinsic_dim))
        
        # Normalize
        data = data / data.max()
        
        # Apply exponential embedding if alpha > 0
        if self.alpha > 0:
            data = exponential_embedding(data, self.alpha)
        
        return data
    

def compute_peak_lfci(lfci_values: np.ndarray, bandwidth: float = 0.3, x_min: float = 0, x_max: float = 10, n_points_peak: int = 1000 ) -> float:
    """
    Compute the peak of the kernel density estimate of lFCI distribution.
    
    Parameters
    ----------
    lfci_values : np.ndarray
        Array of lFCI values
    bandwidth : float
        Bandwidth for kernel density estimation
        
    Returns
    -------
    float
        Peak value of the KDE
    """
    kde = KernelDensity(bandwidth=bandwidth).fit(lfci_values.reshape(-1, 1))
    x_range = np.linspace(x_min, x_max, n_points_peak)
    log_density = kde.score_samples(x_range.reshape(-1, 1))
    density = np.exp(log_density)
    peak_index = np.argmax(log_density)
    return x_range[peak_index], density, x_range

def compute_global_ids(dataset: np.ndarray, 
                       methods: Tuple[str, ...] = ("2NN", "FCI")) -> Dict[str, float]:
    """
    Compute global intrinsic dimension estimates: "2NN" and/or "FCI".
    
    Parameters
    ----------
    dataset : np.ndarray
        Input data of shape (n_samples, n_features)
    methods : Tuple[str, ...]
        Methods to use: "2NN", "FCI", or both
    
    Returns
    -------
    Dict[str, float]
        Dictionary of method -> ID estimate
    """
    results = {}
    
    # Remove identical points (important for 2NN)
    data_obj = Data(dataset)
    data_obj.remove_identical_points()
    
    # Two Nearest Neighbors method
    if "2NN" in methods:
        try:
            id_2nn, _, _ = data_obj.compute_id_2NN(algorithm="base")
            results["2NN"] = float(id_2nn)
        except Exception as e:
            warnings.warn(f"2NN failed: {e}")
            results["2NN"] = np.nan
    
    # FCI method
    if "FCI" in methods:
        try:
            id_fci = compute_id(dataset, method="full")
            results["FCI"] = float(id_fci)
        except Exception as e:
            warnings.warn(f"FCI failed: {e}")
            results["FCI"] = np.nan
    
    return results


def plot_id_results_comparison(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Create visualization of ID estimation results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from run_id_pipeline
    save_path : Optional[str]
        Path to save the figure (if None, display only)
    """
    # Create 2x2 grid but we'll use positions:
    # (0,0): Filtered vs unfiltered comparison 
    # (0,1): GoF vs Delta scatter  
    # (1,0): Summary bar chart 
    # (1,1): Violin plot 
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    data_name = results['data_name']
    global_ids = results['global_ids']
    lfci_raw = results['lfci_raw']
    lfci_processed = results['lfci_processed']
    all_deltas = lfci_raw['deltas']
    all_gof = lfci_raw['gof']
    all_id = lfci_raw['ids']
    all_k = lfci_raw['k']
    gof_thr = lfci_processed['gof_thr']
    est_id = lfci_processed['id_peak']
    density = lfci_processed['kde_density']
    x_range = lfci_processed['kde_x_range']
    optimal_k = lfci_processed['opt_k']

    
    # --- PLOT 1: Filtered vs unfiltered comparison (now in position 0,0) ---
    ax = axes[0, 0]
    raw_data = lfci_raw['ids']
    filtered_idx = lfci_processed['filtered_indices']
    filtered_data = raw_data[filtered_idx] if len(filtered_idx) > 0 else np.array([])
    
    if len(filtered_data) > 0:
        #bins = np.linspace(min(raw_data.min(), filtered_data.min()), 
        #              max(raw_data.max(), filtered_data.max()), 30)
        #ax.hist(raw_data, bins=bins, alpha=0.5, density = True, label='All estimates', color='blue')
        bins = np.linspace(filtered_data.min(), filtered_data.max(), 30)
        ax.hist(filtered_data, bins=bins, alpha=0.7, density = True, label='Filtered estimates', color='orange')
        # Plot
        ax.plot(x_range, density, linewidth=2, color='blue', label='KDE')
        ax.fill_between(x_range.flatten(), density, alpha=0.3, color='blue')
        ax.set_xlim(filtered_data.min(), filtered_data.max())
    else:
        Warning("No filtered data to plot in comparison histogram.")
        ax.hist(raw_data, bins=50, alpha=0.7, label='All estimates', color='blue')
    ax.axvline(est_id, color='red', 
          linestyle='--', linewidth=2, label=f'Peak: {est_id:.2f}')
    ax.set_xlabel('ID Estimate', fontweight='bold')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Effect of Quality Filtering', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # --- PLOT 2: Scatter plot: GoF vs Delta (now in position 0,1) ---
    ax = axes[0, 1]
    
    # Apply filtering criteria
    delta_thr = results['parameters'].get('delta_thr', 2.0)
    cond = (all_deltas < delta_thr) & (all_gof < gof_thr)
    idx_good = np.where(cond)[0]
    
    if len(idx_good) > 0:
        # Create mask for filtered points
        all_indices = np.arange(len(all_deltas))
        good_mask = np.isin(all_indices, idx_good)
        
        # Plot rejected points
        scatter_rejected = ax.scatter(
            all_deltas[~good_mask], 
            all_gof[~good_mask], 
            c=all_id[~good_mask], 
            cmap='viridis', 
            alpha=0.15,
            s=8,
            edgecolors='none',
            label='Rejected',
            zorder=1
        )
        
        # Plot accepted points
        scatter_accepted = ax.scatter(
            all_deltas[good_mask], 
            all_gof[good_mask], 
            c=all_id[good_mask], 
            cmap='viridis', 
            alpha=0.8,
            s=30,
            edgecolors='black',
            linewidths=0.8,
            label='Accepted',
            zorder=2
        )
    else:
        # Plot all points if no filtering
        scatter_all = ax.scatter(
            all_deltas, 
            all_gof, 
            c=all_id, 
            cmap='viridis', 
            alpha=0.6,
            s=20,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Add threshold lines
    ax.axhline(gof_thr, color='red', linestyle='--', 
              linewidth=2, alpha=0.8, 
              label=f'GoF thr: {gof_thr:.2f}',
              zorder=3)
    
    ax.axvline(delta_thr, color='red', linestyle=':', 
              linewidth=2, alpha=0.8,
              label=f'Δ thr: {delta_thr}',
              zorder=3)
    
    # Add shaded region
    ax.fill_between([0, delta_thr], 0, gof_thr,
                   color='green', alpha=0.15, 
                   label='Good region',
                   zorder=0)
    
    ax.set_xlabel(r'{$\delta} (Delta)', fontweight='bold')
    ax.set_ylabel('GoF (Goodness of Fit)', fontweight='bold')
    ax.set_title('Quality Metrics Scatter Plot', fontweight='bold')
    
    if len(idx_good) > 0:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='gray', markersize=8, 
                      alpha=0.3, label='Rejected'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='blue', markersize=10,
                      markeredgecolor='black', label='Accepted'),
            plt.Line2D([0], [0], color='red', linestyle='--', 
                      linewidth=2, label=f'GoF thr: {gof_thr:.2f}'),
            plt.Line2D([0], [0], color='red', linestyle=':', 
                      linewidth=2, label=f'Δ thr: {delta_thr}')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    else:
        ax.legend(loc='upper right', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    if len(idx_good) > 0:
        plt.colorbar(scatter_accepted, ax=ax, label='ID Estimate')
    else:
        plt.colorbar(scatter_all, ax=ax, label='ID Estimate')
    
    # --- PLOT 3: Summary bar chart (now in position 1,0) ---
    ax = axes[1, 0]
    methods = list(global_ids.keys()) + ['lFCI']
    values = list(global_ids.values()) + [lfci_processed['id_peak']]
    
    bars = ax.bar(methods, values, alpha=0.7)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.2f}', ha='center', va='bottom')
    
    ax.set_ylabel('ID Estimate', fontweight='bold')
    ax.set_title('Comparison of Methods', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # --- PLOT 4: Violin plot from data_visualisation.py (now in position 1,1) ---
    ax = axes[1, 1]
    
    # Prepare data for violin plot (matching data_visualisation.py logic)
    k_flat = all_k.flatten()
    gof_flat = all_gof.flatten()
    
    # Get unique k values
    unique_k = np.unique(k_flat)
    k_values = unique_k
    
    # Group GoF values by k value
    all_gof_by_k = []
    for k in k_values:
        mask = (k_flat == k)
        gof_for_k = gof_flat[mask]
        all_gof_by_k.append(gof_for_k)
    
    # Find maximum length for violin plot
    max_len = max(len(arr) for arr in all_gof_by_k if len(arr) > 0)
    
    # Create 2D array with NaN padding
    violin_data = np.full((max_len, len(k_values)), np.nan)
    for i, k in enumerate(k_values):
        gof_arr = all_gof_by_k[i]
        if len(gof_arr) > 0:
            violin_data[:len(gof_arr), i] = gof_arr
    
    # Remove columns with no data
    valid_columns = [i for i in range(len(k_values)) if not np.all(np.isnan(violin_data[:, i]))]
    violin_data_valid = violin_data[:, valid_columns]
    k_values_valid = k_values[valid_columns]
    
    # Create violin plot if we have valid data
    if len(k_values_valid) > 0:
        positions = np.arange(1, len(k_values_valid) + 1)
        
        # Create main violin plot
        parts0 = ax.violinplot(violin_data_valid, showextrema=False, widths=0.7)
        
        # Style all violins with blue color
        for pc in parts0['bodies']:
            pc.set_facecolor("C0")
            pc.set_edgecolor("C0")
            pc.set_alpha(0.8)
        
            
        # Highlight optimal k violin if it's in valid k values
        if optimal_k in k_values_valid:
            opt_k_idx = np.where(k_values_valid == optimal_k)[0][0]
            parts = ax.violinplot(
                violin_data_valid[:, opt_k_idx], 
                positions=[positions[opt_k_idx]],
                showextrema=False, 
                widths=1
            )
            
            # Style the highlighted violin with red color
            for pc in parts['bodies']:
                pc.set_facecolor('r')
                pc.set_edgecolor('r')
                pc.set_alpha(1)
        
        # Set x-axis ticks and labels
        ax.set_xticks(positions)
        ax.set_xticklabels(k_values_valid.astype(int), rotation=90)
    
    # Add horizontal dashed line at GoF threshold
    ax.axhline(gof_thr, color='k', linestyle='dashed', linewidth=2, label=f'GoF threshold: {gof_thr:.3f}')
    
    # Add shaded region
    ax.axhspan(0, gof_thr, alpha=0.1, color='green', label='Good fit region')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Label axes
    ax.set_xlabel("k (Number of Neighbors)", fontweight='bold')
    ax.set_ylabel("GoF (Goodness of Fit)", fontweight='bold')
    ax.set_title("GoF Distribution vs k", fontweight='bold')
    ax.legend(loc='upper right')
    
    # Adjust layout and save
    plt.suptitle(f'Intrinsic Dimension Estimation: {data_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

