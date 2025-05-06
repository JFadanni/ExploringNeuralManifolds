#! /usr/bin/env python3
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.spatial import KDTree

from pyFCI import pyFCI

from task_expanded import TaskExpanded
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def weighted_percentile(X,weights, q):
    """
    Calculate the weighted percentile of a given data array.

    Parameters
    ----------
    X : array-like
        The data array from which the percentile is calculated.
    weights : array-like
        The weights associated with each element in the data array.
    q : float
        The percentile to compute, which must be between 0 and 1.

    Returns
    -------
    float
        The calculated weighted percentile value.

    Notes
    -----
    The function first sorts the data and corresponding weights, then computes
    the cumulative distribution function of the weights. The percentile is 
    determined by finding the point where the cumulative distribution exceeds
    the given percentile `q`.
    """

    sorted_idx = np.argsort(X)
    sorted_x = X[sorted_idx]
    weights = weights / np.sum(weights)
    sorted_W = weights[sorted_idx]
    w_cdf = np.cumsum(sorted_W)
    idx_perc = np.min(np.where(w_cdf>q))
    percentile = (sorted_x[idx_perc-1] + sorted_x[idx_perc])/2
    return percentile

def makedirs(path):
    """
    Make a directory and its parents if needed. If the directory already
    exists, do nothing.
    """
    try:
            os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


if __name__ == "__main__":
    seed = 12232
    rng = np.random.default_rng(seed)


    input_folder = "ID_results_smallNeigh"
    train_folder = "../Multiple_trains"
    output_folder = "ID_results_smartNeigh"
    q = 200
    n_trains = 10


    task_names = ("fdgo", "reactgo", "delaygo", 
                  "fdanti", "reactanti", "delayanti", 
                  "dm1", "dm2", "contextdm1", "contextdm2", "multidm", 
                  "delaydm1", "delaydm2", "contextdelaydm1", "contextdelaydm2", "multidelaydm",
                  "dmsgo","dmsnogo", "dmcgo", "dmcnogo")

    #task_names = ["fdgo", "dm1", "contextdm1",  "dmcgo", "dmcnogo"]


    n_neighborhoods = 12
    min_neighborhood_size = 50
    max_neighborhood_size = 10000
    coefficient = 15
    wl = 20
    threshold = 0.02

    prefix =  "local_fci"  # "smooth_local_fci"


    for task_name in task_names:
        input_dir = f"ID_results_smartNeigh/{task_name}"
        makedirs(f"{output_folder}/{task_name}")
        network_IDs = np.zeros((n_trains,3))

        for train_idx in range(n_trains):
            train_dir = f"{task_name}/train_{train_idx}"
            train = TaskExpanded(task_name,train_idx, train_dir, train_folder, q)
            local_ids, selected_distances, neighbor_indices = train.load_localFCI(input_dir,prefix)
            local_ids_2d, GoF, k = train.unwrap_local_ids()
            weights = train.compute_weights()

            gof_flat = train.flatten_GoF()
            local_ids_flat = train.flatten_id()
            weight_flat = train.flatten_weights()
            
            #Percentile
            network_IDs[train_idx,0] = weighted_percentile(local_ids_flat[gof_flat<threshold], weight_flat[gof_flat<threshold],0.5)

            #Histogram
            hist_g, bin_edges_g = np.histogram(local_ids_flat[gof_flat<threshold],range=(0,30), bins=50, weights=weight_flat[gof_flat<threshold])
            est_index = np.argmax(hist_g)
            est = bin_edges_g[est_index] + np.diff(bin_edges_g)[0]/2
            network_IDs[train_idx,1] = est

#            #Gaussian Mixture
#            gmm = GaussianMixture(n_components=2).fit(local_ids_flat[gof_flat<0.02].reshape(-1,1))
#
#            gaussian_centers = gmm.means_
#            network_IDs[train_idx,2] = np.min(gaussian_centers)

            # Kernel density
            kde = KernelDensity(bandwidth=0.3).fit(local_ids_flat[gof_flat<threshold].reshape(-1,1),
                                       sample_weight=weight_flat[gof_flat<threshold])
            
            x_kde = np.linspace(0,30,900)
            log_den = kde.score_samples(x_kde.reshape(-1,1))
            
            peak = x_kde[np.argmax(log_den)]
            network_IDs[train_idx,2] = peak



#            fig, ax = plt.subplots()
#            ax.hist(local_ids_flat[gof_flat<0.02],range=(0,30), bins=50, weights=weight_flat[gof_flat<0.02])
#            ax.axvline(gmm.means_[0][0],ls="--", color="r")
#            ax.axvline(gmm.means_[1][0],ls="--", color="g")
#            ax.axvline(peak,c="purple")
#            ax.plot(x_kde,np.exp(log_den),lw=1.5)

#            ax.set_title(task_name)
#            plt.show()
#            print(task_name, train_idx, gmm.means_, network_IDs[train_idx,:])


        np.savetxt(f"{output_folder}/{task_name}/network_lFCI_ids_new.dat", network_IDs, fmt='%.3f')
