#! /usr/bin/env python3
import os
import errno
import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.spatial import KDTree

from pyFCI import pyFCI

from task_expanded import TaskExpanded, local_fci_distance_fast, smooth

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


    #task_names = ("dmsgo","dmcgo")

    n_neighborhoods = 12
    min_neighborhood_size = 50
    max_neighborhood_size = 10000
    coefficient = 15
    wl = 20
    threshold = 0.02

    prefix = "local_fci"


    for task_name in task_names:
        output_dir = f"ID_results_smartNeigh/{task_name}"
        makedirs(f"{output_folder}/{task_name}")
        network_IDs = np.zeros((n_trains,2))

        for train_idx in range(n_trains):
            train_dir = f"{task_name}/train_{train_idx}"
            train = TaskExpanded(task_name,train_idx, train_dir, train_folder, q)
            stim_start, stim_end, go_start, go_end, len_task = train.find_relevant_points()
            train.split_data()
            angles = train.compute_angle()
            middle = train.middle
            final = train.final
            #train.sample_plots(300)

            if len(middle)>0:
                all_but_beginning = np.concatenate((middle,final), axis=0)
            else:
                all_but_beginning = final
            train_all_but_beginning = train.r[all_but_beginning,:]
            l_dataset = len(all_but_beginning)
            max_neighborhood_size_t = max_neighborhood_size if l_dataset>max_neighborhood_size else l_dataset - 1000
            neighborhoods = train.set_neighborhoods(n_neighborhoods,min_neighborhood_size, max_neighborhood_size_t )
            n_centers_neighborhoods, n_centers_min, n_centers_max = train.compute_n_centers_neighborhood(l_dataset,n_neighborhoods,coefficient)
            centers = rng.choice(l_dataset, n_centers_max, replace=False)
            train.set_centers(centers)

            X = train_all_but_beginning
            local_ids, selected_distances, neighbor_indices = train.compute_localFCI_ultrafast(X, fci_method="full",n_jobs=6)
            weights = train.compute_weights()
            local_ids_flat = train.flatten_id()
            gof_flat = train.flatten_GoF()
            weight_flat = train.flatten_weights()
            network_IDs[train_idx,0] = weighted_percentile(local_ids_flat[gof_flat<threshold], weight_flat[gof_flat<threshold],0.5)

            train.save_localFCI(output_dir,prefix)

            # Smoothing 
            X_smooth = smooth(X, window_len=wl)
            train_smooth = TaskExpanded(task_name,train_idx, train_dir, train_folder, q)
            neighborhoods = train_smooth.set_neighborhoods(n_neighborhoods,min_neighborhood_size, max_neighborhood_size_t )
            n_centers_neighborhoods, n_centers_min, n_centers_max = train_smooth.compute_n_centers_neighborhood(l_dataset,n_neighborhoods,coefficient)
            train_smooth.set_centers(centers)
            local_ids_smooth, selected_distances_smooth, neighbor_indices_smooth = train_smooth.compute_localFCI_ultrafast(X_smooth, fci_method="full",n_jobs=6)
            weight_smooth = train_smooth.compute_weights()
            local_ids_smooth_flat = train_smooth.flatten_id()
            gof_smooth_flat = train_smooth.flatten_GoF()
            weight_smooth_flat = train_smooth.flatten_weights()
            network_IDs[train_idx,1] = weighted_percentile(local_ids_smooth_flat[gof_smooth_flat<threshold], weight_smooth_flat[gof_smooth_flat<threshold],0.5)
            train_smooth.save_localFCI(output_dir,f"smooth_{prefix}")

        #np.savetxt(f"{output_folder}/{task_name}/network_lFCI_ids.dat", network_IDs, fmt='%.3f')
