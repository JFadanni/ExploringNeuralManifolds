import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from pyFCI import pyFCI
from dadapy import Data

# intrinsic dimension estimator

def ID_twoNN_dadapy(points, twonn=True, scale_dependent=False, threshold=0.5, normalize = False, tolerance = 0.75, **kwargs):
  """
  compute the ID using the TwoNN method implemented in DADApy

  Parameters
  ----------
  points : array
    the data points

  twonn : boolean
    if True, the TwoNN method is used

  scale_dependent : boolean
    if False, the TwoNN method is used
  
  threshold : float
    the threshold for the scale-dependent ID

  Returns
  -------
    id_twoNN : float
      the intrinsic dimension using TwoNN 
    id_twoNN_scale : float
      the intrinsic dimension using TwoNN with scale-dependent ID
    [ids_scaling, rs_scaling, ids_scaling_err] : list
      the intrinsic dimension using scale-dependent ID
  """
  if normalize:
    normalized_points = (points - np.mean(points, axis=0)) / np.std(points, axis=0)
  else:
    normalized_points = points
    
  data = Data(normalized_points)

  # compute the intrinsic dimension
  if twonn:
    id_twoNN,_, rs_twnn = data.compute_id_2NN(algorithm="base")
  if scale_dependent:
    n_min = kwargs.get('n_min', 10)
    algorithm = kwargs.get('algorithm', "base")
  
    ids_scaling, ids_scaling_err, rs_scaling = data.return_id_scaling_2NN(n_min=n_min, algorithm=algorithm)# fix range_max
    plateaus, _, _ = find_plateaus(ids_scaling, min_length=2, tolerance = tolerance)

    if len(plateaus) > 0:
      means = []
      for p in plateaus:
          means.append(np.mean(ids_scaling[p[0]:p[1]]))
      id_twoNN_scale = np.min(means)
    else:
      id_twoNN_scale = ids_scaling[-1]

  if twonn and scale_dependent:
    return id_twoNN, id_twoNN_scale, [ids_scaling, rs_scaling, ids_scaling_err]
  elif twonn and not scale_dependent:
    return id_twoNN, [rs_twnn]
  elif not twonn and scale_dependent:
    return id_twoNN_scale, [ids_scaling, rs_scaling, ids_scaling_err]

def ID_FCI(points, full = False, normalize = True,  **kwargs):
  """
  compute the intrinsic dimension using the FCI method

  Parameters
  ----------
  points : array
    the data points
  
  full : boolean
    if True returns also the fit parameters and the fci integral

  Returns
  -------
    id_fci : float
      the intrinsic dimension using FCI
  """
  if normalize:
    norm_r = pyFCI.center_and_normalize(points)
  else:
    norm_r = points

  fci = pyFCI.FCI(norm_r)
  fit = pyFCI.fit_FCI(fci, threshold = 0.15)
  id_fci = fit[0]

  if full:
    return id_fci, fit, fci
  else:
    return id_fci

def ID_FCI_MC(points, n_iter = 20, full = False, normalize = True, **kwargs):
  """
  compute the intrinsic dimension using the FCI method using the Monte Carlo approach

  Parameters
  ----------
  points : array
    the data points
  
  n_iter : int
    the number of iterations

  full : boolean
    if True returns also the fit parameters and the fci integral

  Returns
  -------
    id_fci : float
      the intrinsic dimension using FCI
  """
  n_samples = kwargs.pop('n_samples', 500)
  samples = kwargs.pop('samples', 500)
  threshold = kwargs.pop('threshold', 0.15)
  if normalize:
    norm_r = pyFCI.center_and_normalize(points)
  else:
    norm_r = points

  fciMC = 0
  for k in range(n_iter):
    fciMC += pyFCI.FCI_MC(norm_r[:,:], n_samples)
  fciMC=fciMC/n_iter

  fit_MC = pyFCI.fit_FCI(fciMC, samples, threshold)
  id_fci = fit_MC[0]

  if full:
    return id_fci, fit_MC, fciMC
  else:
    return id_fci


def ID_PCA(points, normalize = True, full = False):
  """
  compute the intrinsic dimension using the PCA method

  Parameters
  ----------
  points : array
    the data points
  
  normalize : boolean
    if True, the data points are normalized
  
  full : boolean
    if True returns also the PCs and the explained variance

  Returns
  -------
    id_pca : list of int
      the intrinsic dimension using PCA: number of PCs that explain the 90, 95,
      and 99% of the variance
  """

  pca = PCA()
  if normalize:
    Normalized_points = (points - np.mean(points, axis=0)) / np.std(points, axis=0)
  else:
    Normalized_points = points

  y = pca.fit_transform(Normalized_points)
  v = pca.explained_variance_ratio_
  v90 = np.min(np.where(np.cumsum(v)>0.9)[0])
  v95 = np.min(np.where(np.cumsum(v)>0.95)[0])
  v99 = np.min(np.where(np.cumsum(v)>0.99)[0])

  if full:
    return [v90, v95, v99], y, v
  else:
    return [v90, v95, v99]

def participation_ratio(points):
    """
    Calculate the participation ratio from a set of eigenvalues.

    Parameters:
    points (numpy array): the matrix representing the system.

    Returns:
    float: Participation ratio.
    """
    pca = PCA()
    pca.fit(points)
    eigenvalues = pca.explained_variance_
    pr = np.sum(eigenvalues**2)**2 / np.sum(eigenvalues**4)
    return pr


def id_parallel_analysis(data, num_shuffles=250, percentile=95):
    """Compute the participation coefficient from a set of observations.

    Parameters
    ----------
    data : numpy array
        The matrix representing the system. Each row is an observation (e.g.
        a neuron), and each column is a feature (e.g. a sample).

    num_shuffles : int, optional
        The number of times to shuffle the data. Default is 250.

    percentile : int, optional
        The percentile to use to determine the intrinsic dimension. Default is
        95.

    Returns
    -------
    int : The intrinsic dimension of the data.
    """
    def shuffle_data(data_input):
        """
        Create a shuffled version of the original data: (observation x neuron).
        Shuffling is performed independently along each neuron axis.
        
        Parameters:
        data_input (numpy array): Input data with shape (observations, neurons).
        
        Returns:
        numpy array: Shuffled data with the same shape as input.
        """
        shuffled_data = np.zeros_like(data_input)
        for dim in range(data_input.shape[1]):
            shuffled_data[:, dim] = data_input[np.random.permutation(data_input.shape[0]), dim]
        return shuffled_data

    pca = PCA()
    pca.fit(data)
    original_eigs = pca.explained_variance_
    shuffled_eigs = np.zeros((original_eigs.size, num_shuffles))
    for i in range(num_shuffles):
        shuffled_data = shuffle_data(data)
        pca.fit(shuffled_data)
        shuffled_eigs[:, i] = pca.explained_variance_
    shuffled_eigs_percentile = np.percentile(shuffled_eigs, percentile, axis=1)
    dim = np.sum(original_eigs > shuffled_eigs_percentile)

    return dim
