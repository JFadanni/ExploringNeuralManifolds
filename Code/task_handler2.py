import numpy as np
import matplotlib.pyplot as plt
import h5py

from ID_estimator import *
from network_structures import BPTT

import tensorflow as tf

import sys
sys.path.append('../multitask/')
from task import generate_trials, rule_name
from network import Model
import tools

from pyFCI import pyFCI

class Task:

    # Class to handle all operations each task needs

    def __init__(self, task_name, task_dir, basedir=".", batch_size=200, verbose=False, **kwargs):

        # initialize all the matrices
        initial_time = kwargs.get('initial_time', None)
        self.task_name = task_name
        self.basedir = basedir
        self.task_dir = task_dir
        self.q = batch_size
        with h5py.File(f"{basedir}/{task_dir}/RNN_params.h5", "r") as f:
            self.Win  = f["w_in"][:]
            self.Wrec = f["w_rec"][:]
            self.Wout = f["w_out"][:]
            self.brec = f["brec"][:]
            self.bout = f["bout"][:]
        
        with h5py.File(f"{basedir}/{task_dir}/network_activity.h5", "r") as f:
            self.r = f["r"][:]
            self.z = f["z"][:]
            self.x_train = f["x_train"][:]
            self.y_train = f["y_train"][:]

        if verbose:
            print(f"Task {task_name} loaded.")
            print(f"Number of neurons: {np.size(self.Win.T[0])}")

    def network(self):

        # compute the back-propagation through time on the given task
        
        r, z = BPTT(self.Win.T, self.Wrec.T, self.Wout.T, self.brec, self.bout, self.x_train)

        self.r = r
        self.z = z

        return r, z # return recurrent and output part
    def network_yang(self, task_name=None, basedir=None,batch_size=200, **kwargs):
        basedir = self.basedir if basedir is None else basedir
        task_name = self.task_name if task_name is None else task_name
        model_dir = basedir+"/"+ task_name
        rule = task_name

        model = Model(model_dir)
        hp = model.hp
        
        initial_time = kwargs.get('initial_time', None)

        with tf.compat.v1.Session() as sess:
            model.restore()

        #trial = generate_trials(rule, hp, mode='test',batch_size = 1)
            if initial_time:
                trial = generate_trials(rule, hp, mode='random',batch_size = batch_size, initial_time = initial_time)
            else:
                trial = generate_trials(rule, hp, mode='random',batch_size = batch_size)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)
        # All matrices have shape (n_time, n_condition, n_neuron)
        
#            print(np.shape(trial.x), np.shape(h), np.shape(y_hat))

            if initial_time:    
                h = h[initial_time:,:,:]
                y_hat = y_hat[initial_time:,:,:]
            var_list = model.var_list

        # evaluate the parameters after training
            params = [sess.run(var) for var in var_list]
        # get name of each variable
            names  = [var.name for var in var_list]

            y_trained = np.vstack(np.swapaxes(y_hat, 0, 1))
            h_trained = np.vstack(np.swapaxes(h, 0, 1))

            self.r = h_trained
            self.z = y_trained

        return h_trained, y_trained

    def find_relevant_points(self):
    # find relevant points for the task
        x_train = self.x_train
        y_train = self.y_train
        trials_len = x_train.shape[0]//self.q
        self.trials_len = trials_len
        stim = np.where(x_train[:trials_len,1:65].max(1)>0.3)[0]
        stim_start = stim[0] if len(stim)>0 else 0
        stim_end = stim[-1] if len(stim)>0 else 0
        go = np.where(y_train[:trials_len,0]<0.8)[0]
        go_start = go[0] if len(go)>0 else 0
        go_end = go[-1] if len(go)>0 else 0
        self.stim_start = stim_start
        self.stim_end = stim_end
        self.go_start = go_start
        self.go_end = go_end
        return stim_start, stim_end, go_start, go_end, trials_len

    def compute_angle(self):
        # decode target angle from output
        # z: output part
        # n: trial length
        angle = np.zeros(self.q)
        z = self.z
        n = self.trials_len
        q = self.q
        for i in range(q):
            f = z[n*(i+1)-10:n*(i+1),1:]
            fm = np.mean(f,axis=0)
            angle[i] = np.argmax(fm)
        self.angle = angle
        return angle

    def split_data(self):
        q = self.q
        n = self.trials_len
        n1 = self.stim_start
        n2 = self.go_start

        beginning = np.repeat(np.arange(0, n*q, n), n1) + np.tile(np.arange(0, n1), q)
        if n2>n1:
            middle = np.repeat(np.arange(0, n*q, n), n2-n1) + np.tile(np.arange(n1, n2), q)
        else:
            middle = np.array([])
        final = np.repeat(np.arange(0, n*q, n), n-n2) + np.tile(np.arange(n2, n), q)

        self.beginning = beginning
        self.middle = middle
        self.final = final
    
    def connectivity_matrix(self):

        # plot the connectivity matrix
        fig, ax = plt.subplots(2, 2, figsize=(4, 4),tight_layout=True)
        A = ax[0,0].imshow(self.Win.T)
        plt.colorbar(A)
        ax[0,0].set_title("$W_{in}$")

        B = ax[0,1].imshow(self.Wrec.T)
        plt.colorbar(B)
        ax[0,1].set_title("$W_{rec}$")

        C = ax[1,0].imshow(self.Wout.T)
        plt.colorbar(C)
        ax[1,0].set_title("$W_{out}$")
        plt.show()

    def network_plot(self, lim):

        # display the dynamics of the network

        fig, axs = plt.subplots(2, 2, figsize=(7, 4),tight_layout=True)

        fig.suptitle("Train")

        axs[0,0].plot(self.z[:,1:65])
        axs[0,0].set_xlim(0, lim)
        axs[0,0].set_xlabel("time step (ms)")
        axs[0,0].set_ylabel("output")

        axs[0,1].plot(self.y_train[:,1:65])
        axs[0,1].set_xlim(0, lim)
        axs[0,1].set_xlabel("time step (ms)")
        axs[0,1].set_ylabel("target")

        axs[1,0].plot((self.z - self.y_train)[:,1:65])
        axs[1,0].set_xlim(0, lim)
        axs[1,0].set_title("Training error")
        axs[1,0].set_xlabel("time step (ms)")
        axs[1,0].set_ylabel(r"$\Delta$")

        axs[1,1].set_axis_off()
        plt.show()

    def sample_plots(self, lim=200):
        """
        Plots the values of the different input an output channels
        task : task object with the task data and the trial information
        lim: time limit
        """
        relevant = self.find_relevant_points()
        x_train = self.x_train[:lim+1]
        z = self.z[:lim+1]
        r = self.r[:lim+1]
        len_trial = self.trials_len

        fig = plt.figure(figsize=(8,6))
        gs = fig.add_gridspec(4, height_ratios=[1.4,1,1.07,1.07], hspace=0, bottom=0.14,left=0.12)
        axs = gs.subplots(sharex=True)
        numcols = lim+1
        numrows = 32

        norm_r = (r - r.mean()) / r.std()
        #print(norm_r.min(), norm_r.max())
        #adj_r = r - r.min()
        #adj_r /= adj_r.max()
        #print(adj_r.min(), adj_r.max())
        axs[0].imshow(norm_r.T, vmin=-1,vmax=10, cmap="viridis",origin="lower", aspect="auto")
        axs[0].set_yticks([50,150,250],labels=(50,150,250), fontsize="x-large")

        igo = np.repeat(x_train[:,0].reshape(1,-1), 2, axis=0)
        i1go = np.concatenate((igo,x_train[:lim+1,1:33].T), axis=0)

        fix = np.repeat(z[:,0].reshape(1,-1), 2, axis=0)
        ofix = np.concatenate((fix,z[:,1:33].T), axis=0)

        axs[1].imshow(x_train[:,33:-1].T, vmin=0,vmax=1, origin="lower",extent=(-0.5,numcols-0.5,-0.5,numrows-0.5), aspect="auto")
        axs[2].imshow(i1go, vmin=0,vmax=1, origin="lower",extent=(-0.5,numcols-0.5,-2.5,numrows-0.5), aspect="auto")
        axs[3].imshow(ofix, vmin=0,vmax=1, origin="lower", extent=(-0.5,numcols-0.5,-2.5,numrows-0.5), aspect="auto")

        for i in range(lim//len_trial):
            axs[0].axvline((i+1)*len_trial-0.5, color="lime", linewidth=1.5)
            axs[1].axvline((i+1)*len_trial-0.5, color="lime", linewidth=1.5)
            axs[2].axvline((i+1)*len_trial-0.5, color="lime", linewidth=1.5)
            axs[3].axvline((i+1)*len_trial-0.5, color="lime", linewidth=1.5)

        axs[2].axhline(-0.5, color="darkslategray", linewidth=1.5)
        axs[3].axhline(-0.5, color="slategray", linewidth=1.5)


        axs[1].set_xlim(0, lim)
        axs[2].set_xlim(0, lim)
        axs[3].set_xlim(0, lim)

        axs[0].set_ylabel(r"$\mathbf{N}\ $",rotation=0, fontsize="x-large")
        axs[1].set_ylabel(r"$\mathbf{I_2}$",rotation=0, fontsize="x-large")
        axs[2].set_ylabel(r"$\mathbf{I_1}$",rotation=0, fontsize="x-large")
        axs[3].set_ylabel(r"$\mathbf{O}$", rotation=0, fontsize="x-large")
        axs[3].set_xlabel("t", fontweight="bold")

        axs[2].annotate(r"$\mathbf{I_{fix}}$", xy=(0, -1.5), xycoords="data", xytext=(-38, -0.7), textcoords="axes points",fontsize="large", fontweight="bold")
        axs[3].annotate(r"$\mathbf{O_{fix}}$", xy=(0, -1.5), xycoords="data", xytext=(-38, -0.7), textcoords="axes points",fontsize="large", fontweight="bold")

        axs[0].spines["bottom"].set_edgecolor("silver")
        axs[1].spines["top"].set_edgecolor("silver")
        axs[1].spines["bottom"].set_edgecolor("darkslategray")
        axs[2].spines["top"].set_edgecolor("darkslategray")
        axs[2].spines["bottom"].set_edgecolor("silver")
        axs[3].spines["top"].set_edgecolor("silver")

        axs[1].set_yticks([0,16,32],labels=(0,r"$\pi$",r"$2\pi$"), fontsize="x-large")
        axs[2].set_yticks([0,16],labels=(0,r"$\pi$"), fontsize="x-large")
        axs[3].set_yticks([0,16],labels=(0,r"$\pi$"), fontsize="x-large")
    
        fig.align_ylabels()

    def ID_FCI_estimator(self, Niter, full, method = "full", **kwargs):
        """
        compute the intrinsic dimension using the FCI method
        Parameters:
        -----------
        Niter : int
            the number of iterations if method is "mc"

        full : boolean
            if True returns also the fit parameters and the fci integral

        method : string
            the method used to compute the intrinsic dimension, default is "full",
            the other option is "mc".
        Returns:
        --------
        id_fci : float
            the intrinsic dimension using FCI
        """

        if method == "mc":
            # compute the FCI using montecarlo
            if full:
                id_fci, fit, fci = ID_FCI_MC(self.r, Niter, full, **kwargs)
            else:
                id_fci = ID_FCI_MC(self.r, Niter, **kwargs)
        elif method == "full":
            # compute the FCI
            if full:
                id_fci, fit, fci = ID_FCI(self.r, full, **kwargs)
            else:
                id_fci = ID_FCI(self.r, **kwargs)
        else:
            raise ValueError("method must be 'full' or 'mc'")

        self.id_fci = id_fci
        if full:
            return id_fci, fit, fci
        else:
            return id_fci

    def ID_pca(self,normalize, full):
        """
        Compute the intrinsic dimension using the PCA explained variance.

        Parameters
        ----------
        normalize : boolean
            if True, the data points are normalized
        full : boolean
            if True returns also the PCs and the explained variance
        Returns
        -------
        id_pca : float
            the intrinsic dimension using PCA
        y : array
            the projected data points
        v : array
            the explained variance ratio
        """
        if full:
            id_pca, y, v = ID_PCA(self.r, normalize, full)
            self.id_pca = id_pca
            return id_pca, y, v
        else:
            id_pca = ID_PCA(self.r, normalize)
            self.id_pca = id_pca
            return id_pca

    def ID_participation_ratio(self):
        """
        Compute the intrinsic dimension using the participation ratio.

        Parameters
        ----------
        None

        Returns
        -------
        id_pr : float
            The intrinsic dimension using the participation ratio
        """
        d_pr = participation_ratio(self.r)
        self.id_pr = d_pr
        return d_pr

    def ID_parallel_analysis(self,data_range = None,**kwargs):
        """
        Compute the intrinsic dimension using the parallel analysis.

        Parameters
        ----------
        data_range : array, optional, default None
            the range of the data for the parallel analysis
        kwargs : dict
            keyword arguments for the parallel analysis:
            - num_shuffles: default 250,
            - percentile: default 95

        Returns
        -------
        id_pa : float
            The intrinsic dimension using the parallel analysis
        """
        if data_range is None:
            id_pa = id_parallel_analysis(self.r, **kwargs)
        else:
            id_pa = id_parallel_analysis(self.r[data_range], **kwargs)
        self.id_pa = id_pa
        return id_pa

    def id_twoNN(self, twonn=True, scale_dependent=False,**kwargs):

        # compute the intrinsic dimension using two nearest neighbors
        drange = kwargs.get('drange', None)
        if twonn and scale_dependent:
            id_2NN, id_2NN_scale, fitting_params = ID_twoNN_dadapy(self.r[drange].squeeze(), twonn, scale_dependent, **kwargs)
            self.id_2NN = id_2NN
            self.id_2NN_scale = id_2NN_scale
            return id_2NN, id_2NN_scale, fitting_params
        elif twonn and not scale_dependent:
            id_2NN, fitting_params = ID_twoNN_dadapy(self.r[drange].squeeze(), twonn, scale_dependent, **kwargs)
            self.id_2NN = id_2NN
            return id_2NN, fitting_params
        elif not twonn and scale_dependent:
            id_2NN_scale, fitting_params = ID_twoNN_dadapy(self.r[drange].squeeze(), twonn, scale_dependent, **kwargs)
            self.id_2NN_scale = id_2NN_scale
            return id_2NN_scale, fitting_params

    def ID(self):

        # compute the intrinsic dimension

        d, x, y = ID_estimator(self.r)

        self.d = d
        self.x = x
        self.y = y

    def collect_ID_FCI(self, n_iter = 20, method = "mc", **kwargs):
        """
        compute the intrinsic dimension using FCI
        parameters:
        -----------
        n_iter : int
            the number of iterations if method is "mc"

        method : string
            the method used to compute the intrinsic dimension, default is "mc",
            the other option is "full".
        """
        q = self.q
        n = self.trials_len
        n1 = self.stim_start
        n2 = self.go_start

        r = self.r
        norm_r = pyFCI.center_and_normalize(r)

        beginning = self.beginning
        middle = self.middle
        final = self.final
        if method == "full":
            d_beg_fci = ID_FCI(norm_r[beginning,:], normalize=False, **kwargs)
        else:
            d_beg_fci = ID_FCI_MC(norm_r[beginning,:], n_iter, normalize=False, **kwargs)
        print(f"Beginning dimension: {d_beg_fci}")
       
        if n2-n1>0:
            if method == "full":
                d_mid_fci = ID_FCI(norm_r[middle,:], normalize=False, **kwargs)
            else:
                d_mid_fci = ID_FCI_MC(norm_r[middle,:], n_iter, normalize=False, **kwargs)
            print(f"Middle dimension: {d_mid_fci}")
        else:
            d_mid_fci = 0
        if method == "full":
            d_fin_fci = ID_FCI(norm_r[final,:], normalize=False, **kwargs)
        else:
            d_fin_fci = ID_FCI_MC(norm_r[final,:], n_iter, normalize=False, **kwargs) 
        print(f"Final dimension: {d_fin_fci}")

        if method == "full":
            d_tot_fci = ID_FCI(norm_r, normalize=False, **kwargs)
        else:
            d_tot_fci = ID_FCI_MC(norm_r, n_iter, normalize=False, **kwargs)
        print(f"Total dimension: {d_tot_fci}")

        dim_vector_fci = np.array([d_beg_fci, d_mid_fci, d_fin_fci, d_tot_fci])
        self.dim_fci_regions = dim_vector_fci
                
        return dim_vector_fci

    def collect_ID_twoNN(self, display=False, **kwargs):

        beginning = self.beginning
        middle =  self.middle   
        final  = self.final    
        r = self.r

#        norm_r = (r - np.mean(r, axis=0)) / np.std(r, axis=0)

        if len(beginning)>10:
            d_beg, beg = ID_twoNN_dadapy(r[beginning, :])
            print(f"Beginning dimension: {d_beg}")
        else:
            d_beg = 0
        
        if len(middle)>10:
            d_mid, mid = ID_twoNN_dadapy(r[middle, :])
            print(f"Middle dimension: {d_mid}")
        else:
            d_mid = 0

        if len(final)>10:
            d_fin, fin = ID_twoNN_dadapy(r[final, :])
            print(f"Final dimension: {d_fin}")
        else:
            d_fin = 0

        d_tot, tot = ID_twoNN_dadapy(r)
        print(f"Total dimension: {d_tot}")
        dim_vector = np.array([d_beg, d_mid, d_fin, d_tot])
        self.dim_twonn_regions = dim_vector
        if display:
            self.display_ID(beg, mid, fin, tot)

        return dim_vector


    def display_ID(self, beg, mid, fin, tot):

        # plot the intrinsic dimension

        dims = self.dim_vector_twonn
        d_beg = dims[0]
        d_mid = dims[1]
        d_fin = dims[2]
        d_tot = dims[3]
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].scatter(beg[1], beg[0], s=20, marker="x")
        axs[0, 0].plot(beg[1], beg[1]*d_beg, c="red")
        axs[0, 0].set_title(f"Beginning dimension: {np.round(d_beg, 2)}")

        axs[0, 1].scatter(mid[1], mid[0], s=20, marker="x")
        axs[0, 1].plot(mid[1], mid[1]*d_mid, c="red")
        axs[0, 1].set_title(f"Middle dimension: {np.round(d_mid, 2)}")

        axs[1, 0].scatter(fin[1], fin[0], s=20, marker="x")
        axs[1, 0].plot(fin[1], fin[1]*d_fin, c="red")
        axs[1, 0].set_title(f"Final dimension: {np.round(d_fin, 2)}")

        axs[1, 1].scatter(tot[1], tot[0], s=10, marker="x")
        axs[1, 1].plot(tot[1], tot[1]*d_tot, c="red")
        axs[1, 1].set_title(f"Total dimension: {np.round(d_tot, 2)}")

        plt.show()

    
    def scale_ID_twoNN(self, display = False, **kwargs):

        # compute scale dependent ID for every range defined before

        beginning = self.beginning
        middle =  self.middle   
        final  = self.final    
        r = self.r

        # compute the intrinsic dimension
        if len(beginning)>10:
            dim_beg, beg = ID_twoNN_dadapy(r[beginning, :],twonn=False, scale_dependent=True, **kwargs)
        else:
            dim_beg = 0

        if len(middle)>10:
            dim_mid, mid = ID_twoNN_dadapy(r[middle, :],twonn=False, scale_dependent=True, **kwargs)
        else:
            dim_mid = 0

        if len(final)>10:
            dim_fin, fin = ID_twoNN_dadapy(r[final, :],twonn=False, scale_dependent=True, **kwargs)
        else:
            dim_fin = 0
        dim_tot, tot = ID_twoNN_dadapy(r,twonn=False, scale_dependent=True, **kwargs)

        # compute and return the four dimensions

        res = np.array([dim_beg, dim_mid, dim_fin, dim_tot])

        self.dim_twonn_scale_regions = res

        if display:
            if "beg" in locals():
                fig, axs = plt.subplots(2, 2, figsize=(12, 7))
                axs[0, 0].errorbar(beg[1], beg[0], yerr=beg[2], marker="o", capsize=4)
                axs[0, 0].set_title(f"Beginning dimension")
            #axs[0, 0].set_xscale("log")
                axs[0, 0].grid(axis="y")

            if "mid" in locals(): 
                axs[0, 1].errorbar(mid[1], mid[0], yerr=mid[2], marker="o", capsize=4)
                axs[0, 1].set_title(f"Middle dimension")
            #axs[0, 1].set_xscale("log")
                axs[0, 1].grid(axis="y")

            if "fin" in locals():
                axs[1, 0].errorbar(fin[1], fin[0], yerr=fin[2], marker="o", capsize=4)
                axs[1, 0].set_title(f"Final dimension")
            #axs[1, 0].set_xscale("log")
                axs[1, 0].grid(axis="y")

            axs[1, 1].errorbar(tot[1], tot[0], yerr=tot[2], marker="o", capsize=4)
            axs[1, 1].set_title(f"Total dimension")
            #axs[1, 1].set_xscale("log")
            axs[1, 1].grid(axis="y")

            plt.show()

        return res, dim_beg, dim_mid, dim_fin, dim_tot

    def find_dim(self, dim):

        # find the dimension from the plateaux in the scale dependent ID
        plateaus, _, _ = find_plateaus(dims, min_length=2, tolerance = 0.3)

        if len(plateaus) > 0:
          means = []
          for p in plateaus:
              means.append(np.mean(dims[p[0]:p[1]]))
          p_dim = np.min(means)
        else:
          p_dim = ids_scaling[-1]

        return p_dim
