import numpy as np
import quantities as pq

import neo
import elephant.statistics as estats
import elephant.asset as asset
import elephant.spike_train_generation as stg

# ===========================================================================
# Parameters definition
# ===========================================================================
# Data parameters
N = 100                 # Number of spike trains
rate = 15 * pq.Hz       # Firing rate
T = 1 * pq.s            # Length of data 

# Parameters for the ASSET analysis
binsize = 5 * pq.ms     # bin size
fl = 5                  # filter length
fw = 2                  # filter width

kernel_params= (fl, fw, fl)

prob_method = 'b'       # Defines the method to calculate the probability 
                        # matrix: 'a' = analytical, 'b' = bootstrapping

n_surr = 100           # Number of surrogates for the bootstrapping method
dither_T = binsize * 5  # Window size for spike train dithering (bootstrapping)

alpha1 = 1e-2           # threshold for 1st test
alpha2 = 1e-5           # threshold for 2nd test

eps = 5                 # cluster diameter in dbscan
minsize = 3             # min size of a cluster (diagonal structure)
stretch = 10            # stretching coefficient for the euclidean metric

# Parameters for rate estimation
sampl_period = 5.*pq.ms
binwidth = 20 * pq.ms
t_pre = 0 * pq.ms
t_post = 1000 * pq.ms

# =======================================================================
# Data generation
# =======================================================================

# Generate the data
sts = []
for i in range(N):
    np.random.seed(i)
    sts.append(stg.homogeneous_poisson_process(rate, t_stop=T)) 

# =======================================================================
# ASSET Method
# =======================================================================
imat, xx, yy = asset.intersection_matrix(sts, binsize=binsize, dt=T)

# Compute the probability matrix, either analytically or via bootstrapping
if prob_method == 'a':
    # Estimate rates
    fir_rates = list(np.zeros(shape=len(sts)))
    for st_id, st_trial in enumerate(sts):
        fir_rates[st_id] = estats.instantaneous_rate(st_trial,
                                                 sampling_period=sampl_period)
        fir_rates[st_id] = neo.AnalogSignal(
            fir_rates[st_id], t_start=t_pre, t_stop=t_post,
            sampling_period=sampl_period)
    # Compute the probability matrix analytically
    pmat, x_edges, y_edges = asset.probability_matrix_analytical(
        sts, binsize, dt=T, fir_rates=fir_rates)
elif prob_method == 'b':
    # Compute the probability matrix via bootstrapping (Montecarlo)
    pmat, x_edges, y_edges = asset.probability_matrix_montecarlo(
        sts, binsize, dt=T, j = dither_T, n_surr=n_surr)

# Compute the joint probability matrix
jmat = asset.joint_probability_matrix(
    pmat, filter_shape=(fl, fw), alpha=0, pvmin=1e-5)

# Define thresholds
q1, q2 = 1. - alpha1, 1. - alpha2

# Create the mask from pmat and jmat
mmat = asset.mask_matrices([pmat, jmat], [q1, q2])

# Cluster the entries of mmat
cmat = asset.cluster_matrix_entries(mmat, eps, minsize, stretch)

# Extract the SSEs from the cluster matrix
sse_found = asset.extract_sse(sts, xx, yy, cmat)