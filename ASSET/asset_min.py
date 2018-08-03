
import numpy as np
import quantities as pq

import neo
import elephant.statistics as estats
import asset as asset
import elephant.spike_train_generation as stg
import time
from simplepytimer import MultiTimer
import argparse

try:
    from mpi4py import MPI
    mpi_accelerated = True
except:
    mpi_accelerated = False

MultiTimer("import")

# ===========================================================================
# Parameters definition
# ===========================================================================
parser = argparse.ArgumentParser(description='Elephant ASSET Mini Benchmark')
parser.add_argument('--spike-trains', dest='N', default=100, type=int, help='Number of Spike Trains')
parser.add_argument('--firing-rate', dest='rate', default=15, type=int, help='Firing Rate (in Hz)')
parser.add_argument('--data-length', dest='T', default=5, type=int, help='Length of Data (in s)')
parser.add_argument('--bin-size', dest='binsize', default=5, type=int, help='Bin Size (in ms)')
parser.add_argument('--surrogates', dest='n_surr', default=10000, type=int, help='Number of Surrogates (for Bootstrapping)')
parser.add_argument('--prob-method', dest='prob_method', choices=["a", "b"], default="a", type=str, help='Method for calculation of probability; a: analytical, b: bootstrapping')

args = parser.parse_args()
# Data parameters
N = args.N              # Number of spike trains
rate = args.rate * pq.Hz  # Firing rate
T = args.T * pq.s       # Length of data 

# Parameters for the ASSET analysis
binsize = args.binsize * pq.ms  # bin size
fl = 5                  # filter length
fw = 2                  # filter width

kernel_params= (fl, fw, fl)

prob_method = args.prob_method  # Defines the method to calculate the probability 
                                # matrix: 'a' = analytical, 'b' = bootstrapping

n_surr = args.n_surr    # Number of surrogates for the bootstrapping method
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

MultiTimer("init")

# =======================================================================
# Data generation
# =======================================================================

# Generate the data
sts = []
for i in range(N):
    np.random.seed(i)
    sts.append(stg.homogeneous_poisson_process(rate, t_stop=T)) 

MultiTimer("generate data")

# =======================================================================
# ASSET Method
# =======================================================================
imat, xx, yy = asset.intersection_matrix(sts, binsize=binsize, dt=T)

MultiTimer("intersection_matrix")

# Compute the probability matrix, either analytically or via bootstrapping
if prob_method == 'a':
    # Estimate rates
    fir_rates = list(np.zeros(shape=len(sts)))

    #########################################
    ## TODO: MPI 
    if mpi_accelerated:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    ##############################################

    for st_id, st_trial in enumerate(sts):
        # Only calculate one one rank
        if mpi_accelerated and st_id % size != rank:
            continue 

        fir_rates[st_id] = estats.instantaneous_rate(st_trial,
                                                 sampling_period=sampl_period)
        fir_rates[st_id] = neo.AnalogSignal(
            fir_rates[st_id], t_start=t_pre, t_stop=t_post,
            sampling_period=sampl_period)

    MultiTimer("before broadcast")
    # make sure all date on all nodes
    if mpi_accelerated:
        for st_id, st_trial in enumerate(sts):
            fir_rates[st_id] = comm.bcast(fir_rates[st_id], root=st_id % size )

    # Compute the probability matrix analytically
    pmat, x_edges, y_edges = asset.probability_matrix_analytical(
        sts, binsize, dt=T, fir_rates=fir_rates)
    MultiTimer("probability_matrix_analytical")

elif prob_method == 'b':
    # Compute the probability matrix via bootstrapping (Montecarlo)
    pmat, x_edges, y_edges = asset.probability_matrix_montecarlo(
        sts, binsize, dt=T, j = dither_T, n_surr=n_surr)
    MultiTimer("probability_matrix_montecarlo")
# Compute the joint probability matrix
jmat = asset.joint_probability_matrix(
    pmat, filter_shape=(fl, fw), alpha=0, pvmin=1e-5)
MultiTimer("joint_probability_matrix")
# Define thresholds
q1, q2 = 1. - alpha1, 1. - alpha2

# Create the mask from pmat and jmat
mmat = asset.mask_matrices([pmat, jmat], [q1, q2])
MultiTimer("mask_matrices")

# Cluster the entries of mmat
cmat = asset.cluster_matrix_entries(mmat, eps, minsize, stretch)
MultiTimer("cluster_matrix_entries")

# Extract the SSEs from the cluster matrix
sse_found = asset.extract_sse(sts, xx, yy, cmat)
MultiTimer("extract_sses")

file = open("sse_found", "w")
file.write(str(sse_found))
file.close()
file_runtime = open("runtime.csv", "a")
file_runtime.write("{0},{1}\n".format(size, MultiTimer.runtime()))
file_runtime.close()

if rank is 0:
    MultiTimer.print_timings(header=True, seperator=",", prefix="")

if mpi_accelerated and not MPI.COMM_WORLD.Get_rank() is 0:
    exit(0)
   
