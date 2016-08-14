# -*- coding: utf-8 -*-

import fima_stp
import elephant.spike_train_generation as stg
import elephant.conversion as conv
import numpy as np
import quantities as pq
import neo


def generate_stp(occ, xi, t_stop, delays, rate):
    '''
    Generate a spatio-temporal-pattern (STP). One pattern consists in a
    repeated sequence of spikes with fixed inter spikes intervals (delays).
    The starting time of the repetitions of the pattern are randomly generated.
    '''
    # Generating all the first spikes of the repetitions
    s1 = np.arange(0, t_stop.magnitude, t_stop.magnitude / occ)

    # Using matrix algebra to add all the delays
    s1_matr = (s1*np.ones([xi-1, occ])).T
    delays_matr = np.ones(
        [occ, 1])*delays.rescale(t_stop.units).magnitude.reshape([1, xi-1])
    ss = s1_matr+delays_matr

    # Stacking the first and successive spikes
    stp = np.hstack((s1.reshape(occ, 1), ss))

    # Generating the background noise
    noise = [
        list(stg.homogeneous_poisson_process(
            rate, t_stop=t_stop).magnitude) for i in range(xi)]
    # Transofm in to neo SpikeTrain
    stp = [
        neo.core.SpikeTrain(
            (sorted(noise[i]+list(t)))*t_stop.units,
            t_stop) for i, t in enumerate(stp.T)]
    return stp


# Data parametrs
N = 20  # Number of spike trains
rate = 5 * pq.Hz  # Firing rate
T = 1 * pq.s  # Length of data
xi = 10  # Number of spikes forming the Spatial Temporal Pattern (STP)
occ = int(T.magnitude * 10)  # Number of occurrences of the pattern


# Data generation
np.random.seed(N-xi)
sts = generate_stp(occ, xi, T, np.arange(5, 5*(xi), 5)*pq.ms, rate)
for i in range(N-xi):
    np.random.seed(i+xi)
    sts.append(stg.homogeneous_poisson_process(rate, t_stop=T))

# Analysis parameters
wndlen = 50  # Length of the sliding window
width = 1 * pq.ms  # Time resolution of the patterns
dither = 25 * pq.ms  # Time dithering for surrogates generation
alpha = 0.01  # Significance level
n_surr = 100  # Number of surrogate data to generate
n_samples = 500  # Number of samples used to approximate stability

# Boostrap technique to compute the Pattern Spectrum (matrix of p-values)
# (Computational expensive)
PvSpec = fima_stp.pvspec(
    sts, wndlen, width,  dither=dither, n=n_surr, min_z=3, min_c=3)
nsSgnt = fima_stp.sspec(PvSpec, alpha, corr='fdr', report='e')

# Conversion of data in transaction (input format for FP-growth algorithm)
binned_sts = conv.BinnedSpikeTrain(sts, width).to_array()
context, rel_matrix = fima_stp.buildContext(binned_sts, wndlen)
Trans = fima_stp.st2trans(sts, wndlen, width=width)

# Mining the data with FP-growth algorithm
concepts_int = [
    i[0] for i in fima_stp.fpgrowth(
        Trans, target='c', min_z=3, min_c=3, report='a')]

# Computing the stability measure (Computational expensive)
concepts = fima_stp._approximate_stability_extensional(
        concepts_int, rel_matrix, wndlen, n_samples)

# Selecting only significant concepts
concepts_psf = [
    c for c in concepts if (
        len(c[0]), len(c[1])) not in nsSgnt]
concepts_psr = fima_stp.psr(concepts_psf, nsSgnt, wndlen)

# Alternative single call for the entire method
# concepts, sgnf_concepts, pvalues_spectra, nsgnf_spectra = fima_stp.psf(
#    sts, wndlen, width, dither, alpha, min_z=3, min_c=3, n=n_surr,
#    n_samples=n_samples)

# Check the results include the injected pattern
for patt in concepts_psr:
    if list(sorted(patt[0])) == [i*wndlen + i*5 for i in range(xi)]:
            print "Succesfull Detection of the Spatial Temoral Pattern"
            break
