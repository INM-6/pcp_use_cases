import numpy
import neo
import elephant.spike_train_surrogates as surr
import elephant.conversion as conv
from mpi4py import MPI  # for parallelized routines
from itertools import chain, combinations
import numpy as np
import scipy.sparse as sps
import time


def st2trans(sts, wndlen, width):
    """
    Turn a list of spike trains into a list of transaction.

    Parameters
    ----------
    sts : list
    List of neo.Spike_trains to be converted
    wndlen : int
    length of sliding window
    width : quantity
    length of the binsize used to bin the data

    Returs
    --------
    trans : list
    List of all transactions, each element of the list contains the attributes
    of the corresponding object
    """
    # Bin the spike trains
    sts_bool = conv.BinnedSpikeTrain(
        sts, binsize=width).to_bool_array()
    # List of all the possible attributes (spikes)
    attributes = np.array(
        [s*wndlen + t for s in range(len(sts)) for t in range(wndlen)])
    trans = []
    # Assigning to each of the oject (window) his attributes (spikes)
    for w in range(sts_bool.shape[1] - wndlen + 1):
        currentWindow = sts_bool[:, w:w+wndlen]
        # only keep windows that start with a spike
        if np.add.reduce(currentWindow[:, 0]) == 0:
            continue
        trans.append(attributes[currentWindow.flatten()])
    return trans


def buildContext(binned_sts, wndlen):
    """
    Building the context given a matrix (number oftrains x number of bins) of
    binned spike trains
    Parameters
    ----------
    binned_sts :
    Binary matrix containing the binned spike trais
    wndlen :
    length of sliding window

    Returns:
    context : list
    List of tuples composed by the window and the correspondent spikes
    rel_matrix : np.ndarray
    Matrix representation of the binary relation of the context. On the raw
    are listed the objects (windows) on the columns the attributes (spikes).
    """
    # Initialization of the outputs
    context = []
#    cols = []
#    rows = []
    shape = (binned_sts.shape[1] - wndlen + 1, len(binned_sts) * wndlen)
    rel_matrix = np.zeros(shape)
    # Array containing all the possible attributes (spikes)
    attributes = np.array(
        [s*wndlen + t for s in range(len(binned_sts)) for t in range(wndlen)])
    binned_sts = np.array(binned_sts, dtype="bool")

    for w in range(binned_sts.shape[1]-wndlen+1):
        # spikes in the window
        currentWindow = binned_sts[:, w:w+wndlen]
        # only keep windows that start with a spike
        if np.add.reduce(currentWindow[:, 0]) == 0:
            continue
        times = currentWindow.flatten()
        context += [(w, a) for a in attributes[times]]
        rel_matrix[w, :] = times
#        nnz = times.nonzero()[0]
#        cols.extend(nnz)
#        rows.extend([w] * len(nnz))
#        sp_martix = sparse_matrix(rows, cols, shape)
    return context, rel_matrix


def sparse_matrix(rows, cols, shape):
    """
    Converts all contexts into a sparse matrix

    `Cols` is used to create the binary entries of the matrix.

    Parameters
    ----------
    shape: shape of matrix
    rows: list
        Rows of matrix, indicates where an entry in the row is
    cols: list
        Columns of matrix, indicates where an entry in the column is
    Returns
    -------
    sparse matrix: scipy.sparce.csr_matrix
        Sparse matrix representation of the contexts

    """
    return sps.csr_matrix((np.ones_like(cols, dtype=bool), (rows, cols)),
                          shape)


def fpgrowth(
        tracts, target='s', min_c=2, min_z=2, max=None, report='a', algo='s'):
    '''
    Find frequent item sets with the fpgrowth algorithm.

    INPUT:
        tracts [list of lists]
            transaction database to mine. The database must be an iterable of
            transactions; each transaction must be an iterable of items; each
            item must be a hashable object. If the database is a dictionary,
            the transactions are the keys, the values their (integer)
            multiplicities.
        target [str. Default: 's']
            type of frequent item sets to find
            s/a:   sets/all   all     frequent item sets
            c  :   closed     closed  frequent item sets
            m  :   maximal    maximal frequent item sets
            g  :   gens       generators
        min_c [int. Default: 2]
            minimum support of an item set
            (positive: absolute number, negative: percentage)
        min_z  [int. Default: 2]
            minimum number of items per item set
        max  [int. Default: no limit]
            maximum number of items per item set
        report  [str. Default: 'a']
            values to report with an item set
            a     absolute item set support (number of transactions)
            s     relative item set support as a fraction
            S     relative item set support as a percentage
            e     value of item set evaluation measure
            E     value of item set evaluation measure as a percentage
            #     pattern spectrum instead of full pattern set
        algo [str. Default: 's']
            algorithm variant to use:
            s     simple     simple  tree nodes with only link and parent
            c     complex    complex tree nodes with children and siblings
            d     single     top-down processing on a single prefix tree
            t     topdown    top-down processing of the prefix trees
            Variant d does not support closed/maximal item set mining.

    OUTPUT:
        * If *report* == 'a'/'s'/'S'/'e'/'E' return a list of pairs, each
          consisting of a frequent itemset (as a tuple of unit IDs) and a
          value representing that itemset's support or evaluation measure
        * If *report* == '#', return a pattern spectrum as a list of triplets
          (size, supp, cnt), representing pattern size, pattern support, and
          number of patterns with that size and that support found in *tracts*
    '''
    import fim

    # By default, set the maximum pattern size to the number of spike trains
    if max is None:
        max = numpy.max([len(t) for t in tracts])+1

    # Run the original fpgrowth
    fpgrowth_output = fim.fpgrowth(
        tracts=tracts, target=target, supp=-min_c, min=min_z, max=max,
        report=report, algo='s')
    # Return the output
    if report != '#':
        return [(cfis, s[0]) for (cfis, s) in fpgrowth_output]
    else:
        return fpgrowth_output


def pvspec(sts, wndlen, width, dither, n, min_z=2, min_c=2, verbose=False):
    '''
    compute the p-value spectrum of pattern signatures extracted from
    surrogates of parallel spike trains *sts*, under the null hypothesis of
    spiking independence.

    * *n* surrogates are obtained from each spike train by spike dithering
      (--> elephant.core.surrogates.gensurr_dither())
    * closed frequent itemsets (CFISs) are collected from each surrogate data
      (--> fpgrowth())
    * the signatures (size, support) of all CFISs are computed, and their
      occurrence probability estimated by their occurrence frequency
    * CFISs in *sts* whose signatures are significant are returned

    Parameters
    ----------
    sts [list]
        list of neo.core.SpikeTrain objects, interpreted as parallel
        spike trains, or list of (ID, train) pairs. The IDs must be
        hashable. If not specified, they are set to integers 0,1,2,...
    width [quantity.Quantity]
        time span for evaluating spike synchrony.
        * if *method* == 'd', this is the width of the time bins used by
          fpgrowth() routine
        * if *method* == 'c', this is the width of the sliding window
          used by the coconad() routine
    dither [quantity.Quantity]
        spike dithering amplitude. Surrogates are generated by randomly
        dithering each spike around its original position by +/- *dither*
    n [int]
        amount of surrogates to generate to compute the p-value spectrum.
        Should be large (n>=1000 recommended for 100 spike trains in *sts*)
    min_z [int]
        minimum size for a set of synchronous spikes to be considered
        a pattern
    min_c [int]
        minimum support for patterns to be considered frequent
    method [str. Default: 'd']
        which frequent itemset mining method to use to determine patterns
        of synchronous spikes:
        * 'd'|'discrete' : use fpgrowth() (time discretization into bins)
        * 'c'|'continuous': use coconad() (sliding window)
        'c' captures imprecise coincidences much better, but is slower.

    Output
    ------
    a list of triplets (z,c,p), where (z,c) is a pattern signature and
    p is the corresponding p-value (fraction of surrogates containing
    signatures (z*,c*)>=(z,c)). Signatures whose empirical p-value is
    0 are not listed.

    '''

    comm = MPI.COMM_WORLD   # create MPI communicator
    rank = comm.Get_rank()  # get rank of current MPI task
    size = comm.Get_size()  # get tot number of MPI tasks
    len_partition = n // size  # length of each MPI task
    len_remainder = n if len_partition == 0 else n % len_partition

    # If *sts* is a list of SpikeTrains
    if not all([type(elem) == neo.core.SpikeTrain for elem in sts]):
        raise TypeError(
            '*sts* must be either a list of SpikeTrains or a' +
            'list of (id, train) pairs')

    # For each surrogate collect the signatures (z,c) such that (z*,c*)>=(z,c)
    # exists in that surrogate. Group such signatures (with repetition)

    # list of all signatures found in surrogates, initialized to []
    SurrSgnts = []

    if rank == 0:
        for i in xrange(len_partition + len_remainder):
            Surrs = [surr.dither_spikes(
                xx, dither=dither, n=1)[0] for xx in sts]

            # Find all pattern signatures in the current surrogate data set
            SurrTrans = st2trans(Surrs, wndlen, width=width)
            SurrSgnt = [(a, b) for (a, b, c) in fpgrowth(
                SurrTrans, target='c', min_z=min_z, min_c=min_c, report='#')]
            # List all signatures (z,c) <= (z*, c*), for each (z*,c*) in the
            # current surrogate, and add it to the list of all signatures
            FilledSgnt = []
            for (z, c) in SurrSgnt:
                for j in xrange(min_z, z + 1):
                    for k in xrange(min_c, c + 1):
                        FilledSgnt.append((j, k))
            SurrSgnts.extend(list(set(FilledSgnt)))
    else:
        for i in xrange(len_partition):
            Surrs = [surr.dither_spikes(
                xx, dither=dither, n=1)[0] for xx in sts]

            # Find all pattern signatures in the current surrogate data set
            SurrTrans = st2trans(Surrs, wndlen, width=width)
            SurrSgnt = [(a, b) for (a, b, c) in fpgrowth(
                SurrTrans, target='c', min_z=min_z, min_c=min_c, report='#')]
            # List all signatures (z,c) <= (z*, c*), for each (z*,c*) in the
            # current surrogate, and add it to the list of all signatures
            FilledSgnt = []
            for (z, c) in SurrSgnt:
                for j in xrange(min_z, z + 1):
                    for k in xrange(min_c, c + 1):
                        FilledSgnt.append((j, k))
            SurrSgnts.extend(list(set(FilledSgnt)))

    if rank != 0:
        comm.send(SurrSgnts, dest=0)

    if rank == 0:
        for i in xrange(1, size):
            recv_list = comm.recv(source=i)
            SurrSgnts.extend(recv_list)

    # Compute the p-value spectrum, and return it
    PvSpec = {}
    for (z, c) in SurrSgnts:
        PvSpec[(z, c)] = 0
    for (z, c) in SurrSgnts:
        PvSpec[(z, c)] += 1
    scale = 1. / n
    PvSpec = [(a, b, c * scale) for (a, b), c in PvSpec.items()]
    if verbose is True:
        print '    end of pvspec'
    return PvSpec


def conceptFilter(c):
    """Criteria by which to filter concepts from the lattice"""
    # stabilities larger then min_st
    keepConcept = c[2] > 0.3 or c[3] > 0.3
    return keepConcept


def fdr(pvs, alpha):
    '''
    performs False Discovery Rate (FDR) statistical correction on a list of
    p-values, and assesses accordingly which of the associated statistical
    tests is significant at the desired level *alpha*

    INPUT:
        pvs [array]
            list of p-values, each corresponding to a statistical test
        alpha [float]
            significance level (desired FDR-ratio)

    OUTPUT:
        returns a triplet containing:
        * an array of bool, indicating for each p-value whether it was
          significantly low or not
        * the largest p-value that was below the FDR linear threshold
          (effective confidence level). That and each lower p-value are
          considered significant.
        * the rank of the largest significant p-value

    '''

    # Sort the p-values from largest to smallest
    pvs_array = numpy.array(pvs)              # Convert PVs to an array
    pvs_sorted = numpy.sort(pvs_array)[::-1]  # Sort PVs in decreasing order

    # Perform FDR on the sorrted p-values
    m = len(pvs)
    stop = False    # Whether the loop stopped due to a significant p-value.
    for i, pv in enumerate(pvs_sorted):  # For each PV, from the largest on
        if pv > alpha * ((m - i) * 1. / m):  # continue if PV > fdr-threshold
            pass
        else:
            stop = True
            break                          # otherwise stop

    thresh = alpha * ((m - i - 1 + stop) * 1. / m)

    # Return outcome of the test, critical p-value and its order
    return pvs <= thresh, thresh, m - i - 1 + stop


def sspec(x, alpha, corr='', report='#'):
    '''
    Compute the significance spectrum of a pattern spectrum *x*.

    Given *x* as a list of triplets (z,c,p), where z is pattern size, c is
    pattern support and p is the p-value of the signature (z,c), this routine
    assesses the significance of (z,c) using the confidence level *alpha*.
    Bonferroni or FDR statistical corrections can be applied.

    Parameters
    ----------
    x [list]
        a list of triplets (z,c,p), where z is pattern size, c is pattern
        support and p is the p-value of signature (z,c)
    alpha [float]
        significance level of the statistical test
    corr [str. Default: '']
        statistical correction to be applied:
        '' : no statistical correction
        'f'|'fdr' : false discovery rate
        'b'|'bonf': Bonferroni correction
    report [str. Defualt: '#']
        format to be returned for the significance spectrum:
        '#': list of triplets (z,c,b), where b is a boolean specifying
             whether signature (z,c) is significant (True) or not (False)
        's': list containing only the significant signatures (z,c) of *x*
        'e': list containing only the non-significant sugnatures

    Output
    ------
    return significant signatures of *x*, in the format specified by format
    '''

    x_array = numpy.array(x)  # x as a matrix; each row: (size, support, PVs)

    # Compute significance...
    if corr == '' or corr == 'no':  # ...without statistical correction
        tests = x_array[:, -1] <= alpha
    elif corr in ['b', 'bonf']:  # or with Bonferroni correction
        tests = x_array[:, -1] <= alpha * 1. / len(x)
    elif corr in ['f', 'fdr']:  # or with FDR correction
        tests, pval, rank = fdr(x_array[:, -1], alpha=alpha)
    else:
        raise ValueError("*corr* must be either '', 'b'('bonf') or 'f'('fdr')")

    # Return the specified results:
    if report == '#':
        return [(size, supp, test) for (size, supp, pv), test in zip(x, tests)]
    elif report == 's':
        return [
            (size, supp) for ((size, supp, pv), test) in zip(x, tests) if test]
    elif report == 'e':
        return [
            (size, supp) for ((size, supp, pv), test) in zip(
                x, tests) if not test]
    else:
        raise ValueError("report must be either '#' or 's'.")


def _closure_probability_extensional(intent, subset, rel_matrix):
    '''
    Return True if the closure of the subset of the extent given in input is
    equal to the intent given in input

    Parameters
    ----------
    intent : list
    Set of the attributes of the concept
    subset : list
    List of objects that form the subset of the extent to be evaluated
    rel_matrix: ndarray
    Binary matrix that specify the relation that defines the context

    Returns:
    1 if (subset)' == intent
    0 else
    '''
    # computation of the ' operator for the subset
    subset_prime = np.where(np.prod(rel_matrix[subset, :], axis=0) == 1)[0]
    if set(subset_prime) == set(list(intent)):
        return 1
    return 0


def _closure_probability_intensional(extent, subset, rel_matrix):
    '''
    Return True if the closure of the subset of the intent given in input is
    equal to the extent given in input

    Parameters
    ----------
    extent : list
    Set of the objects of the concept
    subset : list
    List of attributes that form the subset of the intent to be evaluated
    rel_matrix: ndarray
    Binary matrix that specify the relation that defines the context

    Returns:
    1 if (subset)' == extent
    0 else
    '''
    # computation of the ' operator for the subset
    subset_prime = np.where(np.prod(rel_matrix[:, subset], axis=1) == 1)[0]
    if set(subset_prime) == set(list(extent)):
        return 1
    return 0


def cmpConcepts(c1, c2):
    """Compare concepts first by extent size, then by stability"""
    if len(c1[1]) > len(c2[1]):
        return 1
    if len(c1[1]) < len(c2[1]):
        return -1
    return cmp(c1[2], c2[2])


def check_superset(patternCandidates, wndlen):
    """
    Given an intent of a concepts and the complete list of all the other
    intents, the function returns True if the intent is explained trivially
    (overlapping window) by one of the other concepts
    """
    patternCandidatesAfterSubpatternFiltering = []
    for pc in patternCandidates:
        spiketrainsPC = set([i // wndlen for i in pc[0]])
        erase = False
        for pc2 in filter(lambda p: len(p[0]) > len(pc[0]), patternCandidates):
            if len(list(pc[1])) <= len(list(pc2[1])):
                spiketrainsPC2 = set([i // wndlen for i in pc2[0]])
                if not(spiketrainsPC <= spiketrainsPC2):
                    continue
                td = None
                allEq = True
                for i in pc[0]:
                    t0 = int(i % 50)
                    t1 = int(
                        filter(lambda i2: i2//wndlen == i//wndlen, pc2[0])[0]
                        % wndlen)
                    if td is None:
                        td = t1 - t0
                    else:
                        allEq &= td == (t1-t0)
                erase |= allEq
                if erase:
                    break

        if not erase:
            patternCandidatesAfterSubpatternFiltering += [pc]

    patternCandidatesAfterSubpatternFiltering.sort(
        cmp=cmpConcepts, reverse=True)
    return patternCandidatesAfterSubpatternFiltering


def _approximate_stability_extensional(
        intents, rel_matrix, wndlen, n_samples, delta=0, epsilon=0):
    """
    Approximate the stability of concepts. Uses the algorithm described
    in Babin, Kuznetsov (2012): Approximating Concept Stability

    If `n` is 0 then an optimal n is calculated according to the
    formula given in the paper (Proposition 6):
     ..math::
                N > frac{1}{2\eps^2} \ln(frac{2}{\delta})

    Parameters
    ----------
    n: int
        Number of iterations to find an approximated stability.
    delta: float
        Probability with at least ..math:$1-\delta$
    epsilon: float
        Absolute error

    Notes
    -----
        If n is larger than the extent all subsets are directly
        calculated, otherwise for small extent size an infinite
        loop can be created while doing the recursion,
        since the random generation will always contain the same
        numbers and the algorithm will be stuck searching for
        other (random) numbers

    """
    comm = MPI.COMM_WORLD   # create MPI communicator
    rank = comm.Get_rank()  # get rank of current MPI task
    size = comm.Get_size()  # get tot number of MPI tasks
    if len(intents) == 0:
        return []
    elif len(intents) <= size:
        rank_idx = [0] * (size + 1) + [len(intents)]
    else:
        rank_idx = list(
            np.arange(
                0, len(intents) - len(intents) % size + 1,
                len(intents)//size)) + [len(intents)]
    # Calculate optimal n
    if delta + epsilon > 0 and n_samples == 0:
        n_samples = np.log(2 / delta) / (2 * epsilon ** 2) + 1
    output = []
    if rank == 0:
        for intent in intents[
                rank_idx[rank]:rank_idx[rank+1]] + intents[
                    rank_idx[-2]:rank_idx[-1]]:
            stab_ext = 0.0
            stab_int = 0.0
            extent = np.where(
                        np.prod(rel_matrix[:, intent], axis=1) == 1)[0]
            intent = np.array(list(intent))
            r_unique_ext = set()
            r_unique_int = set()
            excluded_subset = []
            # Calculate all subsets if n is larger than the power set of
            # the extent
            if n_samples > 2 ** len(extent):
                subsets_ext = chain.from_iterable(
                    combinations(extent, r) for r in range(
                        len(extent) + 1))
                for s in subsets_ext:
                    if any(
                          [set(s).issubset(se) for se in excluded_subset]):
                        continue
                    if _closure_probability_extensional(
                            intent, s, rel_matrix):
                        stab_ext += 1
                    else:
                        excluded_subset.append(s)
            else:
                for _ in range(n_samples):
                    subset_ext = extent[
                        _give_random_idx(r_unique_ext, len(extent))]
                    if any([
                        set(subset_ext).issubset(se) for
                            se in excluded_subset]):
                        continue
                    if _closure_probability_extensional(
                            intent, subset_ext, rel_matrix):
                        stab_ext += 1
                    else:
                        excluded_subset.append(subset_ext)
            stab_ext /= min(n_samples, 2 ** len(extent))
            excluded_subset = []
            # Calculate all subsets if n is larger than the power set of
            # the extent
            if n_samples > 2 ** len(intent):
                subsets_int = chain.from_iterable(
                    combinations(intent, r) for r in range(
                        len(intent) + 1))
                for s in subsets_int:
                    if any(
                          [set(s).issubset(se) for se in excluded_subset]):
                        continue
                    if _closure_probability_intensional(
                         extent, s, rel_matrix):
                        stab_int += 1
                    else:
                        excluded_subset.append(s)
            else:
                for _ in range(n_samples):
                    subset_int = intent[
                        _give_random_idx(r_unique_int, len(intent))]
                    if any([
                        set(subset_int).issubset(se) for
                            se in excluded_subset]):
                        continue
                    if _closure_probability_intensional(
                            extent, subset_int, rel_matrix):
                        stab_int += 1
                    else:
                        excluded_subset.append(subset_int)
            stab_int /= min(n_samples, 2 ** len(intent))
            output.append((intent, extent, stab_int, stab_ext))
    else:
        for intent in intents[rank_idx[rank]:rank_idx[rank+1]]:
            stab_ext = 0.0
            stab_int = 0.0
            extent = np.where(
                        np.prod(rel_matrix[:, intent], axis=1) == 1)[0]
            intent = np.array(list(intent))
            r_unique_ext = set()
            r_unique_int = set()
            excluded_subset = []
            # Calculate all subsets if n is larger than the power set of
            # the extent
            if n_samples > 2 ** len(extent):
                subsets_ext = chain.from_iterable(
                    combinations(extent, r) for r in range(
                        len(extent) + 1))
                for s in subsets_ext:
                    if any(
                          [set(s).issubset(se) for se in excluded_subset]):
                        continue
                    if _closure_probability_extensional(
                          intent, s, rel_matrix):
                        stab_ext += 1
                    else:
                        excluded_subset.append(s)
            else:
                for _ in range(n_samples):
                    subset_ext = extent[
                        _give_random_idx(r_unique_ext, len(extent))]
                    if any([
                        set(subset_ext).issubset(se) for
                            se in excluded_subset]):
                        continue
                    if _closure_probability_extensional(
                            intent, subset_ext, rel_matrix):
                        stab_ext += 1
                    else:
                        excluded_subset.append(subset_ext)
            stab_ext /= min(n_samples, 2 ** len(extent))
            excluded_subset = []
            # Calculate all subsets if n is larger than the power set of
            # the extent
            if n_samples > 2 ** len(intent):
                subsets_int = chain.from_iterable(
                    combinations(intent, r) for r in range(
                        len(intent) + 1))
                for s in subsets_int:
                    if any(
                          [set(s).issubset(se) for se in excluded_subset]):
                        continue
                    if _closure_probability_intensional(
                          extent, s, rel_matrix):
                        stab_int += 1
                    else:
                        excluded_subset.append(s)
            else:
                for _ in range(n_samples):
                    subset_int = intent[
                        _give_random_idx(r_unique_int, len(intent))]
                    if any([
                        set(subset_int).issubset(se) for
                            se in excluded_subset]):
                        continue
                    if _closure_probability_intensional(
                            extent, subset_int, rel_matrix):
                        stab_int += 1
                    else:
                        excluded_subset.append(subset_int)
            stab_int /= min(n_samples, 2 ** len(intent))
            output.append((intent, extent, stab_int, stab_ext))

    if rank != 0:
        comm.send(output, dest=0)
    if rank == 0:
        for i in xrange(1, size):
            recv_list = comm.recv(source=i)
            output.extend(recv_list)

    return output


def _give_random_idx(r_unique, n):
    """ asd """

    r = np.random.randint(n,
                          size=np.random.randint(low=1,
                                                 high=n))
    r_tuple = tuple(r)
    if r_tuple not in r_unique:
        r_unique.add(r_tuple)
        return np.unique(r)
    else:
        return _give_random_idx(r_unique, n)


def _give_random_idx_int(r_unique_int, n):
    """ asd """
    r = np.random.randint(n,
                          size=np.random.randint(low=1,
                                                 high=n))
    r_tuple = tuple(r)
    if r_tuple not in r_unique_int:
        r_unique_int.add(r_tuple)
        return np.unique(r)
    else:
        return _give_random_idx_int(r_unique_int, n)


def psf(sts, wndlen, width, dither, alpha, min_z=2, min_c=2,
        compute_stability=True, filter_concepts=True, n=100,  corr='f',
        n_samples=100, verbose=False):
    '''
    performs pattern spectrum filtering (PSF) on a list of parallel spike
    trains.

    INPUT:
    x [list]
        list of neo.core.SpikeTrain objects, interpreted as parallel
        spike trains, or list of (ID, train) pairs. The IDs must be
        hashable. If not specified, they are set to integers 0,1,2,...
    width [quantity.Quantity]
        width of the time window used to determine spike synchrony
    dither : Quantity, optional
        For methods shifting spike times randomly around their original time
        (spike dithering, train shifting) or replacing them randomly within a
        certain window (spike jittering), dt represents the size of that
        dither / window. For other methods, dt is ignored.

    alpha [float]
        significance level of the statistical test
    min_z [int. Default: 2]
        minimum size for a set of synchronous spikes to be considered
        a pattern
    min_c [int. Default: 2]
        minimum support for patterns to be considered frequent
    compute_stability [bool]
        If True the stability of all the concepts is compute. The output
        depends on the choose of the parameter filter_concepts.
        If False only the significant concepts (pattern spectrum filtering)
        are returned
    filter_concepts [bool]
        In the case compute stability is False this parameter is ignored
        Otherwise if true only concepts with stability larger than 0.3 are
        returned and the concepts are filtered using the pattern spectrum
        If False all the concepts are returned
    n : int, optional
        amount of surrogates to generate to compute the p-value spectrum.
        Should be large (n>=1000 recommended for 100 spike trains in *x*)
        Default: 100
    corr [str. Default: 'f']
        statistical correction to be applied:
        '' : no statistical correction
        'f'|'fdr' : false discovery rate
        'b'|'bonf': Bonferroni correction
    verbose : bool, optional
        whether to print the status of the analysis; might be helpful for
        large n (the analysis can take a while!)

    OUTPUT:
        returns a triplet containing:
        * all the concepts with int or ext stab >=0.3
        * the significant patterns according to PSF
        * the P-value spectrum computed on surrogate data
        * the list of non-significant signatures inferred from the spectrum

    '''
    # Compute the p-value spectrum, and compute non-significant signatures
    if verbose is True:
        print 'psf(): compute p-value spectrum...'

#    if use_mpi:
#        PvSpec = pvspec_mpi(
#        sts, wndlen, width,  shift=shift, n=n, min=min, min_c=min_c)
#    else:
    t0 = time.time()
    PvSpec = pvspec(
        sts, wndlen, width,  dither=dither, n=n, min_z=min_z, min_c=min_c)
    t1 = time.time()
    print 'pvspec time', t1-t0
    comm = MPI.COMM_WORLD   # create MPI communicator
    rank = comm.Get_rank()  # get rank of current MPI task
    # Compute transactions and CFISs of *x*
    if verbose is True:
        print 'psf(): run FIM on input data...'
    binned_sts = conv.BinnedSpikeTrain(sts, width).to_array()
    context, rel_matrix = buildContext(binned_sts, wndlen)
    Trans = st2trans(sts, wndlen, width=width)
    print 'Done conv'
    concepts_int = [
        i[0] for i in fpgrowth(
            Trans, target='c', min_z=min_z, min_c=min_c, report='a')]
    t2 = time.time()
    print 'time fpgrowth data', t2-t1
    if compute_stability:
        # Computing the approximated stability of all the conepts
        concepts = _approximate_stability_extensional(
            concepts_int, rel_matrix, wndlen, n_samples)
        t3 = time.time()
        print 'approx stability time', t3-t2
        if rank == 0:
            if not len(concepts) == len(concepts_int):
                raise ValueError('Approx stability returns less con')
            nsSgnt = sspec(PvSpec, alpha, corr=corr, report='e')
            if filter_concepts is True:
                concepts_stab = filter(conceptFilter, concepts)
#               Extract significant CFISs with pattern spectrum filtering
                concepts_psf = [
                    c for c in concepts if (
                        len(c[0]), len(c[1])) not in nsSgnt]
                # Return concepts, p-val spectrum and non-significant signature
                if verbose is True:
                    print 'psf(): done'
                t4 = time.time()
                print 'time filtering', t4-t3
                return concepts_stab, concepts_psf, PvSpec, nsSgnt
            else:
                return concepts, PvSpec, nsSgnt
        else:
            pass
    else:
        if rank == 0:
            nsSgnt = sspec(PvSpec, alpha, corr=corr, report='e')
            concepts = []
            for intent in concepts_int:
                concepts.append((set(intent), set(
                    np.where(
                        np.prod(rel_matrix[:, intent], axis=1) == 1)[0])))

            if filter_concepts is True:
                # Extract significant CFISs with pattern spectrum filtering
                concepts = [
                    c for c in concepts
                    if (len(c[0]), len(c[1])) not in nsSgnt]
            # Return concepts, p-val spectrum and non-significant signature
            if verbose is True:
                print 'psf(): done'
            return concepts, PvSpec, nsSgnt
        else:
            pass


def psr(concepts_psf, excluded, wndlen, h=0, k=2, l=0, min_size=2,
        min_supp=2):
    '''
    takes a list *cfis* of closed frequent item sets (CFISs) and performs
    pattern set reduction (PSR).
    Same as psr(), but compares each CFIS A in *cfis* to each other one which
    overlaps with A (and not just which includes/is included in A).
    In such a way, if patterns {1,2,3,4} and {1,2,3,5} are present in *cfis*
    and {1,2,3} is not, the comparison between the former two is run anyway.


    PSR determines which patterns in *cfis* are statistically significant
    given any other pattern in *cfis*, on the basis of the pattern size and
    occurrence count ("support"). Only significant patterns are retained.
    The significance of a pattern A is evaluated through its signature
    (|A|,c_A), where |A| is the size and c_A the support of A, by either of:
    * subset filtering: any pattern B is discarded if *cfis* contains a
      superset A of B such that (z_B, c_B-c_A+*h*) \in *excluded*
    * superset filtering: any pattern A is discarded if *cfis* contains a
      subset B of A such that (z_A-z_B+*k*, c_A) \in  *excluded*
    * covered-spikes criterion: for any two patterns A, B with A \subset B, B
      is discarded if (z_B-l)*c_B <= c_A*(z_A-*l*), A is discarded otherwise.
    * combined filtering: combines the three procedures above
    [More: see Torre et al (2013) Front. Comput. Neurosci. 7:132]

    Parameters
    ----------
    cfis [list]
        a list of pairs (A,c), where A is a CFIS (list of int) and c is
        its support (int).
    excluded [list. Default is an empty list]
        a list of non-significant pattern signatures (see above).
        Not used when filter='x' (see below).
    h [int. Default: 0]
        correction parameter for subset filtering (see above).
        Used if *filter* = '<', '<>' or 'c'
    k [int. Default: 2]
        correction parameter for superset filtering (see above).
        Used if *filter* = '>', '<>' or 'c'
    l [int. Default: 0]
        correction parameter for covered-spikes criterion (see above).
        Used if *filter* = 'x' or 'c'
    min_size [int. Default is 2]
        minimum pattern size. Used if *filter* = '<', '<>', 'c'
    min_supp [int. Default is 2]
        minimum pattern support. Used if *filter* = '>', '<>', 'c'

    Output
    ------
    returns a tuple containing the elements of the input argument *cfis*
    that are significant according to the PSR strategy employed.


    See also:
        subsetfilt(), supersetfilt(), subsupfilt(), xfilt(), combinedfilt()

    '''
    return list(
        combinedfilt(
            concepts_psf, excluded, wndlen, h=h, k=k, l=l, min_size=min_size,
            min_supp=min_supp))


def combinedfilt(concepts_psf, excluded, wndlen, h=0, k=2, l=0, min_size=2,
                 min_supp=2):
    '''
    takes a list concepts (see output psf function) and performs
    combined filtering based on the signature (z, c) of each pattern, where
    z is the pattern size and c the pattern support.

    For any two patterns A and B in *cfis* such that B \subset A, check:
    1) (z_B, c_B-c_A+*h*) \in *excluded*, and
    2) (z_A-z_B+*k*, c_A) \in *excluded*.
    Then:
    * if 1) and not 2): discard B
    * if 2) and not 1): discard A
    * if 1) and 2): discard B if c_B*(z_B-*l*) <= c_A*(z_A-*l*), A otherwise;
    * if neither 1) nor 2): keep both patterns.

    INPUT:
      cfis [list]
          list of concepts, each consisting in its intent and extent
      excluded [list. Default: []]
          a list of non-significant pattern signatures (z, c) (see above).
      h [int. Default: 0]
          correction parameter for subset filtering (see above).
      k [int. Default: 0]
          correction parameter for superset filtering (see above).
      l [int. Default: 0]
          correction parameter for covered-spikes criterion (see above).
      min_size [int. Default: 2]
          minimum pattern size.
      min_supp [int. Default: 2]
          minimum pattern support.

    OUTPUT:
      returns a tuple containing the elements of the input argument *cfis*
      that are significant according to combined filtering.


    See also: psr(), subsetfilt(), supersetfilt(), subsupfilt, xfilt()

    '''
    conc = []
    # Extracting from the extent and intent the spike and window times
    for concept in concepts_psf:
        intent = concept[0]
        extent = concept[1]
        spike_times = np.array([st % wndlen for st in intent])
        conc.append((intent, spike_times, extent, len(extent)))

    # by default, select all elements in conc to be returned in the output
    selected = [True for p in conc]

    # scan all conc and their subsets
    for id1, (conc1, s_times1, winds1, count1) in enumerate(conc):
        for id2, (conc2, s_times2, winds2, count2) in enumerate(conc):
            # Collecting all the possible distances between the windows
            # of the two concepts
            time_diff_all = np.array(
                [w2 - min(winds1) for w2 in winds2] + [
                    min(winds2) - w1 for w1 in winds1])
            sorted_time_diff = np.unique(
                time_diff_all[np.argsort(np.abs(time_diff_all))])
            # Rescaling the spike times to reallign to real time
            for time_diff in sorted_time_diff[
                    np.abs(sorted_time_diff) < wndlen]:
                conc1_new = [
                    t_old - time_diff for t_old in conc1]
                # if conc1 is  of conc2 are disjointed or they have both been
                # already de-selected, skip the step
                if set(conc1_new) == set(conc2) or len(
                    set(conc1_new) & set(conc2)) == 0 or (
                            not selected[id1] or not selected[id2]):
                        continue
                # Determine the support
                if hasattr(count1, '__iter__'):
                    count1 = count1[0]
                if hasattr(count2, '__iter__'):
                    count2 = count2[0]
                #TODO: check if this if else necessary
                # Test the case con1 is a superset of con2
                if set(conc1_new).issuperset(conc2):
                    # Determine whether the subset (conc2) should be rejected
                    # according to the test for excess occurrences
                    supp_diff = count2 - count1 + h
                    size1, size2 = len(conc1_new), len(conc2)
                    size_diff = size1 - size2 + k
                    reject_sub = (size2, supp_diff) in excluded or (
                        size2, size2 + 1, supp_diff,
                        supp_diff + 1) in excluded or supp_diff < min_supp

                    # Determine whether the superset (conc1_new) should be
                    # rejected according to the test for excess items
                    reject_sup = (size_diff, count1) in excluded or (
                        size_diff, size_diff + 1, count1,
                        count1 + 1) in excluded or size_diff < min_size
                    # Reject the superset and/or the subset accordingly:
                    if reject_sub and not reject_sup:
                        selected[id2] = False
                        break
                    elif reject_sup and not reject_sub:
                        selected[id1] = False
                        break
                    elif reject_sub and reject_sup:
                        if (size1 - l) * count1 >= (size2 - l) * count2:
                            selected[id2] = False
                            break
                        else:
                            selected[id1] = False
                            break
                    # if both sets are significant given the other, keep both
                    else:
                        continue

                elif set(conc2).issuperset(conc1_new):
                    # Determine whether the subset (conc2) should be rejected
                    # according to the test for excess occurrences
                    supp_diff = count1 - count2 + h
                    size1, size2 = len(conc1_new), len(conc2)
                    size_diff = size2 - size1 + k
                    reject_sub = (size2, supp_diff) in excluded or (
                        size2, size2 + 1, supp_diff,
                        supp_diff + 1) in excluded or supp_diff < min_supp

                    # Determine whether the superset (conc1_new) should be
                    # rejected according to the test for excess items
                    reject_sup = (size_diff, count1) in excluded or (
                        size_diff, size_diff + 1, count1,
                        count1 + 1) in excluded or size_diff < min_size
                    # Reject the superset and/or the subset accordingly:
                    if reject_sub and not reject_sup:
                        selected[id1] = False
                        break
                    elif reject_sup and not reject_sub:
                        selected[id2] = False
                        break
                    elif reject_sub and reject_sup:
                        if (size1 - l) * count1 >= (size2 - l) * count2:
                            selected[id2] = False
                            break
                        else:
                            selected[id1] = False
                            break
                    # if both sets are significant given the other, keep both
                    else:
                        continue
                else:
                    size1, size2 = len(conc1_new), len(conc2)
                    inter_size = len(set(conc1_new) & set(conc2))
                    reject_1 = (size1-inter_size + k, count1) in excluded or \
                        size1-inter_size + k < min_size
                    reject_2 = (
                        size2 - inter_size + k, count1) in excluded or \
                        size2 - inter_size + k < min_size
                    # Reject accordingly:
                    if reject_2 and not reject_1:
                        selected[id2] = False
                        break
                    elif reject_1 and not reject_2:
                        selected[id1] = False
                        break
                    elif reject_1 and reject_2:
                        if (size1 - l) * count1 >= (size2 - l) * count2:
                            selected[id2] = False
                            break
                        else:
                            selected[id1] = False
                            break
                    # if both sets are significant given the other, keep both
                    else:
                        continue

    # Return the selected concepts
    return [p for i, p in enumerate(concepts_psf) if selected[i]]

# import quantities as pq
# a=[]
# for i in range(5):
#    a.append(neo.core.spiketrain.SpikeTrain(
#    np.sort(np.unique([1,i, 6, 12, 17, 23, i+20]))*pq.ms,t_stop=50*pq.ms))
#
# out=psf(
#    a, 5, 1*pq.ms, alpha=0.05, dither = 10*pq.ms, n=100,
#    compute_stability=True,
#    n_samples=100, filter_concepts=True)
