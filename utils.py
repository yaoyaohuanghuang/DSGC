import csv
import os
import random
import math
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import numpy.random as rnd
from sklearn.preprocessing import normalize, StandardScaler

from signet_utils import sqrtinvdiag

def SSBM(n, k, pin, etain, pout=None, sizes='uniform', size_ratio = 2, etaout=None, values='ones'):
    """A signed stochastic block model graph generator.
    Args:
        n: (int) Number of nodes.
        k: (int) Number of communities.
        pin: (float) Sparsity value within communities.
        etain: (float) Noise value within communities.
        pout: (float) Sparsity value between communities.
        etaout: (float) Noise value between communities.
        size_ratio: Only useful for sizes 'fix_ratio', with the largest size_ratio times the number of nodes of the smallest.
        values: (string) Edge weight distribution (within community and without sign flip; otherwise weight is negated):
            'ones': Weights are 1.
            'gaussian': Weights are Gaussian, with variance 1 and expectation of 1.#
            'exp': Weights are exponentially distributed, with parameter 1.
            'uniform: Weights are uniformly distributed between 0 and 1.
        sizes: (string) How to generate community sizes:
            'uniform': All communities are the same size (up to rounding).
            'fix_ratio': The communities have number of nodes multiples of each other, with the largest size_ratio times the number of nodes of the smallest.
            'random': Nodes are assigned to communities at random.
            'uneven': Communities are given affinities uniformly at random, and nodes are randomly assigned to communities weighted by their affinity.
    Returns:
        (a,b),c where a is a sparse n by n matrix of positive edges, b is a sparse n by n matrix of negative edges c is an array of cluster membership.
    """

    if pout == None:
        pout = pin
    if etaout == None:
        etaout = etain

    rndinrange = math.floor(n * n * pin / 2 + n)
    rndin = rnd.geometric(pin, size=rndinrange)
    flipinrange = math.floor(n * n / 2 * pin + n)
    flipin = rnd.binomial(1, etain, size=flipinrange)
    rndoutrange = math.floor(n * n / 2 * pout + n)
    rndout = rnd.geometric(pout, size=rndoutrange)
    flipoutrange = math.floor(n * n / 2 * pout + n)
    flipout = rnd.binomial(1, etaout, size=flipoutrange)
    assign = np.zeros(n, dtype=int)
    ricount = 0
    rocount = 0
    ficount = 0
    focount = 0

    size = [0] * k

    if sizes == 'uniform':
        perm = rnd.permutation(n)
        size = [math.floor((i + 1) * n / k) - math.floor((i) * n / k) for i in range(k)]
        tot = size[0]
        cluster = 0
        i = 0
        while i < n:
            if tot == 0:
                cluster += 1
                tot += size[cluster]
            else:
                tot -= 1
                assign[perm[i]] = cluster
                i += 1
    elif sizes == 'fix_ratio':
        perm = rnd.permutation(n)
        if size_ratio > 1:
            ratio_each = np.power(size_ratio,1/(k-1))
            smallest_size = math.floor(n*(1-ratio_each)/(1-np.power(ratio_each,k)))
            size[0] = smallest_size
            if k>2:
                for i in range(1,k-1):
                    size[i] = math.floor(size[i-1] * ratio_each)
            size[k-1] = n - np.sum(size)
        else: # degenerate case, equaivalent to 'uniform' sizes
            size = [math.floor((i + 1) * n / k) - math.floor((i) * n / k) for i in range(k)]
        tot = size[0]
        cluster = 0
        i = 0
        while i < n:
            if tot == 0:
                cluster += 1
                tot += size[cluster]
            else:
                tot -= 1
                assign[perm[i]] = cluster
                i += 1

    elif sizes == 'random':
        for i in range(n):
            assign[i] = rnd.randint(0, k)
            size[assign[i]] += 1
        perm = [x for clus in range(k) for x in range(n) if assign[x] == clus]

    elif sizes == 'uneven':
        probs = rnd.ranf(size=k)
        probs = probs / probs.sum()
        for i in range(n):
            rand = rnd.ranf()
            cluster = 0
            tot = 0
            while rand > tot:
                tot += probs[cluster]
                cluster += 1
            assign[i] = cluster - 1
            size[cluster - 1] += 1
        perm = [x for clus in range(k) for x in range(n) if assign[x] == clus]
        print('Cluster sizes: ', size)

    else:
        raise ValueError('please select valid sizes')

    index = -1
    last = [0] * k
    for i in range(k):
        index += size[i]
        last[i] = index

    pdat = []
    prow = []
    pcol = []
    ndat = []
    nrow = []
    ncol = []
    for x in range(n):
        me = perm[x]
        y = x + rndin[ricount]
        ricount += 1
        while y <= last[assign[me]]:
            val = fill(values)
            if flipin[ficount] == 1:
                ndat.append(val)
                ndat.append(val)
                ncol.append(me)
                ncol.append(perm[y])
                nrow.append(perm[y])
                nrow.append(me)
            else:
                pdat.append(val)
                pdat.append(val)
                pcol.append(me)
                pcol.append(perm[y])
                prow.append(perm[y])
                prow.append(me)
            ficount += 1
            y += rndin[ricount]
            ricount += 1
        y = last[assign[me]] + rndout[rocount]
        rocount += 1
        while y < n:
            val = fill(values)
            if flipout[focount] != 1:
                ndat.append(val)
                ndat.append(val)
                ncol.append(me)
                ncol.append(perm[y])
                nrow.append(perm[y])
                nrow.append(me)
            else:
                pdat.append(val)
                pdat.append(val)
                pcol.append(me)
                pcol.append(perm[y])
                prow.append(perm[y])
                prow.append(me)
            focount += 1
            y += rndout[rocount]
            rocount += 1
    return (sp.coo_matrix((pdat, (prow, pcol)), shape=(n, n)).tocsc(), sp.coo_matrix((ndat, (nrow, ncol)), shape=(n, n)).tocsc()), assign



def fill(values='ones'):
    if values == 'ones':
        return float(1)
    elif values == 'gaussian':
        return np.random.normal(1)
    elif values == 'exp':
        return np.random.exponential()
    elif values == 'uniform':
        return np.random.uniform()

def fix_network(A_p,A_n,labels,eta=0.1):
    '''find the largest connected component and then increase the degree of nodes with low degrees

    Parameters
    ----------
    A_p : scipy sparse matrix of the positive part
    A_n : scipy sparse matrix of the negative part
    labels : an array of labels of the nodes in the original network
    eta: sign flip probability

    Returns
    -------
    fixed-degree and connected network, submatrices of A_p and A_n, and subarray of labels
    '''
    G = nx.from_scipy_sparse_matrix(A_p-A_n)
    largest_cc = max(nx.connected_components(G))
    A_p_new = sp.lil_matrix(A_p[list(largest_cc)][:,list(largest_cc)])
    A_n_new = sp.lil_matrix(A_n[list(largest_cc)][:,list(largest_cc)])
    labels_new = labels[list(largest_cc)]
    A_bar=sp.lil_matrix(A_p_new+A_n_new)
    A_bar_row_sum = np.array(sp.lil_matrix.sum(A_bar,axis=1)) # sum over columns to get row sums
    if np.sum(A_bar_row_sum<=2): # only do this fix if few degree node exists
        for i in np.arange(len(labels_new))[(A_bar_row_sum<=2).flatten()]:
            row_to_fix = A_bar[i].toarray().flatten()
            if sum(row_to_fix!=0)==1: # only do this fix if it is (still) a degree one node, as we may fix the nodes on the way
                # add two more edges, only add to locations currently without edges
                node_idx = (np.arange(len(labels_new))[row_to_fix==0])[random.sample(range(len(labels_new)-sum(row_to_fix!=0)),2)]
                flip_flag = np.random.binomial(size=2,n=1,p=eta) # whether to do sign flip
                for j, flip in zip(node_idx,flip_flag):
                    # fix A_bar and then adjancency matrix
                    A_bar[i,j] += 1
                    A_bar[j,i] += 1
                    if labels_new[j] == labels_new[i]:
                        if flip:
                            A_n_new[i,j] += 1
                            A_n_new[j,i] += 1
                        else:
                            A_p_new[i,j] += 1
                            A_p_new[j,i] += 1
                    else:
                        if not flip:
                            A_n_new[i,j] += 1
                            A_n_new[j,i] += 1
                        else:
                            A_p_new[i,j] += 1
                            A_p_new[j,i] += 1
            if sum(row_to_fix!=0)==2: # only do this fix if it is (still) a degree two node, as we may fix the nodes on the way
                # add one more edge, only add to locations currently without edges
                node_idx = (np.arange(len(labels_new))[row_to_fix==0])[np.random.randint(len(labels_new)-sum(row_to_fix!=0),size=1)]
                flip_flag = np.random.binomial(size=1,n=1,p=eta) # whether to do sign flip
                for j, flip in zip(node_idx,flip_flag):
                    # fix A_bar and then adjancency matrix
                    A_bar[i,j] += 1
                    A_bar[j,i] += 1
                    if labels_new[j] == labels_new[i]:
                        if flip:
                            A_n_new[i,j] += 1
                            A_n_new[j,i] += 1
                        else:
                            A_p_new[i,j] += 1
                            A_p_new[j,i] += 1
                    else:
                        if not flip:
                            A_n_new[i,j] += 1
                            A_n_new[j,i] += 1
                        else:
                            A_p_new[i,j] += 1
                            A_p_new[j,i] += 1
    return (A_p_new, A_n_new), labels_new

def polarized_ssbm(total_n=100, num_com=3, N=30, K=2, p=0.1, eta=0.1, size_ratio=1):
    ''' function to generate polarized ssbm models
    Parameters
    ----------
    total_n : total number of nodes in the polarized network
    num_com : number of conflicting communities
    N : an array of labels of the nodes in the original network
    K : number of sub-communities within a conflicting community
    p : probability of existence of an edge
    eta: sign flip probability
    size_ratio : the size ratio of the largest to the smallest block in SSBM and community size. 1 means uniform sizes. should be at least 1.

    Returns
    -------
    large_A_p and large_A_n : positive and negative parts of the polarized network
    large_labels : ordered labels of the nodes, with conflicting communities labeled together, cluster 0 is the background
    conflict_groups: an array indicating which conflicting group the node is in, 0 is background
    '''
    select_num = math.floor(total_n*p/4*total_n) # number of links in large_A_p and large_A_n respectively
    # note that we need to add each link twice for the undirected graph
    tuples_full = []
    for x in range(total_n):
        for y in range(total_n):
            tuples_full.append((x,y))
    full_idx = random.sample(tuples_full,select_num*2)
    full_idx = list(set([(x[1],x[0]) for x in full_idx])-set([(x[0],x[1]) for x in full_idx]))
    select_num = math.floor(len(full_idx)/2)
    p_row_idx = []
    p_col_idx = []
    p_dat = []
    for p_idx in full_idx[:select_num]:
        p_row_idx.append(p_idx[0])
        p_col_idx.append(p_idx[1])
        p_dat.append(1)
        p_row_idx.append(p_idx[1])
        p_col_idx.append(p_idx[0])
        p_dat.append(1)
    n_row_idx = []
    n_col_idx = []
    n_dat = []
    for n_idx in full_idx[select_num:2*select_num]:
        n_row_idx.append(n_idx[0])
        n_col_idx.append(n_idx[1])
        n_dat.append(1)
        n_row_idx.append(n_idx[1])
        n_col_idx.append(n_idx[0])
        n_dat.append(1)
    large_A_p = sp.coo_matrix((p_dat, (p_row_idx, p_col_idx)), shape=(total_n, total_n)).tolil()
    large_A_n = sp.coo_matrix((n_dat, (n_row_idx, n_col_idx)), shape=(total_n, total_n)).tolil()
    large_labels = np.zeros(total_n)
    conflict_groups = np.zeros(total_n)
    total_n_com = num_com * N # the total number of nodes in communities
    size = [0] * num_com
    if size_ratio > 1:
        ratio_each = np.power(size_ratio,1/(num_com -1))
        smallest_size = math.floor(total_n_com*(1-ratio_each)/(1-np.power(ratio_each,num_com)))
        size[0] = smallest_size
        if num_com>2:
            for i in range(1,num_com -1):
                size[i] = math.floor(size[i-1] * ratio_each)
        size[num_com-1] = total_n_com - np.sum(size)
    else: # degenerate case, equaivalent to 'uniform' sizes
        size = [math.floor((i + 1) * total_n_com / num_com) - math.floor((i) * total_n_com / num_com) for i in range(num_com)]
    counter = 0 # a counter of how many nodes have already been counted
    for com in range(num_com):
        com_size = size[com] # the size of this conflicting group, a SSBM
        (A_p, A_n), labels = SSBM(n=com_size, k=K, pin=p, etain=eta, sizes='fix_ratio',size_ratio=size_ratio)
        large_A_p[counter:counter+com_size,counter:counter+com_size] = A_p
        large_A_n[counter:counter+com_size,counter:counter+com_size] = A_n
        large_labels[counter:counter+com_size] = labels + (2*com + 1)
        conflict_groups[counter:counter+com_size] = com + 1 # start from 1
        counter += com_size
    # do permutation
    # perm[i] is the new index for node i (i is the old index)
    # label of perm[i] should therefore be the current label of node i, similar for conflict group number
    np.random.seed(2020)
    perm = rnd.permutation(total_n)
    p_row_idx, p_col_idx = large_A_p.nonzero()
    large_A_p_values = sp.csc_matrix(large_A_p).data
    p_row_idx = perm[p_row_idx]
    p_col_idx = perm[p_col_idx]
    large_A_p = sp.coo_matrix((large_A_p_values, (p_row_idx, p_col_idx)), shape=(total_n, total_n)).tocsc()
    n_row_idx, n_col_idx = large_A_n.nonzero()
    large_A_n_values = sp.csc_matrix(large_A_n).data
    n_row_idx = perm[n_row_idx]
    n_col_idx = perm[n_col_idx]
    large_A_n = sp.coo_matrix((large_A_n_values, (n_row_idx, n_col_idx)), shape=(total_n, total_n)).tocsc()
    large_labels_old = large_labels.copy()
    conflict_groups_old = conflict_groups.copy()
    for i in range(total_n):
        large_labels[perm[i]] = large_labels_old[i]
        conflict_groups[perm[i]] = conflict_groups_old[i]
    # now fix the network connectedness and degree
    # first we fix connectedness
    G = nx.from_scipy_sparse_matrix(large_A_p-large_A_n)
    largest_cc = max(nx.connected_components(G),key=len)
    A_p_new = sp.lil_matrix(large_A_p[list(largest_cc)][:,list(largest_cc)])
    A_n_new = sp.lil_matrix(large_A_n[list(largest_cc)][:,list(largest_cc)])
    labels_new = large_labels[list(largest_cc)]
    conflict_groups = conflict_groups[list(largest_cc)]
    A_bar=sp.lil_matrix(A_p_new+A_n_new)
    A_bar_row_sum = np.array(sp.lil_matrix.sum(A_bar,axis=1)) # sum over columns to get row sums
    if np.sum(A_bar_row_sum<=2): # only do this fix if few degree node exists
        for i in np.arange(len(labels_new))[(A_bar_row_sum<=2).flatten()]:
            row_to_fix = A_bar[i].toarray().flatten()
            if sum(row_to_fix!=0)==1: # only do this fix if it is (still) a degree one node, as we may fix the nodes on the way
                # add two more edges, only add to locations currently without edges
                node_idx = (np.arange(len(labels_new))[row_to_fix==0])[random.sample(range(len(labels_new)-sum(row_to_fix!=0)),2)]
                flip_flag = np.random.binomial(size=2,n=1,p=eta) # whether to do sign flip
                for j, flip in zip(node_idx,flip_flag):
                    # fix A_bar and then adjancency matrix
                    A_bar[i,j] = 1 # += 1
                    A_bar[j,i] = 1 # += 1
                    if conflict_groups[i] == conflict_groups[j]: # only apply to conflicting groups
                        if labels_new[j] == labels_new[i]:
                            if flip:
                                A_n_new[i,j] = 1 # += 1
                                A_n_new[j,i] = 1 # += 1
                            else:
                                A_p_new[i,j] = 1 # += 1
                                A_p_new[j,i] = 1 # += 1
                        else:
                            if not flip:
                                A_n_new[i,j] = 1 # += 1
                                A_n_new[j,i] = 1 # += 1
                            else:
                                A_p_new[i,j] = 1 # += 1
                                A_p_new[j,i] = 1 # += 1
                    else:
                        negative = np.random.binomial(size=1, n=1, p= 0.5)[0]
                        if negative:
                            A_n_new[i,j] = 1 # += 1
                            A_n_new[j,i] = 1 # += 1
                        else:
                            A_p_new[i,j] = 1 # += 1
                            A_p_new[j,i] = 1 # += 1
            if sum(row_to_fix!=0)==2: # only do this fix if it is (still) a degree two node, as we may fix the nodes on the way
                # add one more edge, only add to locations currently without edges
                node_idx = (np.arange(len(labels_new))[row_to_fix==0])[np.random.randint(len(labels_new)-sum(row_to_fix!=0),size=1)]
                flip_flag = np.random.binomial(size=1,n=1,p=eta) # whether to do sign flip
                for j, flip in zip(node_idx,flip_flag):
                    # fix A_bar and then adjancency matrix
                    A_bar[i,j] = 1 # += 1
                    A_bar[j,i] = 1 # += 1
                    if conflict_groups[i] == conflict_groups[j]: # only apply to conflicting groups
                        if labels_new[j] == labels_new[i]:
                            if flip:
                                A_n_new[i,j] = 1 # += 1
                                A_n_new[j,i] = 1 # += 1
                            else:
                                A_p_new[i,j] = 1 # += 1
                                A_p_new[j,i] = 1 # += 1
                        else:
                            if not flip:
                                A_n_new[i,j] = 1 # += 1
                                A_n_new[j,i] = 1 # += 1
                            else:
                                A_p_new[i,j] = 1 # += 1
                                A_p_new[j,i] = 1 # += 1
                    else:
                        negative = np.random.binomial(size=1, n=1, p= 0.5)[0]
                        if negative:
                            A_n_new[i,j] = 1 # += 1
                            A_n_new[j,i] = 1 # += 1
                        else:
                            A_p_new[i,j] = 1 # += 1
                            A_p_new[j,i] = 1 # += 1
    return (A_p_new, A_n_new), labels_new, conflict_groups

def split_labels(labels):
    nclass = torch.max(labels) + 1
    labels_split = []
    labels_split_numpy = []
    for i in range(nclass):
        labels_split.append(torch.nonzero((labels==i)).view([-1]))
    for i in range(nclass):
        labels_split_numpy.append(labels_split[i].cpu().numpy())
    labels_split_dif = []
    for i in range(nclass):
        dif_type = [x for x in range(nclass) if x!= i]
        labels_dif = torch.cat([ labels_split[x] for x in dif_type])
        labels_split_dif.append(labels_dif)
    return nclass, labels_split, labels_split_numpy, labels_split_dif

def getClassMean(nclass, labels_split, logits):
    class_mean = torch.cat([torch.mean(logits[labels_split[x]], dim=0).view(-1,1) for x in range(nclass)], dim=1)
    return class_mean


def signed_Laplacian_features(A_p,A_n, num_clusters):
    A = (A_p - A_n).tocsc()
    D_p = sp.diags(A_p.sum(axis=0).tolist(), [0]).tocsc()
    D_n = sp.diags(A_n.sum(axis=0).tolist(), [0]).tocsc()
    Dbar = (D_p + D_n)
    d = sqrtinvdiag(Dbar)
    normA = d * A * d
    L = sp.eye(A_p.shape[0], format="csc") - normA # normalized symmetric signed Laplacian
    (vals, vecs) = sp.linalg.eigsh(L, int(num_clusters), maxiter=A_p.shape[0], which='SA')
    vecs = vecs / vals  # weight eigenvalues by eigenvectors, since smaller eigenvectors are more likely to be informative
    return vecs


def spectral_adjacency_reg(A_p, A_n, k=2, normalisation='none', tau_p=None, tau_n=None, eigens=None, mi=None):
    """generate the graph features using eigenvectors of the regularised adjacency matrix.
    Args:
        k (int): The number of clusters to identify.
        normalisation (string): How to normalise for cluster size:
            'none' - do not normalise.
            'sym' - symmetric normalisation.
            'rw' - random walk normalisation.
            'sym_sep' - separate symmetric normalisation of positive and negative parts.
            'rw_sep' - separate random walk normalisation of positive and negative parts.
        tau_p (int): Regularisation coefficient for positive adjacency matrix.
        tau_n (int): Regularisation coefficient for negative adjacency matrix.
    Returns:
        array of int: Output assignment to clusters.
    Other parameters:
        eigens (int): The number of eigenvectors to take. Defaults to k.
        mi (int): The maximum number of iterations for which to run eigenvlue solvers. Defaults to number of nodes.
    """
    A = (A_p - A_n).tocsc()
    A_p = sp.csc_matrix(A_p)
    A_n = sp.csc_matrix(A_n)
    D_p = sp.diags(A_p.sum(axis=0).tolist(), [0]).tocsc()
    D_n = sp.diags(A_n.sum(axis=0).tolist(), [0]).tocsc()
    Dbar = (D_p + D_n)
    d = sqrtinvdiag(Dbar)
    size = A_p.shape[0]
    if eigens == None:
        eigens = k

    if mi == None:
        mi = size

    if tau_p == None or tau_n == None:
        tau_p = 0.25 * np.mean(Dbar.data) / size
        tau_n = 0.25 * np.mean(Dbar.data) / size

    symmetric = True

    p_tau = A_p.copy().astype(np.float32)
    n_tau = A_n.copy().astype(np.float32)
    p_tau.data += tau_p
    n_tau.data += tau_n

    Dbar_c = size - Dbar.diagonal()

    Dbar_tau_s = (p_tau + n_tau).sum(axis=0) + (Dbar_c * abs(tau_p - tau_n))[None, :]

    Dbar_tau = sp.diags(Dbar_tau_s.tolist(), [0])

    if normalisation == 'none':
        matrix = A
        delta_tau = tau_p - tau_n

        def mv(v):
            return matrix.dot(v) + delta_tau * v.sum()


    elif normalisation == 'sym':
        d = sqrtinvdiag(Dbar_tau)
        matrix = d * A * d
        dd = d.diagonal()
        tau_dd = (tau_p - tau_n) * dd

        def mv(v):
            return matrix.dot(v) + tau_dd * dd.dot(v)

    elif normalisation == 'sym_sep':

        diag_corr = ss.diags([size * tau_p] * size).tocsc()
        dp = sqrtinvdiag(D_p + diag_corr)

        matrix = dp * A_p * dp

        diag_corr = ss.diags([size * tau_n] * size).tocsc()
        dn = sqrtinvdiag(D_n + diag_corr)

        matrix = matrix - (dn * A_n * dn)

        dpd = dp.diagonal()
        dnd = dn.diagonal()
        tau_dp = tau_p * dpd
        tau_dn = tau_n * dnd

        def mv(v):
            return matrix.dot(v) + tau_dp * dpd.dot(v) - tau_dn * dnd.dot(v)

    else:
        print('Error: choose normalisation')

    matrix_o = sp.linalg.LinearOperator(matrix.shape, matvec=mv)

    if symmetric:
        (w, v) = sp.linalg.eigsh(matrix_o, int(eigens), maxiter=mi, which='LA')
    else:
        (w, v) = sp.linalg.eigs(matrix_o, int(eigens), maxiter=mi, which='LR')

    v = v * w  # weight eigenvalues by eigenvectors, since larger eigenvectors are more likely to be informative
    return v

def VSR(A_p, A_n, label):

    A_power_coo_1 = A_p - A_n # -5300

    A_power_coo_2 = A_p.dot(A_p) - (A_p.dot(A_n) + A_n.dot(A_p))  # sum(), fill_diagonal_(fill_value=0).sum() = (-28294, -30654)
    A_power_coo_3 = A_p.dot(A_p.dot(A_p)) + A_n.dot(A_p.dot(A_n)) - (A_p.dot(A_p.dot(A_n)) + A_p.dot(A_n.dot(A_p)) + A_n.dot(A_p.dot(A_p)))
    A_power_coo_4 = A_p.dot(A_p.dot(A_p.dot(A_p))) - (A_p.dot(A_p.dot(A_p.dot(A_n))) + A_p.dot(A_p.dot(A_n.dot(A_p))) + A_p.dot(A_n.dot(A_p.dot(A_p))) + A_n.dot(A_p.dot(A_p.dot(A_p))))

    A_new = A_power_coo_1 + 0.5 * A_power_coo_2 + (1/6) * A_power_coo_3 + (1-1/2-1/6) * A_power_coo_4# -19447 -20627
    # A_new = A_power_coo_1 + 0.5 * A_power_coo_2 + (1-1/2-1/6) * A_power_coo_3
    ind_A_new = A_new.nonzero()
    A_new_coo_p = torch.sparse_coo_tensor(torch.LongTensor(ind_A_new), torch.FloatTensor(np.array(A_new[ind_A_new]).flatten()), size=A_new.shape).to_dense()
    A_new_coo_p.fill_diagonal_(fill_value=0)
    print(A_new_coo_p.max(), A_new_coo_p.min())


    # A_p[A_new_coo_p > 1] = 1  # 16; 14; 9;8;5--2
    A_n[A_new_coo_p < -2] = 1  # 39,38,37,36,33,

    return A_p, A_n




    # A_new_coo_n.fill_diagonal_(fill_value=0)
    # A_new_coo_p.fill_diagonal_(fill_value=0)
    # A_new_coo_p[A_new_coo_p > 0] = 1 # -17921
    # A_new_coo_p[A_new_coo_p < 0] = 0 # 7842.
    # A_new_coo_n[A_new_coo_n < 0] = -1  # -37556
    # A_new_coo_n[A_new_coo_n > 0] = 0  # -42692
    # new_A = sp.csc_matrix(A_new_coo_p - A_new_coo_n)
    # org_A = A_p + A_n
    # cha = org_A - new_A # 删除了 unbalanced tranigles entry是1， 添加的边是-1.
    # print(new_A.sum(), org_A.sum(), cha.max(), cha.min())
    # cha[cha > 0] = 1  # -17921 # 提取删除的边
    # cha[cha < 0] = 0
    # print(new_A.sum(), org_A.sum(), cha.max(), cha.min())
    # del_p = cha * A_p # 删除的正边
    # del_n = cha * A_n # 删除的负边
    # print(del_p.sum(), del_n.sum())
    # A_p = A_p - del_p
    # A_n = A_n - del_n
    # print(type(A_p), type(A_new_coo_p))

    # sss = 2
    # if sss == 3:
    #     A_p_s = sp.csc_matrix(del_p)
    #     A_n_a = sp.csc_matrix(del_n)
    #     ind_A_p = A_p_s.nonzero()
    #     ind_A_n = A_n_a.nonzero()
    #     shaping = A_p.shape
    #     A_p = torch.sparse_coo_tensor(torch.LongTensor(ind_A_p), torch.FloatTensor(np.array(A_p_s[ind_A_p]).flatten()), shaping)
    #     A_n = torch.sparse_coo_tensor(torch.LongTensor(ind_A_n), torch.FloatTensor(np.array(A_n_a[ind_A_n]).flatten()), shaping)
        # if 2 + 2 == 4:
        #     print('runing the analysis of edges within/without cluster')
        #     A_p = A_p.to_dense()
        #     A_n = A_n.to_dense()
        #
        #     # print('self.A_n, min, max', self.A_n, torch.min(self.A_n), torch.max(self.A_n))
        #     # ---------------------------------------analysis of edges within/without cluster------------------
        #     with torch.no_grad():
        #         # print('self.A_p, min, max', self.A_p, torch.min(self.A_p), torch.max(self.A_p))
        #         A_p_cp = A_p
        #         A_n_cp = A_n
        #         ij_0_p, ij_1_p, ij_2_p, ij_3_p, ij_4_p = 0, 0, 0, 0, 0
        #         ij_0_n, ij_1_n, ij_2_n, ij_3_n, ij_4_n = 0, 0, 0, 0, 0
        #         n_01, n_02, n_03, n_04, n_10, n_12, n_13, n_14, n_21, n_20, n_23, n_24, n_31, n_32, n_30, n_34, n_41, n_42, n_43, n_40 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        #         for i in range(A_p.shape[0]):
        #             for j in range(A_p.shape[1]):
        #                 if label[i] == 0 and label[j] == 0:
        #                     ij_0_p += A_p[i][j]
        #                     # if self.A_n[i][j] > 0:
        #                     #     print('[',i,j,']')
        #                     ij_0_n += A_n[i][j]
        #                 # if label[i] == 0 and label[j] == 1:
        #                 #     n_01 += self.A_n[i][j]
        #                 # if label[i] == 0 and label[j] == 2:
        #                 #     n_02 += self.A_n[i][j]
        #                 # if label[i] == 0 and label[j] == 3:
        #                 #     n_03 += self.A_n[i][j]
        #                 # if label[i] == 0 and label[j] == 4:
        #                 #     n_04 += self.A_n[i][j]
        #                 if label[i] == 1 and label[j] == 1:
        #                     ij_1_p += A_p[i][j]
        #                     ij_1_n += A_n[i][j]
        #                 # if label[i] == 1 and label[j] == 2:
        #                 #     n_12 += self.A_n[i][j]
        #                 # if label[i] == 1 and label[j] == 3:
        #                 #     n_13 += self.A_n[i][j]
        #                 # if label[i] == 1 and label[j] == 4:
        #                 #     n_14 += self.A_n[i][j]
        #                 # if label[i] == 1 and label[j] == 0:
        #                 #     n_10 += self.A_n[i][j]
        #                 if label[i] == 2 and label[j] == 2:
        #                     ij_2_p += A_p[i][j]
        #                     ij_2_n += A_n[i][j]
        #                 # if label[i] == 2 and label[j] == 3:
        #                 #     n_23 += self.A_n[i][j]
        #                 # if label[i] == 2 and label[j] == 4:
        #                 #     n_24 += self.A_n[i][j]
        #                 # if label[i] == 2 and label[j] == 0:
        #                 #     n_20 += self.A_n[i][j]
        #                 # if label[i] == 2 and label[j] == 1:
        #                 #     n_21 += self.A_n[i][j]
        #                 if label[i] == 3 and label[j] == 3:
        #                     ij_3_p += A_p[i][j]
        #                     ij_3_n += A_n[i][j]
        #                 # if label[i] == 3 and label[j] == 4:
        #                 #     n_34 += self.A_n[i][j]
        #                 # if label[i] == 3 and label[j] == 0:
        #                 #     n_30 += self.A_n[i][j]
        #                 # if label[i] == 3 and label[j] == 1:
        #                 #     n_31 += self.A_n[i][j]
        #                 # if label[i] == 3 and label[j] == 2:
        #                 #     n_32 += self.A_n[i][j]
        #                 if label[i] == 4 and label[j] == 4:
        #                     ij_4_p += A_p[i][j]
        #                     ij_4_n += A_n[i][j]
        #                 # if label[i] == 4 and label[j] == 0:
        #                 #     n_40 += self.A_n[i][j]
        #                 # if label[i] == 4 and label[j] == 1:
        #                 #     n_41 += self.A_n[i][j]
        #                 # if label[i] == 4 and label[j] == 2:
        #                 #     n_42 += self.A_n[i][j]
        #                 # if label[i] == 4 and label[j] == 3:
        #                 #     n_43 += self.A_n[i][j]
        #         edge_sum_p_within = ij_0_p + ij_1_p + ij_2_p + ij_3_p + ij_4_p
        #         edge_sum_n_within = ij_0_n + ij_1_n + ij_2_n + ij_3_n + ij_4_n
        #         print('edge_0_1_2_3_4_5_p_within', ij_0_p, ij_1_p, ij_2_p, ij_3_p, ij_4_p)
        #         print('edge_0_1_2_3_4_5_n_within', ij_0_n, ij_1_n, ij_2_n, ij_3_n, ij_4_n)
        #         print('edge_sum_p_in', edge_sum_p_within)
        #         print('edge_sum_n_in', edge_sum_n_within)
        #         print('edge_sum_p_between', A_p_cp.sum() - edge_sum_p_within)
        #         print('edge_sum_n_between', A_n_cp.sum() - edge_sum_n_within)
        #         print('ratio of edge_sum_p_within to edge_sum_n_without',
        #               edge_sum_p_within / (A_n_cp.sum() - edge_sum_n_within))
                # print(n_01+n_10, n_02+n_20, n_03+n_30, n_04+n_40, n_12+n_21, n_13+n_31, n_14+n_41, n_23+n_32, n_24+n_42, n_34+n_43)
                # print(A_p_py_0)

    # return A_p, A_n


def get_powers_sparse(A, hop=1, tau=1):
    '''
    function to get adjacency matrix powers
    inputs:
    A: directed adjacency matrix
    hop: the number of hops that would like to be considered for A to have powers.
    tau: the regularization parameter when adding self-loops to an adjacency matrix, i.e. A -> A + tau * I,
        where I is the identity matrix. If tau=0, then we have no self-loops to add.
    output: (torch sparse tensors)
    A_powers: a list of A powers from 0 to hop
    '''
    print('type(A)', type(A))
    shaping = A.shape
    adj0 = sp.eye(shaping[0])
    A_power = A.copy()
    # A_power_dif = A.copy()

    if hop > 1:
        for h in range(2, int(hop)+1):
            A_power = A_power.dot(A)
        #     A_power_dif += A_power
        # A_power = A_power_dif
        ind_A_power = A_power.nonzero()
        A_power_dense = torch.sparse_coo_tensor(torch.LongTensor(ind_A_power), torch.FloatTensor(np.array(A_power[ind_A_power]).flatten()), size=A_power.shape).to_dense()
        A_power_dense[A_power_dense !=0] = 1
        A_power_dense.fill_diagonal_(fill_value=0)
        A_power_csc = sp.csc_matrix(A_power_dense)
        A_bar = normalize(A_power_csc + tau * adj0, norm='l1')   # coo-tensor; dense; csc_matrix; coo-tensor
        ind_power = A_power_csc.nonzero()
        A_powers_norm = torch.sparse_coo_tensor(torch.LongTensor(ind_power), torch.FloatTensor(np.array(A_bar[ind_power]).flatten()), shaping)
    else:
        A_power_csc = sp.csc_matrix(A_power)
        A_bar = normalize(A + tau * adj0, norm='l1')  # l1 row normalization # tau, default: 0.5
        ind_power = A.nonzero()
        A_powers_norm = torch.sparse_coo_tensor(torch.LongTensor(ind_power), torch.FloatTensor(np.array(A_bar[ind_power]).flatten()), shaping)
    # print('type(A_power_csc)', type(A_power_csc.tocsr()))

    return A_powers_norm, A_power_csc


def get_powers_sparse_neg(A_n, A_p, hop=1, tau=1):
    '''
    function to get adjacency matrix powers
    inputs:
    A: directed adjacency matrix
    hop: the number of hops that would like to be considered for A to have powers.
    tau: the regularization parameter when adding self-loops to an adjacency matrix, i.e. A -> A + tau * I,
        where I is the identity matrix. If tau=0, then we have no self-loops to add.
    output: (torch sparse tensors)
    A_powers: a list of A powers from 0 to hop
    '''

    shaping = A_n.shape
    adj0 = sp.eye(shaping[0])
    A_n_power = A_n.copy()
    A_power_dif = A_n.copy()

    if hop > 1:
        for h in range(2, int(hop)+1):
            A_n_power = A_p.dot(A_n_power)
            # A_n_power = A_n_power.dot(A_p) + A_p.dot(A_n_power)
        #     A_power_dif += A_n_power
        # A_n_power = A_power_dif
        ind_A_power = A_n_power.nonzero()
        A_power_dense = torch.sparse_coo_tensor(torch.LongTensor(ind_A_power), torch.FloatTensor(np.array(A_n_power[ind_A_power]).flatten()), size=A_n_power.shape).to_dense()
        A_power_dense[A_power_dense !=0] = 1
        A_power_dense.fill_diagonal_(fill_value=0)
        A_power_csc = sp.csc_matrix(A_power_dense)
        A_bar = normalize(A_power_csc + tau * adj0, norm='l1')   # coo-tensor; dense; csc_matrix; coo-tensor
        ind_power = A_power_csc.nonzero()
        A_powers_norm = torch.sparse_coo_tensor(torch.LongTensor(ind_power), torch.FloatTensor(np.array(A_bar[ind_power]).flatten()), shaping)
    else:
        A_power_csc = sp.csc_matrix(A_n_power)
        A_bar = normalize(A_n + tau * adj0, norm='l1')  # l1 row normalization # tau, default: 0.5
        ind_power = A_n.nonzero()
        A_powers_norm = torch.sparse_coo_tensor(torch.LongTensor(ind_power), torch.FloatTensor(np.array(A_bar[ind_power]).flatten()), shaping)

    return A_powers_norm, A_power_csc

def scipy_sparse_to_torch_sparse(A):
    A = sp.csr_matrix(A)
    return torch.sparse_coo_tensor(torch.LongTensor(A.nonzero()), torch.FloatTensor(A.data), A.shape)

def write_log(args, path):
    with open(path+'/settings.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for para in args:
            writer.writerow([para, args[para]])
    return

def extract_edges(edge_index, valid_nodes):
    edge_index = torch.LongTensor(edge_index)
    ind_list = []
    for i in range(edge_index.shape[1]):
        ii, jj = edge_index[:,i]
        if ii.item() in valid_nodes and jj.item() in valid_nodes:
            ind_list.append(i)
    return edge_index[:,ind_list]