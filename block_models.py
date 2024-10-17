import numpy as np
import scipy.sparse as ss
import numpy.random as rnd
import math
import networkx as nx

np.set_printoptions(2)
rnd.seed(10)
np.random.seed(10)


def SSBM(n, k, pin, etain, pout=None, etaout=None, values='ones', sizes='uniform'):
    """A signed stochastic block model graph generator.

    Args:
        n: (int) Number of nodes.
        k: (int) Number of communities.
        pin: (float) Sparsity value within communities.
        etain: (float) Noise value within communities.
        pout: (float) Sparsity value between communities.
        etaout: (float) Noise value between communities.
        values: (string) Edge weight distribution (within community and without sign flip; otherwise weight is negated):
            'ones': Weights are 1.
            'gaussian': Weights are Gaussian, with variance 1 and expectation of 1.#
            'exp': Weights are exponentially distributed, with parameter 1.
            'uniform: Weights are uniformly distributed between 0 and 1.
        sizes: (string) How to generate community sizes:
            'uniform': All communities are the same size (up to rounding).
            'random': Nodes are assigned to communities at random.
            'uneven': Communities are given affinities uniformly at random, and nodes are randomly assigned to communities weighted by their affinity.

    Returns:
        (a,b),c where a is a sparse n by n matrix of positive edges, b is a sparse n by n matrix of negative edges c is an array of cluster membership.

    """
    print('ppppppppppppppppp', pin, pout, etain, etaout)

    if pout == None:
        pout = pin
    if etaout == None:
        etaout = etain
    # print('original', etain, pout) # 0.02/0.01
    etain, pout = 0.45, 0.035
    print('after', etain, pout)
    rndinrange = math.floor(n * n * pin / 2 + n)
    # print(rndinrange) # 6000 簇内应该有6000条边
    rndin = rnd.geometric(pin, size=rndinrange) # 6000 fffffff
    # print('iiiiiiiiiiiiii', rndin.shape) # 1144
    flipinrange = math.floor(n * n / 2 * pin + n) # 簇内正边个数
    flipin = rnd.binomial(1, etain, size=flipinrange) # 簇内正边翻转为负边
    # print('fffffffff', flipin.shape, flipin)
    rndoutrange = math.floor(n * n / 2 * pout + n)
    rndout = rnd.geometric(pout, size=rndoutrange) # ffffffff
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
    # print('last', last) # [199, 399, 599, 799, 999]
    # print('assign', assign) # [1,3,2,0,4,2,1....] labels
    # print(perm)


    pdat = []
    prow = []
    pcol = []
    ndat = []
    nrow = []
    ncol = []
    cc = 0
    cc_1 = 0
    aa = 0
    aa_1 = 0
    for x in range(n):
        me = perm[x] # 节点编号
        y = x + rndin[ricount] # 6000[]
        ricount += 1
        while y <= last[assign[me]]:    # 标签编号      -1<assign[me]<5
            val = fill(values)
            if flipin[ficount] == 1:
                cc += 1
                ndat.append(val) #
                ndat.append(val)   # 簇内负边： flipin （少数）
                ncol.append(me)
                ncol.append(perm[y])
                nrow.append(perm[y])
                nrow.append(me)
                # print(me, perm[y], assign[me], assign[perm[y]])
            else:
                pdat.append(val)
                pdat.append(val)
                pcol.append(me)
                pcol.append(perm[y]) # 簇内正边： pin
                prow.append(perm[y])
                prow.append(me)
                # print(me, perm[y], assign[me], assign[perm[y]])
                cc_1 += 1

            ficount += 1
            y += rndin[ricount]
            ricount += 1
        y = last[assign[me]] + rndout[rocount]  # 200 +
        rocount += 1
        while y < n:                    # 3333333333
            val = fill(values)
            if flipout[focount] != 1:
                ndat.append(val)
                ndat.append(val)
                ncol.append(me)
                ncol.append(perm[y])   # 簇间 负边
                nrow.append(perm[y])
                nrow.append(me)
                aa += 1
                # print(me, perm[y], assign[me], assign[perm[y]])
            else:
                pdat.append(val)
                pdat.append(val)
                pcol.append(me)
                pcol.append(perm[y])    # 簇间 正边 （少数）
                prow.append(perm[y])
                prow.append(me)
                aa_1 += 1
                # print(me, perm[y], assign[me], assign[perm[y]])
            focount += 1
            y += rndout[rocount]
            rocount += 1
    print('cc, cc_1', cc, cc_1) # 17, 996
    print('aa, aa_1', aa, aa_1)
    print('ppppppppppppppppp', pin, pout, etain, etaout)
    return (ss.coo_matrix((pdat, (prow, pcol)), shape=(n, n)).tocsc(), ss.coo_matrix((ndat, (nrow, ncol)), shape=(n, n)).tocsc()), assign  # positive matrix, negative matrix, labels


def SBAM(n, k, p, eta):
    """A signed Barabási–Albert model graph generator.

    Args:
        n: (int) Number of nodes.
        k: (int) Number of communities.
        p: (float) Sparsity value.
        eta: (float) Noise value.

    Returns:
        (a,b),c where a is a sparse n by n matrix of positive edges, b is a sparse n by n matrix of negative edges c is an array of cluster membership.

    """

    # correspondence between m and p (by equating mean degree)
    m = int(n * p / 2)
    # generate a positive BA graph
    net = nx.barabasi_albert_graph(n=n, m=m, seed=None)

    ndk = int(n / k)

    # set signed, noisy community structure

    truth = np.repeat(np.arange(k - 1), ndk)
    truth = np.hstack((truth, (k - 1) * np.ones(n - ndk * (k - 1))))

    for u, v, d in net.edges(data=True):
        rnd = np.random.uniform()

        if truth[u] == truth[v]:
            if rnd >= eta:
                d['weight'] = 1
            else:
                d['weight'] = -1
        else:
            if rnd >= eta:
                d['weight'] = -1
            else:
                d['weight'] = 1

    truth = truth[net.nodes()]
    A = nx.to_scipy_sparse_matrix(net, format='csc')

    Abar = abs(A)
    A_p = (A + Abar) / 2
    A_n = -(A - Abar) / 2
    A_p.eliminate_zeros()
    A_n.eliminate_zeros()

    return (A_p, A_n), truth


def SRBM(n, k, p, eta):
    """A signed regular graph model generator.

    Args:
        n: (int) Number of nodes.
        k: (int) Number of communities.
        p: (float) Sparsity value.
        eta: (float) Noise value.

    Returns:
        (a,b),c where a is a sparse n by n matrix of positive edges, b is a sparse n by n matrix of negative edges c is an array of cluster membership.

    """

    c = int(n * p)
    net = nx.random_regular_graph(n=n, d=c)
    ndk = int(n / k)

    # set signed, noisy community structure

    truth = np.repeat(np.arange(k - 1), ndk)
    truth = np.hstack((truth, (k - 1) * np.ones(n - ndk * (k - 1))))

    for u, v, d in net.edges(data=True):

        rnd = np.random.uniform()
        if truth[u] == truth[v]:
            if rnd >= eta:
                d['weight'] = 1
            else:
                d['weight'] = -1
        else:
            if rnd >= eta:
                d['weight'] = -1
            else:
                d['weight'] = 1

    truth = truth[net.nodes()]
    A = nx.to_scipy_sparse_matrix(net, format='csc')
    Abar = abs(A)

    A_p = (A + Abar) / 2
    A_n = -(A - Abar) / 2
    A_p.eliminate_zeros()
    A_n.eliminate_zeros()

    return (A_p, A_n), truth


def fill(values='ones'):
    if values == 'ones':
        return float(1)
    elif values == 'gaussian':
        return np.random.normal(1)
    elif values == 'exp':
        return np.random.exponential()
    elif values == 'uniform':
        return np.random.uniform()


if __name__ == '__main__':
    (p, n), t = SBAM(n=100, k=3, p=0.1, eta=0.1)
    print(p.todense())
