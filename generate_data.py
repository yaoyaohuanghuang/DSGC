import os
import random
import math

import torch
import numpy as np
import scipy.sparse as sp
import pickle as pk
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import numpy.random as rnd

from param_parser import parameter_parser
from utils import polarized_ssbm, SSBM, fix_network, spectral_adjacency_reg, signed_Laplacian_features


def to_dataset(args, A_p, A_n, label, save_path, feat_given=None, conflict_groups=None, load_only=False):
    label = label - np.amin(label)
    num_clusters = int(np.amax(label) + 1)
    label = torch.from_numpy(label).long()

    feat_adj_reg = spectral_adjacency_reg(A_p, A_n, num_clusters)
    feat_L = signed_Laplacian_features(A_p, A_n, num_clusters)
    data = Data(y=label, A_p=A_p, A_n=A_n, feat_adj_reg=feat_adj_reg, feat_L=feat_L,
                feat_given=feat_given, conflict_groups=conflict_groups)
    if not load_only:
        dir_name = os.path.dirname(save_path)
        if os.path.isdir(dir_name) == False:
            try:
                os.makedirs(dir_name)
            except FileExistsError:
                print('Folder exists!')
        pk.dump(data, open(save_path, 'wb'))
    return data


def main():
    args = parameter_parser()
    rnd.seed(args.seed)
    random.seed(args.seed)
    if args.dataset[-1] != '/':
        args.dataset += '/'
    if args.dataset == 'SSBM/':
        (A_p, A_n), labels = SSBM(n=args.N, k=args.K, pin=args.p, pout=None, etain=args.eta, sizes='fix_ratio',
                                  size_ratio=args.size_ratio)
        (A_p, A_n), labels = fix_network(A_p, A_n, labels, eta=args.eta)
        conflict_groups = None
        default_values = [args.p, args.K, args.N, args.seed_ratio, args.train_ratio, args.test_ratio, args.size_ratio,
                          args.eta, args.num_trials]
    elif args.dataset == 'polarized/':
        (A_p, A_n), labels, conflict_groups = polarized_ssbm(total_n=args.total_n, num_com=args.num_com, N=args.N,
                                                             K=args.K, p=args.p, eta=args.eta,
                                                             size_ratio=args.size_ratio)
        default_values = [args.total_n, args.num_com, args.p, args.K, args.N, args.seed_ratio, args.train_ratio,
                          args.test_ratio, args.size_ratio, args.eta, args.num_trials]
    default_name_base = '_'.join([str(int(100 * value)) for value in default_values])
    if args.seed != 31:
        default_name_base = 'Seed' + str(args.seed) + '_' + default_name_base
    save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../data/' + args.dataset + default_name_base + '.pk')
    _ = to_dataset(args, A_p, A_n, labels, save_path=save_path, conflict_groups=conflict_groups)
    return


if __name__ == "__main__":
    main()