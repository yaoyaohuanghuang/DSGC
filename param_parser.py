import argparse
import os
import pickle as pk
import torch

torch.manual_seed(10)
torch.cuda.manual_seed_all(10)
torch.backends.cudnn.deterministic = True

with open('.../yearly_dict.pk', 'rb') as handle:
    yearly_dict = pk.load(handle)


def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run SSSNET.")

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--debug', '-D', action='store_true', default=False,
                        help='Debugging mode, minimal setting.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='training ratio during data split.')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='test ratio during data split.')
    parser.add_argument('--set_seed', type=int, default=10, help='Random seed.') # 10
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument("--all_methods",
                        nargs="+",
                        type=str,
                        help="Methods to use.")
    parser.set_defaults(all_methods=['spectral', 'DSGC'])

    # synthetic model hyperparameters below
    parser.add_argument("--seeds",
                        nargs="+",
                        type=int,
                        help="seeds to generate random graphs.")
    parser.set_defaults(seeds=[10])  # [10, 20, 30, 40, 50]

    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of maximum epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,  # default = 0.01
                        help='Initial learning rate.')

    parser.add_argument('--N', type=int, default=1000,  # 1000
                        help='number of nodes in the signed stochastic block model.')
    parser.add_argument('--K', type=int, default=5,  # 2
                        help=' number of blocks in each SSBM.')
    parser.add_argument('--p', type=float, default=0.01,  # 0.02
                        help='probability of the existence of a link within communities, with probability (1-p), we have 0.')
    parser.add_argument('--eta', type=float, default=0,  # 0.1
                        help='direction noise level in the meta-graph adjacency matrix, less than 0.5.')
    parser.add_argument('--div_alpha', type=float, default=0.01, help='diversity term in NSVGD.')
    parser.add_argument('--pbnc_loss', type=float, default=1,
                        help='Ratio of pbnc loss to reconstruction loss in total loss. Default 0.1.')

    parser.add_argument('--total_n', type=int, default=1050,
                        help='total number of nodes in the polarized network.')
    parser.add_argument('--num_com', type=int, default=2,
                        help='number of polarized communities (SSBMs).')
    parser.add_argument('--hop', type=int, default=2,
                        help='Number of hops to consider for the random walk.')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='the regularization parameter when adding self-loops to the positive part of adjacency matrix, i.e. A -> A + tau * I, where I is the identity matrix.')
    parser.add_argument('--seed_ratio', type=float, default=0.1,
                        help='The ratio in the training set of each cluster to serve as seed nodes.')
    parser.add_argument('--size_ratio', type=float, default=1.5,
                        help='The size ratio of the largest to the smallest block. 1 means uniform sizes. should be at least 1.')
    parser.add_argument('--num_trials', type=int, default=2,
                        help='Number of trials to generate results.')

    parser.add_argument('--early_stopping', type=int, default=100,
                        help='Number of iterations to consider for early stopping.')
    parser.add_argument('--directed', action='store_true', help='Directed input graph.')
    parser.add_argument('--no_validation', action='store_true',
                        help='Whether to disable validation and early stopping during traing.')
    parser.add_argument('--regenerate_data', action='store_true', help='Whether to force creation of data splits.')
    parser.add_argument('--load_only', action='store_true', help='Whether not to store generated data.')
    parser.add_argument('--dense', action='store_true', help='Whether not to use torch sparse.')
    

    parser.add_argument('--alpha', type=float, default=0,
                        help='Threshold in triplet loss for seeds.')
    # data loading and logs
    parser.add_argument('--log_root', type=str,
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../logs/'),
                        help='the path saving model.t7 and the training process')
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/'),
                        help='data set folder, for default format see dataset/cora/cora.edges and cora.node_labels')
    parser.add_argument('--dataset', type=str, default='SSBM/', help='data set selection')
    parser.add_argument('--year_index', type=int, default=2,
                        help='Index of the year when using yearly data.')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    if args.dataset[:9].lower() == 'mr_yearly':
        args.dataset = yearly_dict[args.year_index]
    if args.dataset[-1] != '/':
        args.dataset += '/'
    if args.loss_ratio == -1:
        args.w_pbnc = 1
        args.w_pbrc = 0
    else:
        args.w_pbrc = 1
        # args.w_pbnc = args.loss_ratio
        args.w_pbnc = 0
    if args.no_validation:
        args.train_ratio = 1 - args.test_ratio
    if args.debug:
        args.epochs = 2
        args.num_trials = 2
        args.seeds = [10, 20]
        args.log_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../debug_logs/')
    return args
