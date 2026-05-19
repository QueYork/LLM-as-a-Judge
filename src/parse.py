import argparse

def parse_args():
    parse = argparse.ArgumentParser(description='Lossless')

    parse.add_argument('--dataset', type=str, default='coat', help='dataset: kuairec, coat')
    parse.add_argument('--tensorboard', type=int, default=1, help='enable tensorboard')
    parse.add_argument('--train_file', type=str, default='train.txt')
    parse.add_argument('--test_file', type=str, default='small_matrix.txt')
    parse.add_argument('--val_file', type=str, default='val.txt')
    parse.add_argument('--mode', type=str, default='val', help='test mode: val, test, or normal, deciding the range of candidate items.')
    parse.add_argument('--train_batch', type=int, default=4096, help='batch size in training')
    parse.add_argument('--test_batch', type=int, default=100, help='batch size in testing')                         
    parse.add_argument('--neg_ratio', type=int, default=1, help='the number of negative sampling')
    parse.add_argument('--seed', nargs='+', type=int, default=[2021], help='random seed')
    parse.add_argument('--topks', nargs='+', type=int, default=[20, 50, 100], help='top@k test list') 
    
    parse.add_argument('--save', type=bool, default=False, help='enable ckpt saving')
    parse.add_argument('--model', type=str, default='gcn', help='models to be trained from [gcn, mf, lrgccf, ngcf, neumf, mgccf, simgcl, gtn, caged]')  
    parse.add_argument('--layers', type=int, default=2, help='the layer number')
    parse.add_argument('--dim', type=int, default=256, help='embedding dimension')
    parse.add_argument('--epoch', type=int, default=1000, help='epoch of lightgcn training')
    parse.add_argument('--lr', type=float, default=5e-4, help='learning rate of lightgcn')
    parse.add_argument('--weight', type=float, default=1e-4, help='the weight of l2 norm')     
    parse.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    
    # SimGCL parameters
    parse.add_argument('--eps', type=float, default=0, help='simgcl noise scale') 
    parse.add_argument('--lam', type=float, default=0, help='simgcl cl loss weight') 
    
    # CAGED parameters
    parse.add_argument('--latent_dim', type=int, default=256, help='dimension of latent variable z')
    parse.add_argument('--lr2', type=float, default=5e-4, help='learning rate of caged')
    parse.add_argument('--hidden_dim', nargs='+', type=int, default=[512, 1024, 512], help='caged neuron number')
    parse.add_argument('--sc_var', type=float, default=1, help='caged score variance')
    parse.add_argument('--eps_decay', type=float, default=5e-4, help='epsilon decay ratio')
    parse.add_argument('--beta', type=float, default=1, help='kl score amplifier')
    
    return parse.parse_known_args()