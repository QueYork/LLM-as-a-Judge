import os
import sys
from os.path import join
import torch
from src.parse import parse_args

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args, _ = parse_args()
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = join(CODE_PATH, '../preprocess')
C_SOURCE_PATH = join(CODE_PATH, 'sources')
BOARD_PATH = join(CODE_PATH, 'results')
FILE_PATH = join(CODE_PATH, 'checkpoints')
sys.path.append(C_SOURCE_PATH)

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


all_dataset = ['kuairec', 'coat']
GPU = torch.cuda.is_available()
SEED = args.seed
DEVICE = torch.device('cuda' if GPU else 'cpu')

if args.dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {args.dataset} yet!, try {all_dataset}")

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")