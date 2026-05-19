import os
import sys
from os.path import join
import torch
from judge.parse import parse_args
from pprint import pprint

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args, _ = parse_args()
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
RECLIST_PATH = join(CODE_PATH, '../preprocess', args.dataset, 'rec_list')
CKPT_PATH = join(CODE_PATH, '../src/checkpoints')
DATA_PATH = join(CODE_PATH, 'results')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH, exist_ok=True)

API_KEY = args.api_key       
BASE_URL = args.base_url     
MODEL_NAME = args.llm_name   

GPU = torch.cuda.is_available()
SEED = args.seed
DEVICE = torch.device('cuda' if GPU else 'cpu')

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=RuntimeWarning)

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

print('===========config================')
pprint(args)
print('===========end===================')