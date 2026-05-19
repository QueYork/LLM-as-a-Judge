import os
import sys
from os.path import join
import warnings
warnings.filterwarnings('ignore')

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = join(PATH, '../')
sys.path.append(ROOT)
import src.data_loader as Data_Loader
import numpy as np
import pandas as pd

import src.powerboard as board
import src.utils as utils
import src.evals as evals
import src.model as model

MODEL = {
    'gcn': model.LightGCN,
    'mf': model.MF,
    'lrgccf': model.LRGCCF,
    'ngcf': model.NGCF,
    'neumf': model.NeuMF,
    'mgccf': model.MGCCF,
    'simgcl': model.SimGCL,
    'gtn': model.GTN,
    'caged': model.LightGCN,
}

def main():
    dataset = Data_Loader.LoadData(data_name=board.args.dataset)
    rec_model = MODEL[board.args.model](dataset=dataset).to(board.DEVICE)
    
    all_results = {
        'recall': [],
        'ndcg': []
    }
    
    for seed in board.SEED:
        utils.set_seed(seed)
        evals.load_model(rec_model, seed)
        results = evals.Inference(dataset=dataset, model=rec_model, epoch=0, summarizer=None)
        all_results['recall'].append(results['recall'][:3])  
        all_results['ndcg'].append(results['ndcg'][:3])
    
    recall_array = np.array(all_results['recall'])
    ndcg_array = np.array(all_results['ndcg'])

    mean_recall = np.mean(recall_array, axis=0)
    std_recall  = np.std(recall_array, axis=0, ddof=1)
    mean_ndcg   = np.mean(ndcg_array, axis=0)
    std_ndcg    = np.std(ndcg_array, axis=0, ddof=1)

    df = pd.DataFrame({
        f'r{board.args.topks[0]}_mean': [mean_recall[0]],
        f'r{board.args.topks[0]}_std': [std_recall[0]],
        f'n{board.args.topks[0]}_mean': [mean_ndcg[0]],
        f'n{board.args.topks[0]}_std': [std_ndcg[0]],
        f'r{board.args.topks[1]}_mean': [mean_recall[1]],
        f'r{board.args.topks[1]}_std': [std_recall[1]],
        f'n{board.args.topks[1]}_mean': [mean_ndcg[1]],
        f'n{board.args.topks[1]}_std': [std_ndcg[1]],
        f'r{board.args.topks[2]}_mean': [mean_recall[2]],
        f'r{board.args.topks[2]}_std': [std_recall[2]],
        f'n{board.args.topks[2]}_mean': [mean_ndcg[2]],
        f'n{board.args.topks[2]}_std': [std_ndcg[2]],
    })
    print(df)
      
      
if __name__ == '__main__':
    main()
    