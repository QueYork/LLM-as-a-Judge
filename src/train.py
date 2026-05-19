import os
import sys
from os.path import join
import warnings
warnings.filterwarnings('ignore')

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = join(PATH, '../')
sys.path.append(ROOT)

from torch.utils.tensorboard import SummaryWriter
import src.data_loader as Data_Loader
import datetime
import pytz
import torch
import numpy as np
import logging
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
    'caged': model.caged,
}

def main(seed):
    utils.set_seed(seed)
    print('--SEED--:', seed)

    # Load dataset
    dataset = Data_Loader.LoadData(data_name=board.args.dataset)

    # log file path
    path = join(board.BOARD_PATH, board.args.dataset, board.args.model)
    timezone = pytz.timezone('Asia/Shanghai')
    nowtime = datetime.datetime.now(tz=timezone)
    log_path = join(path, nowtime.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + board.args.model)

    # logger initializer
    log_name = utils.create_log_name(log_path)
    utils.log_config(path=log_path, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info(board.args)
    
    # init tensorboard
    if board.args.tensorboard:
        summarizer: SummaryWriter = SummaryWriter(log_path)
    else:
        summarizer = None
    
    # caged training process
    if board.args.model == 'caged':
        u_neighbors = dataset.get_neighbors()
        dataset_caged = torch.utils.data.TensorDataset(u_neighbors[:, 0], u_neighbors[:, 1])
        data_loader = torch.utils.data.DataLoader(dataset_caged, batch_size=board.args.train_batch, 
                                                shuffle=True, generator=torch.Generator().manual_seed(seed))

        # Initialize lightgcn
        gcn = MODEL['gcn'](dataset=dataset).to(board.DEVICE)
        loss_gcn = evals.Loss(gcn)
        
        # Initialize caged 
        caged = MODEL['caged']()
        caged = caged.to(board.DEVICE)
        loss_caged = evals.cagedLoss(caged)
        
        max_recall20 = 0
        max_info = None
        
        eps_decay = board.args.eps_decay
        weight_graph = dataset.get_root_degree() * dataset.Graph.to(board.DEVICE)
        for epoch in range(board.args.epoch):
            """
            ********************* GCN Train **************************
            """
            info = evals.Train_full(dataset=dataset, model=gcn, epoch=epoch, loss_f=loss_gcn, 
                                    neg_ratio=board.args.neg_ratio, summarizer=summarizer)

            board.cprint(f'[testing at epoch-{epoch}]')
            results = evals.Inference(dataset=dataset, model=gcn, epoch=epoch, summarizer=summarizer)

            logging.info(f'[testing at epoch-{epoch}]')
            logging.info(results)

            logging.info(f'EPOCH[{epoch + 1}/{board.args.epoch}] {info} ')
            
            if max_recall20 < results['recall'][0]:
                max_recall20 = results['recall'][0]
                max_info = results
                logging.info(f'Summary at recall = {max_recall20}')
                if board.args.save:
                    evals.save_model(gcn, seed)
                """
                ********************* CAGED Train (only enable when making progress) **************************
                """
                gcn.eval()
                
                with torch.no_grad():
                    user_emb, item_emb = gcn.aggregate_embed()
                    user_emb = gcn.pooling(user_emb)
                    item_emb = gcn.pooling(item_emb)
                
                info = evals.Train_caged(data_loader=data_loader, caged=caged, user_emb=user_emb, item_emb=item_emb, 
                                        loss_f=loss_caged, batch_size=board.args.train_batch)
                logging.info(f'CAGED [Training] {info} ')
                
                with torch.no_grad():
                    new_graph = (1 - eps_decay) * gcn.Graph + eps_decay * weight_graph * evals.Weight_Inference(caged, user_emb, item_emb, data_loader)
                    del gcn.Graph 
                    torch.cuda.empty_cache()  
                    gcn.Graph = new_graph.coalesce()
                    
        logging.info('Final result (highest recall@20): ')   
        logging.info(max_info)
    
    # regular model training process (except caged)
    else:
        rec_model = MODEL[board.args.model](dataset=dataset).to(board.DEVICE)
        loss = evals.Loss(rec_model)
        
        max_recall20 = 0
        max_info = None
        
        for epoch in range(board.args.epoch):
            """
            ********************* Train **************************
            """
            info = evals.Train_full(dataset=dataset, model=rec_model, epoch=epoch, loss_f=loss, 
                                    neg_ratio=board.args.neg_ratio, summarizer=summarizer)

            """
            ********************* Test **************************
            """
            if epoch % 10 == 0:
                board.cprint(f'[testing at epoch-{epoch}]')
                results = evals.Inference(dataset=dataset, model=rec_model, epoch=epoch, summarizer=summarizer)
                
                
                logging.info(f'[testing at epoch-{epoch}]')
                logging.info(results)

                logging.info(f'EPOCH[{epoch + 1}/{board.args.epoch}] {info} ')
                
                if max_recall20 < results['recall'][0]:
                    max_recall20 = results['recall'][0]
                    max_info = results
                    logging.info(f'Reaching recall = {max_recall20}')
                    
                    if board.args.save:
                        evals.save_model(rec_model, seed)
        
        logging.info('Final result (highest recall@20): ')   
        logging.info(max_info)
    
    if board.args.tensorboard:
        summarizer.close()
      
      
if __name__ == '__main__':
    for seed in board.SEED:
        main(seed)
    