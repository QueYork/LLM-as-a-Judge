import os
import sys
from os.path import join
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import src.powerboard as board
import src.evals as evals
import src.model as model
import src.data_loader as Data_Loader

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

class LoadRecList(Dataset):
    def __init__(self, dataset: str = 'kuairec', item_caption: str = 'item_caption.csv', user_caption: str = 'user_caption.csv'):
        self.dataset_name = dataset
        self.dataset = Data_Loader.LoadData(data_name=dataset)
        self.reclist_path = join(board.DATA_PATH, dataset, 'rec_list')
        self.ckpt_path = board.FILE_PATH
        self.item_caption = pd.read_csv(join(self.reclist_path, item_caption))
        if dataset == 'coat':
            self.user_caption = pd.read_csv(join(self.reclist_path, user_caption))
        
    def get_rec_list(self, model: str = 'gcn', seed: int = 2021, topk: int = 100):
        reclist_file = join(self.reclist_path, model, f"{model}-{self.dataset_name}-top{topk}-seed{seed}.json")
        
        if os.path.exists(reclist_file):
            print(f"Load RecList from {reclist_file}...")
            with open(reclist_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        rec_model = MODEL[model](dataset=self.dataset).to(board.DEVICE)
        evals.load_model(rec_model, seed)
        rec_model.eval()
        rec_list = evals.get_topk_recommendations(dataset=self.dataset, model=rec_model, max_k=topk)
        
        item2caption = dict(zip(self.item_caption['vid'], self.item_caption['text_input']))
        final_list = []
        
        test_dict = self.dataset.get_test_dict()
        all_users = list(test_dict.keys())
        for user_idx, item_indices in zip(all_users, rec_list):
            if hasattr(item_indices, 'tolist'):
                item_indices = item_indices.tolist()
            
            user_items = []
            for item_id in item_indices:
                cap = item2caption.get(int(item_id), "") 
                
                user_items.append({
                    'item_id': int(item_id),
                    'caption': str(cap)
                })
            
            final_list.append({
                'user_id': int(user_idx),
                'items': user_items
            })

        os.makedirs(os.path.dirname(reclist_file), exist_ok=True)
        print(f"Saving RecList with captions to {reclist_file}...")
        
        with open(reclist_file, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, ensure_ascii=False, indent=4)
            
        return final_list
        
    def get_pos_list(self):
        pos_list_dir = join(self.reclist_path, 'train_pos')
        pos_list_file = join(pos_list_dir, f"train_pos-{self.dataset_name}.json")
        
        if os.path.exists(pos_list_file):
            print(f"Load PosList from {pos_list_file}...")
            with open(pos_list_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        item2caption = dict(zip(self.item_caption['vid'], self.item_caption['text_input']))
        if self.dataset_name == 'coat':
            user2description = dict(zip(self.user_caption['vid'], self.user_caption['text_input']))
            
        all_pos_items = self.dataset.get_all_pos()
        test_dict = self.dataset.get_test_dict()
        all_users = list(test_dict.keys())
        
        final_list = []
        
        print("Generating PosList with captions...")
        for user_idx in all_users:
            item_indices = all_pos_items[user_idx]
            user_items = []
            for item_id in item_indices:
                item_id = int(item_id) 
                cap = item2caption.get(item_id, "") 
                
                user_items.append({
                    'item_id': int(item_id),
                    'caption': str(cap)
                })
            
            user_data = {
                'user_id': int(user_idx),
                'items': user_items
            }
            
            if self.dataset_name == 'coat':
                user_data['description'] = str(user2description.get(int(user_idx), ""))
                
            final_list.append(user_data)
            

        os.makedirs(pos_list_dir, exist_ok=True)
        print(f"Saving PosList to {pos_list_file}...")
        
        with open(pos_list_file, 'w', encoding='utf-8') as f:
            json.dump(final_list, f, ensure_ascii=False, indent=4)
            
        return final_list    
        
