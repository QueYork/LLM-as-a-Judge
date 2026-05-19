import os
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
import src.powerboard as board
from time import time


class LoadData(Dataset):
    def __init__(self, data_name='kuairec'):
        path = join(board.DATA_PATH, data_name, 'cleaned_data')
        # print(f'dataset loading [{path}]')
        self.mode_dict = {'train': 0, 'test': 1}
        self.mode = self.mode_dict['train']
        self.path = path
        
        # File paths
        train_file = join(self.path, board.args.train_file)
        test_file = join(self.path, board.args.test_file)
        val_file = join(self.path, board.args.val_file) 
        
        self.n_users = 0
        self.m_items = 0
        self.data_name = data_name

        # Containers
        train_unique_users, train_users, train_items = [], [], []
        test_unique_users, test_users, test_items = [], [], []
        val_unique_users, val_users, val_items = [], [], [] 
        
        self.train_instance_size = 0
        self.test_instance_size = 0
        self.val_instance_size = 0 

        # --- Load Train ---
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip()
                    l = l.strip('\n').split(' ')

                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    train_unique_users.append(uid)
                    train_users.extend([uid] * len(items))
                    train_items.extend(items)
                    self.n_users = max(self.n_users, uid)
                    self.m_items = max(self.m_items, max(items))
                    self.train_instance_size += len(items)
        self.train_unique_users = np.array(train_unique_users)
        self.train_users = np.array(train_users)
        self.train_items = np.array(train_items)

        # --- Load Test ---
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) > 1:
                        uid = int(l[0])
                        items = [int(i) for i in l[1:]]
                        test_unique_users.append(uid)
                        test_users.extend([uid] * len(items))
                        test_items.extend(items)
                        self.n_users = max(self.n_users, uid)
                        self.m_items = max(self.m_items, max(items))
                        self.test_instance_size += len(items)
                    else:
                        raise NotImplementedError(uid)
        self.test_unique_users = np.array(test_unique_users)
        self.test_users = np.array(test_users)
        self.test_items = np.array(test_items)
        self.test_items_unique = np.unique(self.test_items)
        
        # --- Load Val ---
        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if len(l) > 1:
                        uid = int(l[0])
                        items = [int(i) for i in l[1:]]
                        val_unique_users.append(uid)
                        val_users.extend([uid] * len(items))
                        val_items.extend(items)
                        self.n_users = max(self.n_users, uid)
                        self.m_items = max(self.m_items, max(items))
                        self.val_instance_size += len(items)
        self.val_unique_users = np.array(val_unique_users)
        self.val_users = np.array(val_users)
        self.val_items = np.array(val_items)
        
        self.m_items += 1
        self.n_users += 1
        
        if data_name == 'coat':
            self.raw_test_items = {}
            raw_test_file = join(self.path, 'raw_small_matrix.txt')
            with open(raw_test_file, 'r') as f:
                for line in f:
                    row = [int(x) for x in line.strip().split()]
                    user = row[0]
                    items = row[1:]
                    self.raw_test_items[user] = items
        elif data_name == 'kuairec':
            self.non_test_items = np.setdiff1d(np.arange(self.m_items), self.test_items_unique)
       
        self.Graph = None
        # print(f'{self.n_users} users')
        # print(f'{self.m_items} items')
        # print(f'{self.train_instance_size} interactions for training')
        # print(f'{self.test_instance_size} interactions for testing')
        # print(f'{self.val_instance_size} interactions for validation')  
        # print(f'{self.data_name} sparsity:  '
        #       f'{(self.train_instance_size + self.test_instance_size) / self.n_users / self.m_items}')

        # (users, items), bipartite graph
        self.user_item_net = sp.csr_matrix((np.ones(self.train_instance_size), (self.train_users, self.train_items)),
                                           shape=(self.n_users, self.m_items))

        self.users_D = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # pre-process
        self.all_pos = self._get_user_posItems(list(range(self.n_users)))
        self.test_dict = self._build_test()
        self.val_dict = self._build_val()
        # print(f"{self.data_name} is loaded")
        
        # for MGCCF only
        self._UU = None
        self._VV = None
        
        # for caged only
        self.u_neighbors = None 
        if board.args.model == 'caged':
            self.u_neighbors = self._get_user_neighbors(list(range(self.n_users)))

    def get_num_users(self):
        return self.n_users

    def get_num_items(self):
        return self.m_items

    def get_num_train_instance(self):
        return self.train_instance_size

    def get_test_dict(self):
        if board.args.mode == 'val':
            return self.val_dict
        return self.test_dict

    def get_all_pos(self):
        return self.all_pos
    
    def _build_test(self):
        """
        :return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.test_items):
            user = self.test_users[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def _build_val(self): 
        """
        :return:
            dict: {user: [items]}
        """
        val_data = {}
        for i, item in enumerate(self.val_items):
            user = self.val_users[i]
            if val_data.get(user):
                val_data[user].append(item)
            else:
                val_data[user] = [item]
        return val_data

    def _get_user_posItems(self, user_index, mode='ori'):
        """
        Modified to handle different modes (val vs test)
        """
        pos_items = []
        # Mode val: Return Train + Test
        if mode == 'val':
            if self.data_name == 'kuairec':
                for user in user_index:
                    train_items = self.user_item_net[user].nonzero()[1]
                    # (Masking both ensures validation metric only sees validation items)
                    combined = np.concatenate((train_items, self.test_items_unique))
                    pos_items.append(combined)
            elif self.data_name == 'coat':
                for user in user_index:
                    train_items = self.user_item_net[user].nonzero()[1]
                    test_items = self.raw_test_items[user]
                    combined = np.concatenate((train_items, test_items))
                    pos_items.append(combined)
                
        # Mode test: Return "All except test"
        elif mode == 'test':
            if self.data_name == 'kuairec':
                pos_items = [self.non_test_items[:] for _ in range(len(user_index))]
            elif self.data_name == 'coat':
                for user in user_index:
                    non_test_items = np.setdiff1d(np.arange(self.m_items), self.raw_test_items[user])
                    pos_items.append(non_test_items)
            
        # Normal/Train mode: Return just Train    
        else:
            for user in user_index:
                train_items = self.user_item_net[user].nonzero()[1]        
                pos_items.append(train_items)
                
        return pos_items

    def _convert_sp_mat_to_sp_tensor(self, matrix):
        """
        sparse matrix converting between scipy.sparse and torch.sparse
        :return: scipy.sparse_matrix of input matrix
        """
        coo = matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def load_sparse_graph(self):
        print('loading adjacency matrix')
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(os.path.join(self.path, 's_pre_adj_mat.npz'))
                print('load graph successfully ...')
                norm_adj = pre_adj_mat
            except:
                print('generating graph')
                start = time()
                adj_mat = sp.lil_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                R = self.user_item_net.tolil()

                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f'construction complete with {end - start}s, saving norm_mat...')
                sp.save_npz(os.path.join(self.path, 's_pre_adj_mat.npz'), norm_adj)

            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce()
        return self.Graph.to(board.DEVICE)


    # for MGCCF only
    def _row_l2_normalize_csr(self, mat: sp.csr_matrix) -> sp.csr_matrix:
        mat = mat.tocsr(copy=True)
        row_norm = np.sqrt(mat.multiply(mat).sum(axis=1)).A1
        row_norm[row_norm == 0.0] = 1.0
        inv = 1.0 / row_norm
        D_inv = sp.diags(inv, format='csr')
        return D_inv @ mat

    def _col_l2_normalize_csc(self, mat: sp.csc_matrix) -> sp.csc_matrix:
        mat = mat.tocsc(copy=True)
        col_norm = np.sqrt(mat.multiply(mat).sum(axis=0)).A1
        col_norm[col_norm == 0.0] = 1.0
        inv = 1.0 / col_norm
        D_inv = sp.diags(inv, format='csc')
        return mat @ D_inv

    def _symmetric_normalize(self, mat: sp.csr_matrix) -> sp.csr_matrix:
        mat = mat.tocsr(copy=False)
        deg = np.asarray(mat.sum(axis=1)).ravel()
        deg[deg == 0.0] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        D_left = sp.diags(d_inv_sqrt, format='csr')
        mat = mat @ sp.diags(d_inv_sqrt, format='csr')
        mat = D_left @ mat
        return mat.tocsr(copy=False)

    def _topk_per_row(self, mat: sp.csr_matrix, k: int, min_sim: float = 0.0, remove_self: bool = True) -> sp.csr_matrix:
        mat = mat.tocsr(copy=False)
        n_rows = mat.shape[0]
        indptr = mat.indptr
        indices = mat.indices
        data = mat.data

        new_indptr = [0]
        new_indices = []
        new_data = []

        for r in range(n_rows):
            start, end = indptr[r], indptr[r + 1]
            row_idx = indices[start:end]
            row_val = data[start:end]

            if remove_self:
                mask = row_idx != r
                row_idx = row_idx[mask]
                row_val = row_val[mask]

            if row_val.size == 0:
                new_indptr.append(len(new_indices))
                continue

            if min_sim > 0.0:
                m = row_val >= min_sim
                row_idx = row_idx[m]
                row_val = row_val[m]

            if row_val.size > k:
                topk_idx = np.argpartition(-row_val, kth=k-1)[:k]
                row_idx = row_idx[topk_idx]
                row_val = row_val[topk_idx]

            order = np.argsort(-row_val, kind='mergesort')
            row_idx = row_idx[order]
            row_val = row_val[order]

            new_indices.extend(row_idx.tolist())
            new_data.extend(row_val.tolist())
            new_indptr.append(len(new_indices))

        new_indptr = np.array(new_indptr, dtype=np.int32)
        new_indices = np.array(new_indices, dtype=np.int32)
        new_data = np.array(new_data, dtype=np.float32)

        n_cols = mat.shape[1]
        return sp.csr_matrix((new_data, new_indices, new_indptr), shape=(n_rows, n_cols))

    def _save_npz_safe(self, path, mat: sp.csr_matrix):
        sp.save_npz(path, mat)

    def _load_npz_safe(self, path):
        return sp.load_npz(path)

    def load_user_user_graph(self, k=10, min_sim=0.0, symmetrize=True, normalize=True, cache_file='uu_topk.npz'):
        if self._UU is not None:
            return self._UU

        npz_path = os.path.join(self.path, cache_file)
        if os.path.exists(npz_path):
            try:
                UU = self._load_npz_safe(npz_path)
                print(f'load UU graph from {npz_path}')
            except Exception as e:
                print(f'failed to load UU npz, rebuild. reason: {e}')
                UU = None
        else:
            UU = None

        if UU is None:
            print('generating UU cosine graph')
            t0 = time()
    
            R_norm = self._row_l2_normalize_csr(self.user_item_net)
            UU = R_norm @ R_norm.T  # U x U
            UU = self._topk_per_row(UU, k=k, min_sim=min_sim, remove_self=True)
            if symmetrize:
                UT = UU.transpose().tocsr(copy=False)
                UU = UU.maximum(UT)
            if normalize:
                UU = self._symmetric_normalize(UU)
            self._save_npz_safe(npz_path, UU)
            print(f'UU built in {time()-t0:.3f}s, nnz={UU.nnz}, avg deg≈{UU.nnz/self.n_users:.2f}')

        torch_UU = self._convert_sp_mat_to_sp_tensor(UU).coalesce().to(board.DEVICE)
        self._UU = torch_UU
        return self._UU

    def load_item_item_graph(self, k=10, min_sim=0.0, symmetrize=True, normalize=True, cache_file='vv_topk.npz'):
        if self._VV is not None:
            return self._VV

        npz_path = os.path.join(self.path, cache_file)
        if os.path.exists(npz_path):
            try:
                VV = self._load_npz_safe(npz_path)
                print(f'load VV graph from {npz_path}')
            except Exception as e:
                print(f'failed to load VV npz, rebuild. reason: {e}')
                VV = None
        else:
            VV = None

        if VV is None:
            print('generating VV cosine graph')
            t0 = time()

            R_csc = self.user_item_net.tocsc(copy=True)
            R_csc = self._col_l2_normalize_csc(R_csc)
            Rt = R_csc.transpose().tocsr(copy=False)  # I x U
            VV = Rt @ Rt.transpose().tocsr(copy=False)  # I x I
            VV = self._topk_per_row(VV, k=k, min_sim=min_sim, remove_self=True)
            if symmetrize:
                VT = VV.transpose().tocsr(copy=False)
                VV = VV.maximum(VT)
            if normalize:
                VV = self._symmetric_normalize(VV)
            self._save_npz_safe(npz_path, VV)
            print(f'VV built in {time()-t0:.3f}s, nnz={VV.nnz}, avg deg≈{VV.nnz/self.m_items:.2f}')

        torch_VV = self._convert_sp_mat_to_sp_tensor(VV).coalesce().to(board.DEVICE)
        self._VV = torch_VV
        return self._VV
    
    
    # for caged only
    def _get_user_neighbors(self, user_index):
        u_neighbors = torch.LongTensor([])
        for user in user_index:
            u_pos_i = self.user_item_net[user].nonzero()[1]
            u_neighbors = torch.concat([u_neighbors, torch.stack((torch.full((len(u_pos_i),), user), torch.tensor(u_pos_i)), dim=1)])
        return u_neighbors
    
    def get_root_degree(self):
        degree = self.Graph.values()
        degree = degree[degree != 0]
        degree = 1.0 / degree
        degree = degree.mean()

        return degree.item()
    
    def get_neighbors(self):
        return self.u_neighbors
    