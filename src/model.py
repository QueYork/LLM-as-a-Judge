import torch
import torch.nn as nn
import torch.nn.functional as F
import src.powerboard as board
import src.data_loader as data_loader
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import sum, mul, fill_diag, remove_diag
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def get_scores(self, user_index):
        raise NotImplementedError

# LightGCN from "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, 2020"
class LightGCN(BasicModel):
    def __init__(self, dataset):
        super(LightGCN, self).__init__()
        self.dataset: data_loader.LoadData = dataset

        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.dim = board.args.dim
        self.num_layers = board.args.layers
        self.user_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim)
        self.item_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim)
        
        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)
        board.cprint('LightGCN initialized with NORMAL distribution.')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.load_sparse_graph()

    # Embedding aggregation
    def aggregate_embed(self): 
        user_embed = self.user_embed.weight # [#users, dim]
        item_embed = self.item_embed.weight # [#items, dim]

        all_emb = torch.cat([user_embed, item_embed])
        emb_list = [all_emb]      
        for layer in range(self.num_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            emb_list.append(all_emb)
        con_embed_list = torch.stack(emb_list, dim=1)
        
        return con_embed_list[:self.num_users, :], con_embed_list[self.num_users:, :] # [n_entity, num_layer + 1, dim]
    
    def pooling(self, embed: torch.Tensor):
        return torch.mean(embed, dim=1)  # avg pooling
    
    def _BPR_loss(self, user_embed, pos_embed, neg_embed):
        pos_scores = torch.mul(user_embed, pos_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_embed, neg_embed)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss
    
    def loss(self, user_index, pos_index, neg_index):
        user_emb, item_emb = self.aggregate_embed()
        
        nocon_u_e = user_emb[user_index]
        nocon_pos_e = item_emb[pos_index]
        nocon_neg_e = item_emb[neg_index]
        
        user_emb = self.pooling(user_emb)
        item_emb = self.pooling(item_emb)
        
        u_e = user_emb[user_index]
        pos_e = item_emb[pos_index]
        neg_e = item_emb[neg_index]
        
        reg_loss = 0.5 * (torch.norm(nocon_u_e[:, 0, :]) ** 2
                       + torch.norm(nocon_pos_e[:, 0, :]) ** 2
                       + torch.norm(nocon_neg_e[:, 0, :]) ** 2) / float(len(user_index))
        
        loss1 = self._BPR_loss(u_e, pos_e, neg_e)
        return loss1, reg_loss

    def get_scores(self, user_index):
        all_user_embed, all_item_embed = self.aggregate_embed()
        
        all_user_embed = self.pooling(all_user_embed)
        all_item_embed = self.pooling(all_item_embed)
        
        user_embed = all_user_embed[user_index.long()]
        scores = self.f(torch.matmul(user_embed, all_item_embed.t()))
        return scores

# Matrix Factorization from "Matrix Factorization Techniques for Recommender Systems, 2009"
class MF(BasicModel):
    def __init__(self, dataset):
        super(MF, self).__init__()
        self.dataset = dataset
        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.dim = board.args.dim

        self.user_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim)
        self.item_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim)

        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)
        board.cprint('MF initialized with NORMAL distribution.')

        self.f = nn.Sigmoid()

    def _BPR_loss(self, user_embed, pos_embed, neg_embed):
        pos_scores = torch.mul(user_embed, pos_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_embed, neg_embed)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss

    def loss(self, user_index, pos_index, neg_index):
        u_e = self.user_embed(user_index)
        pos_e = self.item_embed(pos_index)
        neg_e = self.item_embed(neg_index)

        reg_loss = 0.5 * (torch.norm(u_e) ** 2
                        + torch.norm(pos_e) ** 2
                        + torch.norm(neg_e) ** 2) / float(len(user_index))

        loss1 = self._BPR_loss(u_e, pos_e, neg_e)
        return loss1, reg_loss

    def get_scores(self, user_index):
        all_user_embed = self.user_embed.weight    # [num_users, dim]
        all_item_embed = self.item_embed.weight    # [num_items, dim]

        user_embed = all_user_embed[user_index.long()]  # [batch_size, dim]
        scores = self.f(torch.matmul(user_embed, all_item_embed.t()))  # [batch_size, num_items]
        return scores
    
# LR-GCCF from "Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach, 2020"
class LRGCCF(BasicModel):
    def __init__(self, dataset):
        super(LRGCCF, self).__init__()
        self.dataset: data_loader.LoadData = dataset

        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.dim = board.args.dim
        self.num_layers = board.args.layers

        self.user_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim)
        self.item_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim)

        # Linear Residual
        self.linear_layers = nn.ModuleList(
            [nn.Linear(self.dim, self.dim, bias=False) for _ in range(self.num_layers)]
        )
        
        for layer in self.linear_layers:
            nn.init.normal_(layer.weight, std=0.1)
        
        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)
        board.cprint('LR-GCCF initialized with NORMAL distribution.')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.load_sparse_graph()

    # Embedding aggregation
    def aggregate_embed(self):  
        user_embed = self.user_embed.weight  # [num_users, dim]
        item_embed = self.item_embed.weight  # [num_items, dim]

        all_emb = torch.cat([user_embed, item_embed])
        emb_list = [all_emb]

        for layer in range(self.num_layers):
            # h_{l+1} = A * h_l * W_l + h_l
            neighbor_emb = torch.sparse.mm(self.Graph, all_emb)   
            transformed_emb = self.linear_layers[layer](neighbor_emb)  
            
            # all_emb = A * all_emb * W_l + all_emb
            all_emb = transformed_emb + all_emb                  

            emb_list.append(all_emb)

        con_embed_list = torch.stack(emb_list, dim=1)
        return con_embed_list[:self.num_users, :], con_embed_list[self.num_users:, :]

    def pooling(self, embed: torch.Tensor):
        return torch.mean(embed, dim=1)

    def _BPR_loss(self, user_embed, pos_embed, neg_embed):
        pos_scores = torch.mul(user_embed, pos_embed)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user_embed, neg_embed)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss

    def loss(self, user_index, pos_index, neg_index):
        user_emb, item_emb = self.aggregate_embed()
        
        nocon_u_e = user_emb[user_index]
        nocon_pos_e = item_emb[pos_index]
        nocon_neg_e = item_emb[neg_index]
        
        user_emb = self.pooling(user_emb)
        item_emb = self.pooling(item_emb)
        
        u_e = user_emb[user_index]
        pos_e = item_emb[pos_index]
        neg_e = item_emb[neg_index]
        
        reg_loss = 0.5 * (torch.norm(nocon_u_e[:, 0, :]) ** 2
                       + torch.norm(nocon_pos_e[:, 0, :]) ** 2
                       + torch.norm(nocon_neg_e[:, 0, :]) ** 2) / float(len(user_index))
        
        loss1 = self._BPR_loss(u_e, pos_e, neg_e)
        return loss1, reg_loss

    def get_scores(self, user_index):
        all_user_embed, all_item_embed = self.aggregate_embed()
        
        all_user_embed = self.pooling(all_user_embed)
        all_item_embed = self.pooling(all_item_embed)
        
        user_embed = all_user_embed[user_index.long()]
        scores = self.f(torch.matmul(user_embed, all_item_embed.t()))
        return scores
    
# NGCF from "Neural Graph Collaborative Filtering, 2019"
class NGCF(BasicModel):
    def __init__(self, dataset):
        super(NGCF, self).__init__()
        self.dataset: data_loader.LoadData = dataset

        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.dim = board.args.dim
        self.num_layers = board.args.layers
        self.dropout = board.args.dropout 

        self.user_embed = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim)
        self.item_embed = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

        self.W1 = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(self.num_layers)])
        self.W2 = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(self.num_layers)])

        for layer in range(self.num_layers):
            nn.init.xavier_uniform_(self.W1[layer].weight)
            nn.init.xavier_uniform_(self.W2[layer].weight)
        
        board.cprint('NGCF initialized with XAVIER distribution.')
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)  
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.load_sparse_graph()

    def aggregate_embed(self):  
        user_embed = self.user_embed.weight  # [num_users, dim]
        item_embed = self.item_embed.weight  # [num_items, dim]
        all_emb = torch.cat([user_embed, item_embed])  # [num_user+num_item, dim]
        ego_emb = all_emb    

        emb_list = [all_emb]

    
        # Message + interaction 
        for layer in range(self.num_layers):
            # Aggregate neighborhood messages
            side_emb = torch.sparse.mm(self.Graph, all_emb)  # A*E
            
            # Add interaction（e_l ⊙ e_0)
            interact_emb = side_emb * ego_emb  # element-wise
            neighbor_msg = self.W1[layer](side_emb) + self.W2[layer](interact_emb)

            # Activation + dropout
            all_emb = self.leaky_relu(neighbor_msg)
            all_emb = F.dropout(all_emb, p=self.dropout, training=self.training)
            
            # Normalize the embeddings
            norm = torch.norm(all_emb, p=2, dim=1, keepdim=True) + 1e-10
            all_emb = all_emb / norm

            emb_list.append(all_emb)

        con_embed_list = torch.stack(emb_list, dim=1)
        return con_embed_list[:self.num_users, :], con_embed_list[self.num_users:, :]

    def pooling(self, embed: torch.Tensor):
        return torch.mean(embed, dim=1)

    def _BPR_loss(self, user_embed, pos_embed, neg_embed):
        pos_scores = torch.mul(user_embed, pos_embed).sum(dim=1)
        neg_scores = torch.mul(user_embed, neg_embed).sum(dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss

    def loss(self, user_index, pos_index, neg_index):
        user_emb, item_emb = self.aggregate_embed()

        nocon_u_e = user_emb[user_index]
        nocon_pos_e = item_emb[pos_index]
        nocon_neg_e = item_emb[neg_index]
        
        user_emb = self.pooling(user_emb)
        item_emb = self.pooling(item_emb)
        
        u_e = user_emb[user_index]
        pos_e = item_emb[pos_index]
        neg_e = item_emb[neg_index]
        
        reg_loss = 0.5 * (torch.norm(nocon_u_e[:, 0, :])**2 +
                          torch.norm(nocon_pos_e[:, 0, :])**2 +
                          torch.norm(nocon_neg_e[:, 0, :])**2) / float(len(user_index))
        
        loss1 = self._BPR_loss(u_e, pos_e, neg_e)
        return loss1, reg_loss

    def get_scores(self, user_index):
        all_user_embed, all_item_embed = self.aggregate_embed()
        all_user_embed = self.pooling(all_user_embed)
        all_item_embed = self.pooling(all_item_embed)
        
        user_embed = all_user_embed[user_index.long()]
        scores = self.f(torch.matmul(user_embed, all_item_embed.T))
        return scores
    
# MGCCF from "Multi-Graph Convolution Collaborative Filtering, 2019"
class MGCCF(BasicModel):
    def __init__(self, dataset):
        super(MGCCF, self).__init__()
        self.dataset = dataset
        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.dim = board.args.dim
        self.num_layers = board.args.layers

        self.user_embed = nn.Embedding(self.num_users, self.dim)
        self.item_embed = nn.Embedding(self.num_items, self.dim)
        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)

        self.Graph = self.dataset.load_sparse_graph()        # (U+I)x(U+I)
        self.UU = self.dataset.load_user_user_graph()        # UxU
        self.VV = self.dataset.load_item_item_graph()        # IxI

        self.act = nn.Tanh()
        self.user_trans = nn.ModuleList([nn.Linear(self.dim, self.dim, bias=True) for _ in range(self.num_layers)])
        self.item_trans = nn.ModuleList([nn.Linear(self.dim, self.dim, bias=True) for _ in range(self.num_layers)])

        # MGE
        self.user_mge = nn.Linear(self.dim, self.dim, bias=True)
        self.item_mge = nn.Linear(self.dim, self.dim, bias=True)

        # Skip-connection 
        self.user_skip = nn.Linear(self.dim, self.dim, bias=True)
        self.item_skip = nn.Linear(self.dim, self.dim, bias=True)

        self.f = nn.Sigmoid()
        board.cprint('MGCCF initialized.')

    def _sparse_mm(self, A: torch.Tensor, X: torch.Tensor):
        return torch.sparse.mm(A, X)

    def _bipar_gcn_embed(self):
        user_e0 = self.user_embed.weight
        item_e0 = self.item_embed.weight
        all_emb = torch.cat([user_e0, item_e0], dim=0)

        user_list = [user_e0]
        item_list = [item_e0]

        cur_all = all_emb
        for k in range(self.num_layers):
            cur_all = self._sparse_mm(self.Graph, cur_all)  # [U+I, d]
            cur_user = cur_all[:self.num_users, :]
            cur_item = cur_all[self.num_users:, :]

            cur_user = self.act(self.user_trans[k](cur_user))
            cur_item = self.act(self.item_trans[k](cur_item))

            user_list.append(cur_user)
            item_list.append(cur_item)
            cur_all = torch.cat([cur_user, cur_item], dim=0)

        user_stack = torch.stack(user_list, dim=1)
        item_stack = torch.stack(item_list, dim=1)
        return user_stack, item_stack

    def _mge_embed(self, user_base: torch.Tensor, item_base: torch.Tensor):
        agg_u = self._sparse_mm(self.UU, user_base)
        agg_v = self._sparse_mm(self.VV, item_base)
        zu = self.act(self.user_mge(agg_u))
        zv = self.act(self.item_mge(agg_v))
        return zu, zv

    def _skip_embed(self):
        su = self.act(self.user_skip(self.user_embed.weight))
        sv = self.act(self.item_skip(self.item_embed.weight))
        return su, sv

    def aggregate_embed(self):
        user_stack, item_stack = self._bipar_gcn_embed()
        return user_stack, item_stack

    def pooling(self, embed: torch.Tensor):
        return torch.mean(embed, dim=1)

    def get_final_embeddings(self):
        user_stack, item_stack = self.aggregate_embed()
        h_u = self.pooling(user_stack)  # [U, d]
        h_v = self.pooling(item_stack)  # [I, d]

        zu, zv = self._mge_embed(h_u, h_v)
        su, sv = self._skip_embed()

        eu = h_u + zu + su
        ev = h_v + zv + sv
        return eu, ev, user_stack, item_stack

    def _BPR_loss(self, user_embed, pos_embed, neg_embed):
        pos_scores = torch.sum(user_embed * pos_embed, dim=1)
        neg_scores = torch.sum(user_embed * neg_embed, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss

    def loss(self, user_index, pos_index, neg_index):
        eu, ev, user_stack, item_stack = self.get_final_embeddings()

        u_e = eu[user_index]
        pos_e = ev[pos_index]
        neg_e = ev[neg_index]
        
        nocon_u_e = user_stack[user_index, 0, :]
        nocon_pos_e = item_stack[pos_index, 0, :]
        nocon_neg_e = item_stack[neg_index, 0, :]

        reg_loss = 0.5 * (
            torch.norm(nocon_u_e) ** 2 +
            torch.norm(nocon_pos_e) ** 2 +
            torch.norm(nocon_neg_e) ** 2
        ) / float(len(user_index))

        bpr = self._BPR_loss(u_e, pos_e, neg_e)
        return bpr, reg_loss  

    def get_scores(self, user_index):
        eu, ev, _, _ = self.get_final_embeddings()
        user_embed = eu[user_index.long()]
        scores = self.f(torch.matmul(user_embed, ev.t()))
        return scores

# NeuMF from "Neural Collaborative Filtering, 2017"
class NeuMF(BasicModel):
    def __init__(self, dataset):
        super(NeuMF, self).__init__()
        self.dataset: data_loader.LoadData = dataset

        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.dim = board.args.dim
        
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim)
        
        self.fc_layers = torch.nn.ModuleList()
        self.num_layers = board.args.layers
        
        self.dropout = nn.Dropout(p=board.args.dropout)
        
        for _ in range(self.num_layers - 1):
            self.fc_layers.append(torch.nn.Linear(in_features=self.dim*2, out_features=self.dim*2))
        self.fc_layers.append(torch.nn.Linear(in_features=self.dim*2, out_features=self.dim))
        
        self.affine_output = torch.nn.Linear(in_features=self.dim*2, out_features=1)
        self.logistic = torch.nn.Sigmoid()
        
        for sm in self.modules():
            if isinstance(sm, (nn.Embedding, nn.Linear)):
                torch.nn.init.normal_(sm.weight.data, 0.0, 0.1)
        board.cprint('NeuMF initialized with NORMAL distribution.')
        
    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)
            # mlp_vector = self.dropout(mlp_vector)


        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating       
    
    def loss(self, user_index, pos_index, neg_index):
        # u_e_mlp = self.embedding_user_mlp(user_index)
        # pos_e_mlp = self.embedding_item_mlp(pos_index)
        # neg_e_mlp = self.embedding_item_mlp(neg_index)
        
        # u_e_mf = self.embedding_user_mf(user_index)
        # pos_e_mf = self.embedding_item_mf(pos_index)
        # neg_e_mf = self.embedding_item_mf(neg_index)
        
        # u_e = torch.cat([u_e_mlp, u_e_mf], dim=-1)
        # pos_e = torch.cat([pos_e_mlp, pos_e_mf], dim=-1)
        # neg_e = torch.cat([neg_e_mlp, neg_e_mf], dim=-1)

        # reg_loss = 0.5 * (torch.norm(u_e) ** 2
        #                 + torch.norm(pos_e) ** 2
        #                 + torch.norm(neg_e) ** 2) / float(len(user_index))
        
        pos_scores = self.forward(user_index, pos_index)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = self.forward(user_index, neg_index)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss1 = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss1, 0
    
    def get_scores(self, user_index):
        num_users = len(user_index)
        num_items = self.num_items

        all_items = torch.arange(num_items, device=board.DEVICE).unsqueeze(0).expand(num_users, -1)
        all_users = user_index.unsqueeze(1).expand(-1, num_items)

        all_users_flat = all_users.reshape(-1)
        all_items_flat = all_items.reshape(-1)

        scores = self.forward(all_users_flat, all_items_flat)  # [B*I, 1]
        scores = scores.view(num_users, num_items)

        return scores

# SimGCL from "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation, 2022"
class SimGCL(BasicModel):
    def __init__(self, dataset):
        super(SimGCL, self).__init__()
        self.dataset = dataset

        self.__init_model()

    def __init_model(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.dim = board.args.dim
        self.num_layers = board.args.layers

        self.cl_rate = board.args.lam      # contrastive loss coefficient lam
        self.eps = board.args.eps              # perturbation strength ε

        self.user_embed = nn.Embedding(self.num_users, self.dim)
        self.item_embed = nn.Embedding(self.num_items, self.dim)

        nn.init.normal_(self.user_embed.weight, std=0.1)
        nn.init.normal_(self.item_embed.weight, std=0.1)
        board.cprint('SimGCL initialized with NORMAL distribution.')

        self.Graph = self.dataset.load_sparse_graph()
        self.f = nn.Sigmoid()
    
    def propagate(self, perturbed=False):
        user_e = self.user_embed.weight
        item_e = self.item_embed.weight
        all_e = torch.cat([user_e, item_e])

        embeddings = [all_e]
        for layer in range(self.num_layers):
            all_e = torch.sparse.mm(self.Graph, all_e) 
            if perturbed:
                noise = torch.rand_like(all_e)
                noise = F.normalize(noise, dim=-1) * self.eps
                all_e += torch.sign(all_e) * noise
            embeddings.append(all_e)
        
        embeddings = torch.stack(embeddings, dim=1)  # [N, num_layers+1, dim]
        return embeddings[:self.num_users, :], embeddings[self.num_users:, :]

    def pooling(self, embed):
        return torch.mean(embed, dim=1)

    def info_nce(self, x1: torch.Tensor, x2: torch.Tensor, temperature=0.2):
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)
        pos_score = torch.sum(x1 * x2, dim=-1)
        pos_score = torch.exp(pos_score / temperature)

        sim_matrix = torch.matmul(x1, x2.T)
        sim_matrix = torch.exp(sim_matrix / temperature)
        denominator = torch.sum(sim_matrix, dim=-1)
        loss = -torch.log(pos_score / denominator)
        return torch.mean(loss)

    def _BPR_loss(self, user_embed, pos_embed, neg_embed):
        pos_scores = torch.sum(user_embed * pos_embed, dim=1)
        neg_scores = torch.sum(user_embed * neg_embed, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss

    def loss(self, user_index, pos_index, neg_index):
        user_emb, item_emb = self.propagate()
        user_final = self.pooling(user_emb)
        item_final = self.pooling(item_emb)

        u_e = user_final[user_index]
        pos_e = item_final[pos_index]
        neg_e = item_final[neg_index]
        bpr_loss = self._BPR_loss(u_e, pos_e, neg_e)

        reg_loss = 0.5 * (
            torch.norm(self.user_embed(user_index)) ** 2 +
            torch.norm(self.item_embed(pos_index)) ** 2 +
            torch.norm(self.item_embed(neg_index)) ** 2
        ) / float(len(user_index))

        user_view1, item_view1 = self.propagate(perturbed=True)
        user_view2, item_view2 = self.propagate(perturbed=True)

        user_view1 = self.pooling(user_view1)
        user_view2 = self.pooling(user_view2)
        item_view1 = self.pooling(item_view1)
        item_view2 = self.pooling(item_view2)

        u_idx = torch.unique(user_index)
        i_idx = torch.unique(pos_index)
        user_cl_loss = self.info_nce(user_view1[u_idx], user_view2[u_idx])
        item_cl_loss = self.info_nce(item_view1[i_idx], item_view2[i_idx])
        cl_loss = self.cl_rate * (user_cl_loss + item_cl_loss)

        return bpr_loss + cl_loss, reg_loss

    def get_scores(self, user_index):
        user_emb, item_emb = self.propagate()
        user_emb = self.pooling(user_emb)
        item_emb = self.pooling(item_emb)

        u_e = user_emb[user_index.long()]
        scores = self.f(torch.matmul(u_e, item_emb.t()))
        return scores

# GTN from "Graph Trend Filtering Networks for Recommendations, 2022"
class GeneralPropagation(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 add_self_loops_l1: bool = True,
                 normalize: bool = True,
                 mode: str = None,
                 node_num: int = None,
                 num_classes: int = None,
                 **kwargs):

        super(GeneralPropagation, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.mode = mode
        self.dropout = 0.1
        self.add_self_loops = add_self_loops
        self.add_self_loops_l1 = add_self_loops_l1
        self.cached = cached
        self.normalize = normalize
        
        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_inc = None

        self.node_num = node_num
        self.num_classes = num_classes
        self.lam = board.args.lam
        self.max_value = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_inc = None

    def get_incident_matrix(self, edge_index: Adj):
        size = edge_index.sizes()[1]
        row_index = edge_index.storage.row()
        col_index = edge_index.storage.col()
        mask = row_index >= col_index
        row_index = row_index[mask]
        col_index = col_index[mask]
        edge_num = row_index.numel()
        row = torch.cat([torch.arange(edge_num), torch.arange(edge_num)]).cuda()
        col = torch.cat([row_index, col_index])
        value = torch.cat([torch.ones(edge_num), -1 * torch.ones(edge_num)]).cuda()
        inc = SparseTensor(row=row, rowptr=None, col=col, value=value,
                           sparse_sizes=(edge_num, size))
        return inc

    def inc_norm(self, inc, edge_index, add_self_loops):
        if add_self_loops:
            edge_index = fill_diag(edge_index, 1.0)
        else:
            edge_index = remove_diag(edge_index)
        deg = sum(edge_index, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        inc = mul(inc, deg_inv_sqrt.view(1, -1))  ## col-wise
        return inc

    def check_inc(self, edge_index, inc, normalize=False):
        # return None  ## not checking it
        nnz = edge_index.nnz()
        if normalize:
            deg = torch.eye(edge_index.sizes()[0])  # .cuda()
        else:
            deg = sum(edge_index, dim=1).cpu()
            deg = torch.diag(deg)
        inc = inc.cpu()
        lap = (inc.t() @ inc).to_dense()
        adj = edge_index.cpu().to_dense()

        lap2 = deg - adj
        diff = torch.sum(torch.abs(lap2 - lap)) / nnz
        # import ipdb; ipdb.set_trace()
        assert diff < 0.000001, f'error: {diff} need to make sure L=B^TB'

    def forward(self, x: Tensor, edge_index: Adj, x_idx: Tensor = None,
                edge_weight: OptTensor = None, mode=None, niter=None,
                data=None) -> Tensor:
    
        if self.normalize:
            if isinstance(edge_index, Tensor):
                raise ValueError('Only support SparseTensor now')

            elif isinstance(edge_index, SparseTensor):
                ## first cache incident_matrix (before normalizing edge_index)
                cache = self._cached_inc
                if cache is None:
                    incident_matrix = self.get_incident_matrix(edge_index=edge_index)
                    
                    incident_matrix = self.inc_norm(inc=incident_matrix, edge_index=edge_index,
                                                    add_self_loops=self.add_self_loops_l1)

                    if self.cached:
                        self._cached_inc = incident_matrix
                        self.init_z = torch.zeros((incident_matrix.sizes()[0], x.size()[-1])).cuda()
                else:
                    incident_matrix = self._cached_inc

                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype)

                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        K_ = self.K if niter is None else niter
        if mode == None: mode = self.mode
        assert edge_weight is None
        if K_ <= 0:
            return x

        hh = x

        x, xs = self.gtn_forward(x=x, hh=hh, incident_matrix=incident_matrix, K=K_)
        return x, xs

    def gtn_forward(self, x, hh, K, incident_matrix):
        lambda2 = self.lam
        beta = 0.5
        gamma = 1

        ############################# parameter setting ##########################
        if lambda2 > 0: z = self.init_z.detach()

        xs = []
        for k in range(K):
            grad = x - hh
            smoo = x - gamma * grad
            temp = z + beta / gamma * (incident_matrix @ (smoo - gamma * (incident_matrix.t() @ z)))

            z = self.proximal_l1_conjugate(x=temp, lambda2=lambda2, beta=beta, gamma=gamma, m="L1")
            # import ipdb; ipdb.set_trace()

            ctz = incident_matrix.t() @ z

            x = smoo - gamma * ctz

            x = F.dropout(x, p=self.dropout, training=self.training)

        # print("wihtout average")
        light_out = x

        return light_out, xs

    def proximal_l1_conjugate(self, x: Tensor, lambda2, beta, gamma, m):
        if m == 'L1':
            x_pre = x
            x = torch.clamp(x, min=-lambda2, max=lambda2)
        else:
            raise ValueError('wrong prox')
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

class GTN(BasicModel):
    def __init__(self, dataset):
        super(GTN, self).__init__()
        self.dataset: data_loader.LoadData = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.get_num_users()
        self.num_items = self.dataset.get_num_items()
        self.latent_dim = board.args.dim
        self.n_layers = board.args.layers   
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        board.cprint('GTN initialized with NORMAL distribution.')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.load_sparse_graph()
        self.gp = GeneralPropagation(self.n_layers, cached=True)

    def computer(self, corrupted_graph=None):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        if corrupted_graph == None:
            g_droped = self.Graph
        else:
            g_droped = corrupted_graph

        x = all_emb
        rc = g_droped.indices()
        r = rc[0]
        c = rc[1]
        num_nodes = g_droped.shape[0]
        edge_index = SparseTensor(row=r, col=c, value=g_droped.values(), sparse_sizes=(num_nodes, num_nodes))
        emb, embs = self.gp.forward(x, edge_index, mode='GTN')
        light_out = emb

        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def get_scores(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

# CAGED from "Causality-aware Graph Aggregation Estimater for Popularity Debiasing in Top-K Recommendation, 2025"
class caged(BasicModel):
    def __init__(self):
        super(caged, self).__init__()
        self.__init_model()

    def __init_model(self):
        self.input_dim = board.args.dim
        self.condition_dim = board.args.dim
        self.hidden_dim = board.args.hidden_dim
        self.latent_dim = board.args.latent_dim
        
        self.sc_var = board.args.sc_var
        self.beta = board.args.beta
        
        # Encoder
        self.encoder = nn.Sequential()
        for i, dim in enumerate(self.hidden_dim):
            if i == 0:
                self.encoder.add_module(name="L{:d}".format(i), module=nn.Linear(self.input_dim + self.condition_dim, dim))
            else:
                self.encoder.add_module(name="L{:d}".format(i), module=nn.Linear(self.hidden_dim[i-1], dim))
            if i + 1 < len(self.hidden_dim): 
                self.encoder.add_module(name="A{:d}".format(i), module=nn.Tanh()) 
        
        # Encoder output: z's mean and log var
        self.fc_mu = nn.Linear(self.hidden_dim[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim[-1], self.latent_dim)

        # Decoder
        self.decoder = nn.Sequential()
        for i, dim in enumerate(self.hidden_dim):
            if i == 0:
                self.decoder.add_module(name="L{:d}".format(i), module=nn.Linear(self.latent_dim + self.condition_dim, dim))
                self.decoder.add_module(name="A{:d}".format(i), module=nn.Tanh()) 
               
            else: 
                self.decoder.add_module(name="L{:d}".format(i), module=nn.Linear(self.hidden_dim[i-1], dim))
                self.decoder.add_module(name="A{:d}".format(i), module=nn.Tanh()) 
        self.decoder.add_module(name="L{:d}".format(len(self.hidden_dim)), module=nn.Linear(self.hidden_dim[-1], self.input_dim))
        
    def encode(self, x, u):
        x_u = torch.cat([x, u], dim=1)  
        h = self.encoder(x_u)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, u):
        z_u = torch.cat([z, u], dim=1)  
        return self.decoder(z_u)
    
    def forward(self, x, u):        # (x, u) pairs
        x_sig = x
        u_sig = u
        
        mu, logvar = self.encode(x_sig, u_sig)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, u_sig)
        return recon_x, x_sig, mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kld_loss
    
    def get_scores(self, x, u):  
        recon_x, x_sig, mu, logvar = self.forward(x, u)
        scores = self.sc_var * F.mse_loss(recon_x, x_sig, reduction='none') - self.beta * 0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        scores = torch.sum(scores, dim=1)
        scores = (-scores).exp()
        return scores
    