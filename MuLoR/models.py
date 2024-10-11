import layers
from layers import *
from torch.nn.parameter import Parameter
from functools import reduce
import torch
import numpy as np
from utils import layers as layers_temp

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.normalize = False
        self.attention = False
        
        self.gc1 = GraphConvolution(nfeat, nhid, bias=False)
        self.gc2 = GraphConvolution(nhid, nclass, bias=False)
            
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        x = F.relu(x)
        return F.log_softmax(x, dim=1)


class HGAT(nn.Module):
    '''
    input: feature_list, adj_list
    return:list(len = n_type) of tensor(bsz * 4, n_nodes, dim2)
    '''
    def __init__(self, nfeat_list, nhid, out_dim, dropout):
        super(HGAT, self).__init__()
        self.para_init()
        self.attention = True
        self.lower_attention = True  #是否需要交叉type的attention
        self.embedding = False  #是否需要先线性化映成同样维度的embedding
        self.write_emb = False
        dim_1st = nhid
        dim_2nd = out_dim
        self.norm1 = torch.nn.LayerNorm(dim_1st,eps=1e-5,elementwise_affine= True)
        self.norm2 = torch.nn.LayerNorm(dim_2nd,eps=1e-5,elementwise_affine= True)
        if self.write_emb:
            self.emb = None

        self.nonlinear = F.elu_

        self.ntype = len(nfeat_list)   #nfeat_list=[d1,d2,d3] 表示三种type嵌入的维度
        if self.embedding:
            self.mlp = nn.ModuleList()
            n_in = [nhid for _ in range(self.ntype)]
            for t in range(self.ntype):
                self.mlp.append(MLP(nfeat_list[t], n_in[t]))
        else:
            n_in = nfeat_list

        # dim_1st = 1000
        # dim_2nd = nhid

        
        self.gc2 = nn.ModuleList()
        if not self.lower_attention:
            self.gc1 = nn.ModuleList()
            for t in range(self.ntype):
                self.gc1.append(GraphConvolution(n_in[t], dim_1st, bias=False) )
                self.bias1 = Parameter(torch.FloatTensor(dim_1st))
                stdv = 1. / math.sqrt(dim_1st)
                self.bias1.data.uniform_(-stdv, stdv)
        else:
            self.gc1 = GraphAttentionConvolution(n_in, dim_1st, gamma=0.1)
        self.gc2.append(GraphConvolution(dim_1st, dim_2nd, bias=True) )
        # self.gc2 = GraphAttentionConvolution([dim_1st] * self.ntype, dim_2nd)

        if self.attention:
            self.at1 = nn.ModuleList()
            self.at2 = nn.ModuleList()
            for t in range(self.ntype):
                self.at1.append(SelfAttention(dim_1st, t, 50) )  #(in_features, idx, hidden_dim)
                self.at2.append(SelfAttention(dim_2nd, t, 50) )
           
        # self.outlayer = torch.nn.Linear(dim_2nd, nclass)

        self.dropout = dropout

    def para_init(self):
        self.attention = True
        self.embedding = False
        self.lower_attention = False
        self.write_emb = False

    def forward(self, x_list, adj_list):
        if self.embedding:
            x0 = [None for _ in range(self.ntype)]
            for t in range(self.ntype):
                x0[t] = self.mlp[t](x_list[t])
        else:
            x0 = x_list
        
        if not self.lower_attention:
            x1 = [None for _ in range(self.ntype)]
            # 第一层gat，与第一层后的dropout
            for t1 in range(self.ntype):
                x_t1 = []
                for t2 in range(self.ntype):
                    if adj_list[t1][t2] is None:
                        continue
                    idx = t2
                    x_t1.append(self.gc1[idx](x0[t2], adj_list[t1][t2]) + self.bias1 )
                if self.attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1) )
                else:
                    x_t1 = reduce(torch.add, x_t1)

                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        else:
            x1 = [None for _ in range(self.ntype)]
            x1_in = self.gc1(x0, adj_list)  # list（len=type_num）of list(len=type_num) of tensor： size=（bsz*4， n_nodes，dim1）
            for t1 in range(len(x1_in)):
                x_t1 = x1_in[t1]  #list of tensor:size = (bsz*4,n_nodes, dim1)
                if self.attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1)) #stack: (bsz*4, 2,n_nodes,dim1)
                    #x_t1: tensor(bsz * 4, n_nodes, dim1)
                    #weights: tensor(bsz * 4, n_nodes, 2, 1)
                    # if t1 == 0:
                        # self.f.write('{0}\t{1}\t{2}\n'.format(weights[0][0].item(), weights[0][1].item(), weights[0][2].item()))
                else:
                    x_t1 = reduce(torch.add, x_t1)
                # x_t1 = self.norm1(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x_t1 = self.nonlinear(x_t1)
                x1[t1] = x_t1
        if self.write_emb:
            self.emb = x1[0]        
        
        x2 = [None for _ in range(self.ntype)]
        # 第二层gcn，与第二层后的softmax
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                if adj_list[t1][t2] is None:
                    continue
                x_t1.append(self.gc2[0](x1[t2], adj_list[t1][t2]))  #append(tensor(bsz * 4, n_nodes, dim2))
            if self.attention:
                x_t1, weights = self.at2[t1](torch.stack(x_t1, dim=1))  #stack: (bsz*4, 2,n_nodes,dim2)  --> tensor(bsz * 4, n_nodes, dim2)
            else:
                x_t1 = reduce(torch.add, x_t1)

            x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
            x_t1 = self.nonlinear(x_t1)
            x2[t1] = x_t1    #tensor(bsz * 4, n_nodes, dim2)
        return x2
    def inference(self, x_list, adj_list, adj_all = None):
        return self.forward(x_list, adj_list, adj_all)

        
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, features, adjs):
        bsz = features.size()[0]
        device = features.device
        output = torch.zeros_like(features).to(device)
        for bs in range(bsz):
            x = features[bs,:,:]
            adj = adjs[bs,:,:]
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            output[bs] = x

        return output
from torch_geometric.nn import RGCNConv
class RGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=2, dropout=0.1, num_relations=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.dropout_rate = dropout
        self.convs.append(RGCNConv(in_channels, out_channels, num_relations))
        for i in range(n_layers - 2):
            self.convs.append(RGCNConv(out_channels, out_channels, num_relations))
        self.convs.append(RGCNConv(out_channels, out_channels, num_relations))

    def forward(self, feature_list, adj_list, aug_pun_adj, pooled_output, p_nodes_mask, o_nodes_mask):
        device = pooled_output.device
        encoded_spans = feature_list[0]  # [bs, node_n, feature_dim]
        bs, node_n, feature_dim = encoded_spans.size()

        aug_adj = aug_pun_adj[0]  # [bs, node_n, node_n]
        punct_graph = aug_pun_adj[1]  # [bs, node_n, node_n]

        # 创建边的掩码
        mask_punct = (punct_graph == 1) & (aug_adj != 1)
        mask_aug = aug_adj == 1

        # 合并掩码
        edge_mask = mask_punct | mask_aug

        # 获取边的类型
        edge_types = torch.zeros_like(aug_adj, dtype=torch.long, device=device)
        edge_types[mask_punct] = 0
        edge_types[mask_aug] = 1
        edge_types = edge_types[edge_mask]

        # 获取边的索引
        edge_indices = edge_mask.nonzero(as_tuple=False)  # [num_edges, 3]; 列分别是 [batch_index, i, j]

        # 调整节点索引以匹配全局节点编号
        edge_indices_i = edge_indices[:, 0] * node_n + edge_indices[:, 1]
        edge_indices_j = edge_indices[:, 0] * node_n + edge_indices[:, 2]

        edge_index = torch.stack([edge_indices_i, edge_indices_j], dim=0).to(device)  # 形状: [2, num_edges]

        # 节点特征
        nodes = encoded_spans.view(-1, feature_dim).to(device)  # [bs * node_n, feature_dim]

        # 运行卷积层
        out = nodes  # [bs * node_n, feature_dim]
        for i, layer_module in enumerate(self.convs):
            out = F.dropout(out, self.dropout_rate, training=self.training)
            out = layer_module(out, edge_index, edge_types)
            out = F.elu(out)

        # 重塑输出为 [bs, node_n, feature_dim]
        out = out.view(bs, node_n, feature_dim)

        return out


class Message_gcn(nn.Module):
    def __init__(self, in_channels,  out_channels, n_layers=2, dropout=0.1,num_relations=2,sep_ie_layers = True,info_exchange = True):
        super(Message_gcn, self).__init__()
        #RGCN
        self.convs = torch.nn.ModuleList()
        self.relu = F.relu
        self.dropout_rate = dropout
        self.convs.append(RGCNConv(in_channels, out_channels, num_relations))
        for i in range(n_layers - 2):
            self.convs.append(RGCNConv(out_channels, out_channels, num_relations))
        self.convs.append(RGCNConv(out_channels, out_channels, num_relations))
        #HGCN
        self.layers = n_layers
        self.hgcn_layers = nn.ModuleList(
            [HypergraphConv(in_channels, out_channels, use_attention=True, heads=4, concat=False) for _ in range(n_layers)])
        self.dropout_rate = dropout
        self.info_exchange = info_exchange
        self.sep_ie_layers = sep_ie_layers
        if sep_ie_layers:
            self.ie_layers = nn.ModuleList([layers_temp.MLP(in_channels*2, out_channels*2, out_channels*2, 2, 0.1) for _ in range(n_layers)])
        else:
            self.ie_layer = layers_temp.MLP(in_channels*2, out_channels, out_channels*2, 2, 0.1)
    def forward(self,feature_list, adj_list,aug_pun_adj,pooled_output):

        # con node and edge HGCN
        device = pooled_output.device
        encoded_spans = feature_list[0]
        pooled_output_temp = pooled_output.unsqueeze(1)
        SVO_emb = feature_list[1]
        node_n = encoded_spans.size()[1]
        edge_n = SVO_emb.size()[1]
        sent2word_adj = adj_list[0][1]
        word2sent_adj = adj_list[1][0]
        bs = word2sent_adj.size()[0]
        all_H = []
        for k in range(bs):
            H = np.ones((node_n, edge_n + 1)) * 0
            for t in range(node_n):
                H[t][0] = 1
            if edge_n !=0:
                for i in range(node_n):
                    for j in range(edge_n):
                        if sent2word_adj[k][i][j]==1:
                            H[i][j+1] = 1
            all_H.append(H)

        all_hyperedge_index = []
        for one_H in all_H:
            node_index = []
            edge_index = []
            for i in range(node_n):
                for j in range(edge_n+1):
                    if one_H[i][j]==1:
                        node_index.append(i)
                        edge_index.append(j)
            hyperedge_index_orign = torch.tensor([node_index, edge_index])
            all_hyperedge_index.append(hyperedge_index_orign)
        all_out = torch.Tensor().to(device)


        word2sent_adj = adj_list[1][0]
        aug_adj = aug_pun_adj[0]
        punct_graph = aug_pun_adj[1]
        bs = word2sent_adj.size()[0]

        all_edge_index = []
        all_edge_type = []

        for k in range(bs):
            out_index = []
            in_index = []
            edge_type = []
            for i in range(node_n):
                for j in range(node_n):
                    if punct_graph[k][i][j] == 1 and aug_adj[k][i][j] != 1:
                        out_index.append(i)
                        in_index.append(j)
                        edge_type.append(0)
                    elif aug_adj[k][i][j] == 1:
                        out_index.append(i)
                        in_index.append(j)
                        edge_type.append(1)
            edge_index_orign = torch.tensor([out_index, in_index])
            all_edge_index.append(edge_index_orign)
            all_edge_type.append(torch.tensor(edge_type))

        all_out_RGCN = torch.Tensor().to(device)
        all_out_HGCN = torch.Tensor().to(device)
        for index in range(bs):
            # HGCN
            hyperedge_index = all_hyperedge_index[index]
            out_HGCN = encoded_spans[index]
            hyperedge_attr = torch.cat((pooled_output_temp[index], SVO_emb[index]), 0)
            out_HGCN = out_HGCN.to(device)
            hyperedge_index = hyperedge_index.to(device)
            hyperedge_attr = hyperedge_attr.to(device)
            # RGCN
            out_RGCN = encoded_spans[index]
            edge_index = all_edge_index[index]
            edge_type_index = all_edge_type[index]
            out_RGCN = out_RGCN.to(device)
            flag = 0

            for i in range(len(self.hgcn_layers)-len(self.convs)):
                out_HGCN = F.dropout(out_HGCN, self.dropout_rate, training=self.training)
                out_HGCN = self.hgcn_layers[i](out_HGCN, hyperedge_index, hyperedge_attr=hyperedge_attr)
                out_HGCN = F.relu(out_HGCN)
                flag = 1
            for i in range(len(self.convs)):
                if flag == 0:
                    # HGCN
                    out_HGCN = F.dropout(out_HGCN, self.dropout_rate, training=self.training)
                    out_HGCN = self.hgcn_layers[i](out_HGCN, hyperedge_index, hyperedge_attr=hyperedge_attr)
                    out_HGCN = F.relu(out_HGCN)
                    #RGCN
                    out_RGCN = F.dropout(out_RGCN, self.dropout_rate, training=self.training)
                    out_RGCN = out_RGCN.to(device)
                    edge_index = edge_index.to(device)
                    edge_type_index = edge_type_index.to(device)
                    out_RGCN = self.convs[i](out_RGCN, edge_index, edge_type_index)
                    out_RGCN = F.relu(out_RGCN)
                else:
                    # only RGCN
                    out_RGCN = F.dropout(out_RGCN, self.dropout_rate, training=self.training)
                    out_RGCN = out_RGCN.to(device)
                    edge_index = edge_index.to(device)
                    edge_type_index = edge_type_index.to(device)
                    out_RGCN = self.convs[i](out_RGCN, edge_index, edge_type_index)
                    out_RGCN = F.relu(out_RGCN)
                if self.info_exchange == True :

                    X_RGCN = out_RGCN[0,:]
                    X_HGCN = out_HGCN[0,:]
                    context_node_feats = torch.cat([X_RGCN, X_HGCN], dim=0)

                    if self.sep_ie_layers:
                        context_node_feats = self.ie_layers[i](context_node_feats)
                    else:
                        context_node_feats = self.ie_layer(context_node_feats)
                    # print('context_node_feats', context_node_feats.shape)
                    context_X_RGCN, context_X_HGCN = torch.split(context_node_feats, [out_RGCN.size(1), out_HGCN.size(1)], dim=0)
                    out_RGCN[0,:] = context_X_RGCN
                    out_HGCN[0,:] = context_X_HGCN
                    # _X = X.view_as(_X)
            out_RGCN = out_RGCN.unsqueeze(0)
            all_out_RGCN = torch.cat((all_out_RGCN,out_RGCN))
            out_HGCN = out_HGCN.unsqueeze(0)
            all_out_HGCN = torch.cat((all_out_HGCN, out_HGCN))
        return all_out_RGCN,all_out_HGCN



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
import transformers
from transformers import BertPreTrainedModel, RobertaModel, RobertaConfig, AutoModel, DebertaV2Config, AlbertConfig
from torch_geometric.nn import HypergraphConv, RGCNConv
from layers import FFNLayer, masked_softmax, weighted_sum, ResidualGRU, ArgumentGCN
from eval_attention import SCAttention
from new_train_rhlf import in_check

class MyHGAT(nn.Module):


    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,
                argument_bpe_ids: torch.LongTensor,
                punct_bpe_ids: torch.LongTensor,
                keytokensids: torch.LongTensor,
                keymask: torch.LongTensor,
                key_segid: torch.LongTensor,
                SVO_ids: torch.LongTensor,
                SVO_mask: torch.LongTensor,
                adj_SVO: torch.LongTensor,
                labels: torch.LongTensor,
                passage_mask: torch.LongTensor,
                question_mask: torch.LongTensor,
                sentence_embed=None,
                ) -> Tuple:

        # ...（省略，与之前的代码一致）

        sent_rep_HGCN = self.HGCN(feature_list, adj_list, pooled_output, p_nodes_mask, o_nodes_mask, p_o_mask,
                                  flat_SVO_mask)


class HGCN(nn.Module):
    def __init__(self, in_channel, out_channel, n_layer, dropout=0.2):
        super(HGCN, self).__init__()
        self.layers = n_layer
        self.hgcn_layers = nn.ModuleList([
            HypergraphConv(in_channel, out_channel, use_attention=True, heads=4, concat=False)
            for _ in range(n_layer)
        ])
        self.dropout_rate = dropout

    def forward(self, feature_list, adj_list, pooled_output, p_nodes_mask, o_nodes_mask, p_o_mask, flat_SVO_mask):
        device = pooled_output.device
        encoded_spans = feature_list[0].to(device)  # [bs, node_n, feature_dim]
        SVO_emb = feature_list[1].to(device)        # [bs, edge_n, feature_dim]
        bs = encoded_spans.size(0)                  # batch size (should be num_graph * 4)
        node_n = encoded_spans.size(1)
        feature_dim = encoded_spans.size(2)
        num_graph = bs // 4

        # 对 p_o_mask 进行填充，使其内层列表长度一致
        max_p_o_len = max(len(seq) for seq in p_o_mask)
        p_o_mask_padded = [seq + [0] * (max_p_o_len - len(seq)) for seq in p_o_mask]
        p_o_mask_tensor = torch.tensor(p_o_mask_padded, dtype=torch.long, device=device)  # [bs, max_p_o_len]

        # 对其他掩码也进行类似的处理（如果需要）
        # ...

        # Reshape data to [num_graph, 4, ...]
        encoded_spans = encoded_spans.view(num_graph, 4, node_n, feature_dim)
        SVO_emb = SVO_emb.view(num_graph, 4, -1, feature_dim)
        p_o_mask_tensor = p_o_mask_tensor.view(num_graph, 4, -1)  # [num_graph, 4, max_p_o_len]
        flat_SVO_mask = flat_SVO_mask.view(num_graph, 4, -1)
        sent2word_adj = adj_list[0][1].view(num_graph, 4, *adj_list[0][1].shape[-2:])

        # Calculate flags for indexing
        flag = (p_o_mask_tensor == 1).sum(dim=2)  # [num_graph, 4]
        flag_1 = (p_o_mask_tensor == 2).sum(dim=2)  # [num_graph, 4]

        # Compute cumulative indices
        index_ids = torch.zeros(num_graph, 5, dtype=torch.long, device=device)
        index_ids[:, 0] = flag[:, 0] + flag_1[:, 0]
        for j in range(1, 4):
            index_ids[:, j] = index_ids[:, j - 1] + flag_1[:, j]
        index_ids[:, -1] = index_ids[:, -2] + flag_1[:, 3]

        # Collect encoded spans
        all_encoded_spans = []
        for i in range(num_graph):
            spans = []
            for j in range(4):
                if j == 0:
                    end_idx = index_ids[i, j]
                    spans.append(encoded_spans[i, j, :end_idx])
                else:
                    start_idx = index_ids[i, j - 1]
                    end_idx = index_ids[i, j]
                    spans.append(encoded_spans[i, j, start_idx:end_idx])
            encoded_spans_temp = torch.cat(spans, dim=0)  # [node_n_i, feature_dim]
            all_encoded_spans.append(encoded_spans_temp)

        # Process each graph
        all_out = []
        for i in range(num_graph):
            edge_n = flat_SVO_mask[i].sum().int().item()
            node_n_i = all_encoded_spans[i].size(0)
            # Build all_new_adj
            all_new_adj_list = []
            for j in range(4):
                ed_temp = flat_SVO_mask[i, j].sum().int().item()
                p_end = index_ids[i, 0].item()
                pre_o_end = index_ids[i, j].item()
                o_end = index_ids[i, j + 1].item()

                # Adjust indices to prevent negative or zero dimensions
                temp_0 = sent2word_adj[i, j, :p_end, :ed_temp]
                temp_1_size = pre_o_end - p_end
                temp_1 = torch.zeros(max(temp_1_size, 0), ed_temp, device=device)
                temp_2 = sent2word_adj[i, j, p_end:pre_o_end + (o_end - pre_o_end), :ed_temp]
                temp_3_size = node_n_i - o_end
                temp_3 = torch.zeros(max(temp_3_size, 0), ed_temp, device=device)

                # Concatenate tensors
                td = torch.cat([temp_0, temp_1, temp_2, temp_3], dim=0)
                # Ensure td has the correct number of rows
                if td.size(0) != node_n_i:
                    # Adjust td to match node_n_i
                    td = F.pad(td, (0, 0, 0, node_n_i - td.size(0)), "constant", 0)
                all_new_adj_list.append(td)

            # Ensure all tensors in all_new_adj_list have the same number of rows
            for idx, td in enumerate(all_new_adj_list):
                if td.size(0) != node_n_i:
                    # Adjust td to match node_n_i
                    td = F.pad(td, (0, 0, 0, node_n_i - td.size(0)), "constant", 0)
                    all_new_adj_list[idx] = td

            all_new_adj = torch.cat(all_new_adj_list, dim=1)  # [node_n_i, total_edge_n]

            # Build incidence matrix H
            H = torch.zeros(node_n_i, edge_n + 5, device=device)
            for t in range(node_n_i):
                if t < index_ids[i, 0]:
                    H[t, 0] = 1
                elif t < index_ids[i, 1]:
                    H[t, 1] = 1
                elif t < index_ids[i, 2]:
                    H[t, 2] = 1
                elif t < index_ids[i, 3]:
                    H[t, 3] = 1
                elif t < index_ids[i, 4]:
                    H[t, 4] = 1
            if all_new_adj.size(0) != node_n_i:
                all_new_adj = all_new_adj[:node_n_i, :]
            H[:, 5:] = all_new_adj

            # Build hyperedge indices
            hyperedge_index = H.nonzero(as_tuple=False).t().contiguous().to(device)

            # Build hyperedge attributes
            f0_q = index_ids[i, 0].item()
            f0_o_1 = index_ids[i, 1].item()
            f0_o_2 = index_ids[i, 2].item()
            f0_o_3 = index_ids[i, 3].item()
            f0_o_4 = index_ids[i, 4].item()
            q_em = all_encoded_spans[i][:f0_q].sum(dim=0, keepdim=True)
            o_em_0 = all_encoded_spans[i][f0_q:f0_o_1].sum(dim=0, keepdim=True)
            o_em_1 = all_encoded_spans[i][f0_o_1:f0_o_2].sum(dim=0, keepdim=True)
            o_em_2 = all_encoded_spans[i][f0_o_2:f0_o_3].sum(dim=0, keepdim=True)
            o_em_3 = all_encoded_spans[i][f0_o_3:f0_o_4].sum(dim=0, keepdim=True)

            SVO_slices = [flat_SVO_mask[i, j].sum().int().item() for j in range(4)]
            o_a = SVO_emb[i, 0, :SVO_slices[0]]
            o_b = SVO_emb[i, 1, :SVO_slices[1]]
            o_c = SVO_emb[i, 2, :SVO_slices[2]]
            o_d = SVO_emb[i, 3, :SVO_slices[3]]

            hyperedge_attr = torch.cat([q_em, o_em_0, o_em_1, o_em_2, o_em_3, o_a, o_b, o_c, o_d], dim=0)

            # Run HGCN layers
            out = all_encoded_spans[i].to(device)
            for layer in self.hgcn_layers:
                out = F.dropout(out, self.dropout_rate, training=self.training)
                out = layer(out, hyperedge_index, hyperedge_attr=hyperedge_attr)
                out = F.relu(out)
            all_out.append(out)

        # Reconstruct outputs
        all_out_em_list = []
        for i in range(num_graph):
            f0 = index_ids[i, 0].item()
            f1 = index_ids[i, 1].item()
            f2 = index_ids[i, 2].item()
            f3 = index_ids[i, 3].item()
            f4 = index_ids[i, 4].item()

            option_a = torch.cat([all_out[i][:f0], all_out[i][f0:f1]], dim=0)
            option_b = torch.cat([all_out[i][:f0], all_out[i][f1:f2]], dim=0)
            option_c = torch.cat([all_out[i][:f0], all_out[i][f2:f3]], dim=0)
            option_d = torch.cat([all_out[i][:f0], all_out[i][f3:f4]], dim=0)

            max_len = encoded_spans.size(2)
            options = [option_a, option_b, option_c, option_d]
            for idx, option in enumerate(options):
                pad_size = max_len - option.size(0)
                if pad_size > 0:
                    padding = torch.zeros(pad_size, feature_dim, device=device)
                    option = torch.cat([option, padding], dim=0)
                options[idx] = option.unsqueeze(0)  # [1, max_len, feature_dim]
            all_out_em_list.extend(options)

        all_out_em = torch.cat(all_out_em_list, dim=0)  # [bs, max_len, feature_dim]

        return all_out_em


