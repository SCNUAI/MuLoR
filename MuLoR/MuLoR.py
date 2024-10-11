import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
import transformers
from transformers import BertPreTrainedModel, RobertaModel, RobertaConfig, AutoModel, DebertaV2Config, AlbertConfig
from models import HGAT, GAT, HGCN, RGCN, Message_gcn
from layers import FFNLayer, masked_softmax, weighted_sum, ResidualGRU, ArgumentGCN
from eval_attention import SCAttention
from new_train_rhlf import in_check

class MuLoR(nn.Module):
    def __init__(self,
                 config,
                 max_rel_id,
                 feature_dim_list,
                 device,
                 use_pool: bool = True,
                 dropout_prob: float = 0.1,
                 token_encoder_type: str = "Roberta",
                 ) -> None:
        super(MuLoR, self).__init__()

        self.max_rel_id = max_rel_id
        self.Roberta = AutoModel.from_pretrained(config)
        config_temp = RobertaConfig.from_pretrained(config)
        self.use_pool = use_pool
        self.RGCN = RGCN(config_temp.hidden_size, config_temp.hidden_size, n_layers=2, dropout=0.1)
        self.HGCN = HGCN(config_temp.hidden_size, config_temp.hidden_size, n_layer=2, dropout=0.1)
        self.scale = 1e-5
        self.dim = config_temp.hidden_size
        self.weight = 0.6
        if self.use_pool:
            self.dropout = nn.Dropout(0.1)
        self.devices = device
        self.MLP = FFNLayer(config_temp.hidden_size * 3, config_temp.hidden_size, 1, dropout_prob)
        self.norm = nn.LayerNorm(config_temp.hidden_size)
        self._proj_sequence_h = nn.Linear(config_temp.hidden_size, 1, bias=False)
        self.classifer = nn.Linear(config_temp.hidden_size, 1)
        self.gru = ResidualGRU(config_temp.hidden_size, dropout_prob, 2)

    def split_into_spans_9(self, seq, seq_mask, split_bpe_ids, flat_passage_mask, flat_question_mask, device):
        def _consecutive(seq: list, vals: np.array):
            groups_seq = []
            output_vals = copy.deepcopy(vals)
            for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
                groups_seq.append(list(map(itemgetter(1), g)))
            output_seq = []
            for i, ids in enumerate(groups_seq):
                output_seq.append(ids[0])
                if len(ids) > 1:
                    output_vals[ids[0]:ids[-1] + 1] = min(output_vals[ids[0]:ids[-1] + 1])
            return groups_seq, output_seq, output_vals

        embed_size = seq.size(-1)
        device = seq.device
        encoded_spans = []
        span_masks = []
        edges = []
        node_in_seq_indices = []
        p_nodes_mask = []
        o_nodes_mask = []
        p_o_mask = []

        for item_seq_mask, item_seq, item_split_ids, p_mask, o_mask in zip(seq_mask, seq, split_bpe_ids,
                                                                           flat_passage_mask, flat_question_mask):
            p_len = p_mask.sum().item()
            o_len = o_mask.sum().item()
            if item_split_ids[:p_len].sum() == 0:
                item_split_ids[p_len - 1] = 1

            p_split_ids = item_split_ids[:p_len - 1].detach().cpu().numpy()
            o_split_ids = item_split_ids[p_len - 1:p_len + o_len - 1].detach().cpu().numpy()
            p_split_ids_indices = np.where(p_split_ids > 0)[0].tolist()
            o_split_ids_indices = np.where(o_split_ids > 0)[0].tolist()
            n_p_split_ids_indices = len(p_split_ids_indices)
            n_o_split_ids_indices = len(o_split_ids_indices) + 1
            p_node_mask = [1 for i in range(n_p_split_ids_indices)] + [0 for i in range(n_o_split_ids_indices)]
            o_node_mask = [0 for i in range(n_p_split_ids_indices)] + [1 for i in range(n_o_split_ids_indices)]
            p_nodes_mask.append(p_node_mask)
            o_nodes_mask.append(o_node_mask)
            item_seq_len = item_seq_mask.sum().item()
            item_seq = item_seq[:item_seq_len]
            item_split_ids = item_split_ids[:item_seq_len]
            item_split_ids = item_split_ids.detach().cpu().numpy()
            split_ids_indices = np.where(item_split_ids > 0)[0].tolist()
            grouped_split_ids_indices, split_ids_indices, item_split_ids = _consecutive(
                split_ids_indices, item_split_ids)
            n_split_ids = len(split_ids_indices)
            po_mask = []
            item_spans, item_mask = [], []
            item_edges = []
            item_node_in_seq_indices = []
            item_edges.append(item_split_ids[split_ids_indices[0]])

            for i in range(n_split_ids):
                if i == n_split_ids - 1:
                    span = item_seq[split_ids_indices[i] + 1:]
                    if not len(span) == 0:
                        item_spans.append(span.sum(0))
                        item_mask.append(1)
                        if len(p_split_ids_indices) != 0:
                            if split_ids_indices[i] < p_split_ids_indices[-1]:
                                po_mask.append(1)
                            else:
                                po_mask.append(2)
                        else:
                            po_mask.append(1)
                else:
                    span = item_seq[split_ids_indices[i] + 1:split_ids_indices[i + 1]]
                    if not len(span) == 0:
                        item_spans.append(span.sum(0))
                        item_mask.append(1)
                        item_edges.append(item_split_ids[split_ids_indices[i + 1]])
                        item_node_in_seq_indices.append([k for k in range(grouped_split_ids_indices[i][-1] + 1,
                                                                          grouped_split_ids_indices[i + 1][0])])
                        if len(p_split_ids_indices) != 0:
                            if split_ids_indices[i] < p_split_ids_indices[-1]:
                                po_mask.append(1)
                            else:
                                po_mask.append(2)
                        else:
                            po_mask.append(1)
            p_o_mask.append(po_mask)
            encoded_spans.append(item_spans)
            span_masks.append(item_mask)
            edges.append(item_edges)
            node_in_seq_indices.append(item_node_in_seq_indices)

        max_nodes = max(map(len, span_masks))
        span_masks = [spans + [0] * (max_nodes - len(spans)) for spans in span_masks]
        span_masks = torch.from_numpy(np.array(span_masks))
        span_masks = span_masks.to(self.devices).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)
        encoded_spans = [spans + [pad_embed] * (max_nodes - len(spans)) for spans in encoded_spans]
        encoded_spans = [torch.stack(lst, dim=0) for lst in encoded_spans]
        encoded_spans = torch.stack(encoded_spans, dim=0)
        encoded_spans = encoded_spans.to(self.devices).float()

        truncated_edges = [item[1:-1] for item in edges]

        return encoded_spans, span_masks, truncated_edges, node_in_seq_indices, p_nodes_mask, o_nodes_mask, p_o_mask

    def get_adjacency_matrices_2(self, edges: List[List[int]], n_nodes: int, device):
        batch_size = len(edges)
        argument_graph = torch.zeros((batch_size, n_nodes, n_nodes), device=device)
        punct_graph = torch.zeros((batch_size, n_nodes, n_nodes), device=device)
        for b, sample_edges in enumerate(edges):
            for i, edge_value in enumerate(sample_edges):
                if edge_value == 1:
                    if i + 2 < n_nodes:
                        argument_graph[b, i + 1, i + 2] = 1
                elif edge_value == 2:
                    if i + 1 < n_nodes:
                        argument_graph[b, i, i + 1] = 1
                elif edge_value == 3:
                    if i + 1 < n_nodes:
                        argument_graph[b, i + 1, i] = 1
                elif edge_value == 4:
                    if i + 1 < n_nodes:
                        argument_graph[b, i, i + 1] = 1
                        argument_graph[b, i + 1, i] = 1
                elif edge_value == 5:
                    if i + 1 < n_nodes:
                        punct_graph[b, i, i + 1] = 1
                        punct_graph[b, i + 1, i] = 1
        return argument_graph, punct_graph

    def creat_SVO_id_batch(self, SVO_ids):
        SVO_id_batch = []
        bsz = SVO_ids.size(0)
        for bs in range(bsz):
            SVO_id_list = []
            for id in SVO_ids[bs, :].detach().cpu().numpy():
                if id != 1:
                    SVO_id_list.append([id])
                else:
                    SVO_id_list.append([])
            SVO_id_batch.append(SVO_id_list)
        return SVO_id_batch

    def create_graph(self, sequence_output, flat_punct_bpe_ids, flat_argument_bpe_ids, flat_attention_mask,
                     flat_input_ids, flat_passage_mask, flat_question_mask, device):
        new_punct_id = self.max_rel_id + 1
        new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids
        _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids

        overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
        flat_all_bpe_ids = _flat_all_bpe_ids * (
                1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask

        encoded_spans, span_mask, edges, node_in_seq_indices, p_nodes_mask, o_nodes_mask, p_o_mask = self.split_into_spans_9(
            sequence_output,
            flat_attention_mask,
            flat_all_bpe_ids, flat_passage_mask, flat_question_mask, device=self.devices)

        bsz = encoded_spans.size(0)
        max_node = encoded_spans.size(1)
        ids_in_sentence_nodes = []
        for bs in range(bsz):
            ids_in_sentence_nodes_list = []
            for t in range(max_node):
                ids_in_sentence_nodes_list.append([])
            bpe_list = flat_all_bpe_ids[bs, :].detach().cpu().numpy()
            input_ids_list = flat_input_ids[bs, :].detach().cpu().numpy()
            k = 0
            for i in range(len(bpe_list)):
                if bpe_list[i] == 0:
                    if k <= max_node - 1:
                        ids_in_sentence_nodes_list[k].append(input_ids_list[i])
                    continue
                if bpe_list[i] == -1:
                    break
                if bpe_list[i] == 4:
                    if k <= max_node - 1 and ids_in_sentence_nodes_list[k] == []:
                        ids_in_sentence_nodes_list[k].append(input_ids_list[i])
                        continue
                    else:
                        if k <= max_node - 2:
                            k += 1
                            ids_in_sentence_nodes_list[k].append(input_ids_list[i])
                        continue
                if bpe_list[i] == 5:
                    if k <= max_node - 1 and ids_in_sentence_nodes_list[k] == []:
                        continue
                    else:
                        k += 1
                        continue
            ids_in_sentence_nodes.append(ids_in_sentence_nodes_list)

        argument_graph, punctuation_graph = self.get_adjacency_matrices_2(edges, n_nodes=encoded_spans.size(1),
                                                                          device=encoded_spans.device)
        eyes = torch.eye(max_node, max_node).unsqueeze(0).repeat(bsz, 1, 1).to(self.devices)
        graph = argument_graph + punctuation_graph + eyes
        graph_mask = (graph > 0)
        graph = torch.where(graph_mask, torch.ones_like(graph), torch.zeros_like(graph))
        graph = graph.to(self.devices)

        return encoded_spans, graph, ids_in_sentence_nodes, span_mask, node_in_seq_indices, argument_graph, punctuation_graph, p_nodes_mask, o_nodes_mask, p_o_mask

    def create_similar_adj(self, feature_word, device):
        bsz, length, emb_size = feature_word.size()
        non_zero_mask = (feature_word.abs().sum(dim=-1) > 0).float()
        norm_feature_word = feature_word / (feature_word.norm(dim=-1, keepdim=True) + 1e-8)
        cosine_similarity = torch.bmm(norm_feature_word, norm_feature_word.transpose(1, 2))
        adj_matrix = (cosine_similarity > 0.5).float() * (non_zero_mask.unsqueeze(2) * non_zero_mask.unsqueeze(1))
        adj_matrix = adj_matrix + torch.eye(length, device=device).unsqueeze(0)
        return adj_matrix

    def create_word2sentence(self, word_ids_batch, ids_in_sentence_nodes, device):
        bsz = len(word_ids_batch)
        word_node_num = len(word_ids_batch[0])
        sent_node_num = len(ids_in_sentence_nodes[0])
        word2sent_adj = torch.zeros(bsz, word_node_num, sent_node_num).to(self.devices)
        sent2word_adj = torch.zeros(bsz, sent_node_num, word_node_num).to(self.devices)

        for bs in range(bsz):
            adj_batch = torch.zeros(word_node_num, sent_node_num).to(self.devices)
            word_ids = word_ids_batch[bs]
            sent_ids = ids_in_sentence_nodes[bs]
            for i in range(word_node_num):
                word_set = set(word_ids[i])
                for j in range(sent_node_num):
                    if word_set.intersection(sent_ids[j]):
                        adj_batch[i][j] = 1
            word2sent_adj[bs] = adj_batch
            sent2word_adj[bs] = adj_batch.T

        return word2sent_adj.to(self.devices), sent2word_adj.to(self.devices)

    def SSL(self, emb_hgnn, emb_lgcn):
        pos = torch.sum(emb_hgnn * emb_lgcn, dim=1)
        neg1 = torch.sum(emb_lgcn * emb_hgnn[torch.randperm(emb_hgnn.size()[0])], dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos), reduction='sum')
        neg_loss = F.binary_cross_entropy_with_logits(neg1, torch.zeros_like(neg1), reduction='sum')
        con_loss = pos_loss + neg_loss
        return con_loss

    def create_sent2sent_type3(self, word2sent_adj, device):
        sent2sent_adj = torch.bmm(word2sent_adj.transpose(1, 2), word2sent_adj)
        sent2sent_adj = (sent2sent_adj > 0).float()
        sent_node_num = sent2sent_adj.size(1)
        sent2sent_adj = sent2sent_adj + torch.eye(sent_node_num, device=device).unsqueeze(0)
        return sent2sent_adj.to(self.devices)

    def get_gcn_info_vector(self, indices, node, size, device):
        batch_size = size[0]
        gcn_info_vec = torch.zeros(size=size, dtype=torch.float, device=device)
        for b in range(batch_size):
            for ids, emb in zip(indices[b], node[b]):
                gcn_info_vec[b, ids] = emb
            gcn_info_vec[b, 0] = node[b].mean(0)
        return gcn_info_vec

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

        num_choices = input_ids.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_key_segid = key_segid.view(-1, key_segid.size(-1)) if key_segid is not None else None
        flat_passage_mask = passage_mask.view(-1, passage_mask.size(-1)) if passage_mask is not None else None
        flat_question_mask = question_mask.view(-1, question_mask.size(-1)) if question_mask is not None else None
        flat_argument_bpe_ids = argument_bpe_ids.view(-1, argument_bpe_ids.size(-1)) if argument_bpe_ids is not None else None
        flat_punct_bpe_ids = punct_bpe_ids.view(-1, punct_bpe_ids.size(-1)) if punct_bpe_ids is not None else None
        flat_SVO_ids = SVO_ids.view(-1, SVO_ids.size(-1)) if SVO_ids is not None else None
        flat_keyword_ids = keytokensids.view(-1, keytokensids.size(-1)) if keytokensids is not None else None
        flat_SVO_mask = SVO_mask.view(-1, SVO_mask.size(-1)) if SVO_mask is not None else None
        flat_keyword_mask = keymask.view(-1, keymask.size(-1)) if keymask is not None else None

        corpus_outputs = self.Roberta(flat_input_ids, attention_mask=flat_attention_mask)
        sequence_output = corpus_outputs[0]
        pooled_output = corpus_outputs[1]

        SVO_outputs = self.Roberta(flat_SVO_ids, attention_mask=flat_SVO_mask, token_type_ids=None)
        SVO_emb = SVO_outputs[0]

        SVO_id_batch = self.creat_SVO_id_batch(flat_SVO_ids)
        SVO_node_mask = torch.zeros((len(SVO_id_batch), len(SVO_id_batch[0]))).to(self.devices)
        for bs in range(SVO_node_mask.size(0)):
            for j in range(len(SVO_id_batch[bs])):
                if SVO_id_batch[bs][j] != []:
                    SVO_node_mask[bs, j] = 1
        SVO_truncation = int(max(SVO_node_mask.sum(1)).item())
        SVO_node_mask = SVO_node_mask[:, :SVO_truncation]
        SVO_emb = SVO_emb[:, :SVO_truncation, :]
        SVO_id_batch = [span[:SVO_truncation] for span in SVO_id_batch]

        encoded_spans, graph, ids_in_sentence_nodes, sent_mask, node_in_seq_indices, argument_graph, punct_graph, p_nodes_mask, o_nodes_mask, p_o_mask = self.create_graph(
            sequence_output, flat_punct_bpe_ids,
            flat_argument_bpe_ids, flat_attention_mask, flat_input_ids, flat_passage_mask, flat_question_mask,
            device=self.devices)
        if sentence_embed is not None:
            encoded_spans = in_check(encoded_spans, sentence_embed)

        feature_list = [encoded_spans, SVO_emb]

        words2words_adj = self.create_similar_adj(SVO_emb, device=self.devices)
        word2sent_adj, sent2word_adj = self.create_word2sentence(SVO_id_batch, ids_in_sentence_nodes,
                                                                device=self.devices)
        sent2sent_adj3 = self.create_sent2sent_type3(word2sent_adj, device=self.devices)

        sent2sent_adj = graph + sent2sent_adj3
        sent2sent_mask = (sent2sent_adj > 0)
        sent2sent_adj = torch.where(sent2sent_mask, torch.ones_like(sent2sent_adj), torch.zeros_like(sent2sent_adj))
        sent2sent_adj = sent2sent_adj.to(self.devices)

        aug_pun_adj = [argument_graph, punct_graph]
        type_sent_adj = [sent2sent_adj, sent2word_adj]
        type_word_adj = [word2sent_adj, words2words_adj]
        adj_list = [type_sent_adj, type_word_adj]
        sent_rep_HGCN = self.HGCN(feature_list, adj_list, pooled_output, p_nodes_mask, o_nodes_mask, p_o_mask,
                                  flat_SVO_mask)
        sent_rep_GCN = self.RGCN(feature_list, adj_list, aug_pun_adj, pooled_output, p_nodes_mask, o_nodes_mask)

        hgcn_emb = sent_rep_HGCN.view(-1, self.dim)
        rgcn_emb = sent_rep_GCN.view(-1, self.dim)
        cl_loss = self.SSL(hgcn_emb, rgcn_emb)
        sent_rep = self.weight * sent_rep_HGCN + (1-self.weight) * sent_rep_GCN
        graph_info_vec = self.get_gcn_info_vector(node_in_seq_indices, sent_rep, size=sequence_output.size(),
                                                  device=sequence_output.device)

        sequence_output = self.gru(self.norm(sequence_output + graph_info_vec))
        sequence_h2_weight = self._proj_sequence_h(sequence_output).squeeze(-1)
        passage_h2_weight = masked_softmax(sequence_h2_weight.float(), flat_passage_mask.float())
        passage_h2 = weighted_sum(sequence_output, passage_h2_weight)
        question_h2_weight = masked_softmax(sequence_h2_weight.float(), flat_question_mask.float())
        question_h2 = weighted_sum(sequence_output, question_h2_weight)

        if self.use_pool:
            pooled_output = self.dropout(pooled_output)
        output_feats = torch.cat([passage_h2, question_h2, pooled_output], dim=1)
        logit = self.MLP(output_feats)
        reshaped_logits = logit.view(-1, num_choices)
        outputs = (reshaped_logits,)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss + self.scale * cl_loss,) + outputs
        return outputs
