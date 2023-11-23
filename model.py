import dgl, math, torch
from dgl.nn import GATConv, PNAConv
import numpy as np
import networkx as nx
import torch.nn as nn
import dgl.function as fn


class RPGD(nn.Module):
    def __init__(self, args, n_user, n_item, n_category):
        super(RPGD, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_category = n_category

        self.n_hid = args.n_hid
        self.n_layers = args.n_layers
        self.mem_size = args.mem_size

        self.emb = nn.Parameter(torch.empty(n_user + n_item + n_category, self.n_hid))
        self.emb_social = nn.Parameter(torch.empty(n_user, self.n_hid))
        self.norm = nn.LayerNorm((args.n_layers + 1) * self.n_hid)

        self.layers = nn.ModuleList()
        for i in range(0, self.n_layers):
            self.layers.append(GNNLayer(self.n_hid, self.n_hid, self.mem_size, 5,
                                        layer_norm=True, dropout=args.dropout,
                                        activation=nn.LeakyReLU(0.2, inplace=True)))
        self.layers_social = nn.ModuleList()

        self.pool = GraphPooling('mean')

        self.reset_parameters()
        self.inner_social = nn.Sequential(*[nn.Linear(2 * (self.n_layers + 1) * self.n_hid, (self.n_layers + 1) * self.n_hid), nn.Dropout(args.dropout), nn.LeakyReLU(0.2, inplace=True)])
        self.score_social = nn.Sequential(*[nn.Linear((self.n_layers + 1) * self.n_hid, 1), nn.Dropout(args.dropout), nn.Sigmoid()])

        self.gatconvs = nn.ModuleList()
        for i in range(0, self.n_layers):
            # self.gatconvs.append(GATConv(self.n_hid, self.n_hid, 2))
            self.gatconvs.append(PNAConv(self.n_hid, self.n_hid, ['mean', 'max', 'sum'], ['identity', 'amplification'], 2))
        self.sfmx_loss = torch.nn.CrossEntropyLoss()
    def reset_parameters(self):
        nn.init.normal_(self.emb)
        nn.init.normal_(self.emb_social)

    def predict(self, user, item):
        return torch.einsum('bc, bc -> b', user, item) / user.shape[1]

    def forward(self, graph, ori_graph, usr1 = None, usr2 = None):
        x = self.emb
        sfm_loss = 0
        all_emb = [x]
        for idx, layer in enumerate(self.layers):
            x, label, category = layer(graph, x)                             # GNNLayer
            sfm_loss += self.sfmx_loss(category, label)
            all_emb += [x]
        x = torch.cat(all_emb, dim=1)
        x = self.norm(x)

        # Pooling
        guu_es = graph.edata['type'] == 0
        graph_uu = dgl.graph((graph.edges()[0][guu_es], graph.edges()[1][guu_es]), num_nodes=self.n_user)
        graph_uu = dgl.add_self_loop(graph_uu)
        user_pool = self.pool(graph_uu, x[:self.n_user])


        social_x = self.emb_social

        all_social_emb = [social_x]
        for idx, gatconv in enumerate(self.gatconvs):
            social_x = gatconv(ori_graph, social_x)
            # social_x = gatconv(graph_uu, social_x)
            # social_x = torch.mean(social_x, dim=1)
            all_social_emb += [social_x]
        social_x = torch.cat(all_social_emb, dim=1)
        social_x = self.norm(social_x)

        for i in range(self.n_user):
            x[i] = (x[i] + social_x[i]) / 2
        # x=self.norm(x)

        if usr1 != None:
            user1_rep = x[usr1] + user_pool[usr1]
            user2_rep = x[usr2] + user_pool[usr2]
            lat = torch.cat([user1_rep, user2_rep], dim=-1)
            lat = self.inner_social(lat) + user1_rep + user2_rep
            scores = torch.reshape(self.score_social(lat), [-1])
            preds = (social_x[usr1] * social_x[usr2]).sum(-1)


            # salLoss = 1e-7 * (torch.maximum(torch.tensor(0.0), 1.0 - scores * preds)).sum()
            salLoss = 1e-7 * (torch.maximum(torch.tensor(0.0), 1.0 - scores * preds)).sum()# - 2 * (user1_rep * social_x[usr1]).mean() + (user1_rep * social_x[usr2]).mean() + (user2_rep * social_x[usr1]).mean()

            return x, user_pool, sfm_loss, salLoss

        return x, user_pool


class GraphPooling(nn.Module):
    def __init__(self, pool_type):
        super(GraphPooling, self).__init__()
        self.pool_type = pool_type
        if pool_type == 'mean':
            self.reduce_func = fn.mean(msg='m', out='h')
        elif pool_type == 'max':
            self.reduce_func = fn.max(msg='m', out='h')
        elif pool_type == 'min':
            self.reduce_func = fn.min(msg='m', out='h')

    def forward(self, g, feat):
        with g.local_scope():
            g.ndata['x'] = feat
            g.update_all(fn.copy_u('x', 'm'), self.reduce_func)
            return g.ndata['h']


class BPRLoss(nn.Module):
    def __init__(self, lamb_reg):  # 0.0005
        super(BPRLoss, self).__init__()
        self.lamb_reg = lamb_reg

    def forward(self, pos_preds, neg_preds, *reg_vars):
        batch_size = pos_preds.size(0)

        bpr_loss = -0.5 * (pos_preds - neg_preds).sigmoid().log().sum() / batch_size
        reg_loss = torch.tensor([0.], device=bpr_loss.device)
        for var in reg_vars:
            reg_loss += self.lamb_reg * 0.5 * var.pow(2).sum()
        reg_loss /= batch_size

        loss = bpr_loss + reg_loss

        return loss, [bpr_loss.item(), reg_loss.item()]



class GNNLayer(nn.Module):
    def __init__(self,
                in_feats,
                out_feats,
                mem_size,
                num_rels,
                bias=True,
                activation=None,
                self_loop=True,
                dropout=0.0,
                layer_norm=False): # True
        super(GNNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.mem_size = mem_size

        self.num_rels = num_rels
        self.bias = bias                # True
        self.activation = activation    # LeakyRelu
        self.self_loop = self_loop      # True
        self.layer_norm = layer_norm    # True

        self.transformer_encoders = nn.TransformerEncoderLayer(in_feats, 4, batch_first=True, dim_feedforward=16)
        # self.transformer_encoders_interaction = nn.TransformerEncoderLayer(in_feats, 1, batch_first=True, dim_feedforward=16)
        self.transformer_predict = nn.Sequential(*[nn.Linear(in_feats, in_feats), nn.ReLU(), nn.Linear(in_feats, 1),
                             nn.Sigmoid()])

        self.mem_category = nn.Linear(in_feats, mem_size)

        self.CLS = nn.Embedding(self.mem_size, in_feats)
        # self.SEP = nn.Embedding(self.mem_size, in_feats)
        self.SEP = nn.Parameter(torch.randn(in_feats))
        # self.mem_weight = nn.Linear(mem_size, out_feats * in_feats, bias=False)
        self.mem_weight = nn.Linear(in_feats, out_feats * in_feats, bias=False)
        self.CLS.weight.data.normal_(mean=0, std=1)
        # self.SEP.weight.data.normal_(mean=0, std=1)


        if self.bias:
            self.h_bias = nn.Parameter(torch.empty(out_feats))
            nn.init.zeros_(self.h_bias)

        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feats)

        self.dropout = nn.Dropout(dropout)                              # None

        # self.i2u_map = nn.Linear(out_feats, out_feats, bias=False)
        # self.u2i_map = nn.Linear(out_feats, out_feats, bias=False)
        # self.i2c_map = nn.Linear(out_feats, out_feats, bias=False)
        # self.c2i_map = nn.Linear(out_feats, out_feats, bias=False)

    def message_func1(self, edges):
        msg = torch.empty((edges.src['h'].shape[0], self.out_feats),
                           device=edges.src['h'].device)

        self.category = []
        self.label = []
        for etype in range(self.num_rels):
            loc = edges.data['type'] == etype
            if loc.sum() == 0:
                continue
            src = edges.src['h'][loc]
            dst = edges.dst['h'][loc]


            h_src = torch.unsqueeze(src, 1)
            h_dst = torch.unsqueeze(dst, 1)
            mask_matrix_batch = torch.zeros(src.shape[0], 4, device='cuda:0')
            # mask_matrix_batch = torch.zeros(src.shape[0], 4, device='cpu')
            mem_emb = 0
            for idx in range(self.mem_size):
                CLS = torch.unsqueeze(self.CLS.weight[idx].repeat(h_src.shape[0], 1), 1)
                # SEP = torch.unsqueeze(self.SEP.weight[idx].repeat(h_src.shape[0], 1), 1)
                SEP = torch.unsqueeze(self.SEP.repeat(h_src.shape[0], 1), 1)
                transformer_embedding = torch.cat([CLS, h_src, SEP, h_dst], dim=1)
                # mem_emb += [
                #     self.transformer_predict(self.transformer_encoders(transformer_embedding, src_key_padding_mask=mask_matrix_batch)[:, 0, :])]
                # mem_emb += self.transformer_encoders(transformer_embedding, src_key_padding_mask=mask_matrix_batch)[:, 0, :]
                tmp = self.transformer_encoders(transformer_embedding, src_key_padding_mask=mask_matrix_batch)[:, 0, :]
                self.category.append(self.mem_category(tmp))
                self.label.extend([idx] * src.shape[0])
                mem_emb += tmp
            # mem_emb = torch.cat(mem_emb, dim=1)
            w = self.mem_weight(mem_emb)  # 65084 * 256
            w = w.view(-1, self.out_feats, self.in_feats)
            sub_msg = torch.einsum('boi, bi -> bo', w, src)
            # if etype == 1:
            #     msg[loc] = self.i2u_map(sub_msg)
            # elif etype == 2:
            #     msg[loc] = self.u2i_map(sub_msg)
            # elif etype == 3:
            #     msg[loc] = self.c2i_map(sub_msg)
            # elif etype ==4:
            #     msg[loc] = self.i2c_map(sub_msg)

            msg[loc] = sub_msg
        self.category = torch.cat(self.category, dim=0)
        return {'m': msg}

    def forward(self, g, feat):  # feat
        with g.local_scope():
            g.ndata['h'] = feat
            g.update_all(self.message_func1, fn.mean(msg='m', out='h'))
            # g.update_all(self.message_func2, fn.mean(msg='m', out='h'))

            node_rep = g.ndata['h']
            if self.layer_norm:
                node_rep = self.layer_norm_weight(node_rep)
            if self.bias:
                node_rep = node_rep + self.h_bias
            if self.self_loop:
                # h = self.node_ME(feat, feat)
                h_src = torch.unsqueeze(feat, 1)
                h_dst = torch.unsqueeze(feat, 1)
                mask_matrix_batch = torch.zeros(feat.shape[0], 4, device='cuda:0')
                # mask_matrix_batch = torch.zeros(feat.shape[0], 4, device='cpu')
                mem_emb = 0
                for idx in range(self.mem_size):
                    CLS = torch.unsqueeze(self.CLS.weight[idx].repeat(h_src.shape[0], 1), 1)
                    # SEP = torch.unsqueeze(self.SEP.weight[idx].repeat(h_src.shape[0], 1), 1)
                    SEP = torch.unsqueeze(self.SEP.repeat(h_src.shape[0], 1), 1)
                    transformer_embedding = torch.cat(
                        [CLS, h_src,
                         SEP, h_dst],
                        dim=1)
                    # mem_emb += [
                    #     self.transformer_predict(
                    #         self.transformer_encoders(transformer_embedding, src_key_padding_mask=mask_matrix_batch)[:, 0, :])]
                    mem_emb += self.transformer_encoders(transformer_embedding, src_key_padding_mask=mask_matrix_batch)[:, 0, :]
                # mem_emb = torch.cat(mem_emb, dim=1)

                h = self.mem_weight(mem_emb)  # 65084 * 256
                h = h.view(-1, self.out_feats, self.in_feats)
                h = torch.einsum('boi, bi -> bo', h, feat)

                node_rep = node_rep + h
            if self.activation:  # leakyrelu
                node_rep = self.activation(node_rep)

            node_rep = self.dropout(node_rep)   # None
            return node_rep, torch.LongTensor(self.label).cuda(), self.category
            # return node_rep, torch.LongTensor(self.label), self.category
