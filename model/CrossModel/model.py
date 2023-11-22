import torch
import torch.nn as nn
import dgl.nn as dglnn
import math
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer
from termcolor import colored

from module.rnn import BiLSTMEncoder, GRU2dLayer, BGRU2dLayer
from module.inferencer import LinearCRF
from module.embedder import Embedder, TokenEmbedder

from data import START_TAG, STOP_TAG, PAD_TAG, label2idx, idx2labels, iobes_idx2labels, iobes_label2idx



class HGAT(nn.Module):
    def __init__(self, args, bertModel):
        super(HGAT, self).__init__()
        self.embedder = TokenEmbedder(bertModel, args)
        if args.lstm_share_param:
            self.review_lstm_encoder = BiLSTMEncoder(args.bert_feature_dim, args.hidden_dim, num_lstm_layers=2)
            self.reply_lstm_encoder = BiLSTMEncoder(args.bert_feature_dim, args.hidden_dim, num_lstm_layers=2)
        else:
            self.review_lstm_encoder = BiLSTMEncoder(args.bert_feature_dim, args.hidden_dim, num_lstm_layers=2)
            self.reply_lstm_encoder = BiLSTMEncoder(args.bert_feature_dim, args.hidden_dim, num_lstm_layers=2)

        # hgcn
        rel_name = ['v2v', 'b2b', 'v2b', 'b2v', 'sl-rep', 'sl-rev']
        self.activation = nn.ReLU()
        self.hgcn = nn.ModuleList([RelGraphConvLayer(
            args.hidden_dim,
            args.hidden_dim, 
            rel_name,
            num_bases=len(rel_name),
            activation=self.activation,
            self_loop=False,
            dropout=0.2)
            for _ in range(args.layers)])

        # Layer Norm
        self.review_ln = nn.LayerNorm(args.hidden_dim)
        self.reply_ln = nn.LayerNorm(args.hidden_dim)
        
        self.tfg = PositionWisedTableEncoder(args)
        
        crf_label2idx, crf_idx2labels, iobes = label2idx, idx2labels, False
        if args.share_crf_param:
            self.review_crf = LinearCRF(crf_label2idx, crf_idx2labels, START_TAG=START_TAG, STOP_TAG=STOP_TAG, PAD_TAG=PAD_TAG, iobes=iobes)
            self.reply_crf = self.review_crf
        else:
            self.review_crf = LinearCRF(crf_label2idx, crf_idx2labels, START_TAG=START_TAG, STOP_TAG=STOP_TAG, PAD_TAG=PAD_TAG, iobes=iobes)
            self.reply_crf = LinearCRF(crf_label2idx, crf_idx2labels, START_TAG=START_TAG, STOP_TAG=STOP_TAG, PAD_TAG=PAD_TAG, iobes=iobes)

        classifier = [nn.Linear(args.hidden_dim, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 2)]
        self.hidden2tag = nn.Sequential(*classifier)
        self.hidden2biotag = nn.Linear(args.hidden_dim, len(crf_label2idx))
        self.args = args
    
    def forward(self, review_embedder_input, reply_embedder_input, review_input_mask, reply_input_mask, review_seq_lens, reply_seq_lens, review_bio_tags, reply_bio_tags, graphs, tags):
        review_feature = self.embedder(*review_embedder_input)
        reply_feature = self.embedder(*reply_embedder_input)

        if self.args.lstm_share_param:
            review_lstm_feature = self.review_lstm_encoder(review_feature, review_seq_lens)
            reply_lstm_feature = self.reply_lstm_encoder(reply_feature, reply_seq_lens)
        else:
            review_lstm_feature = self.review_lstm_encoder(review_feature, review_seq_lens)
            reply_lstm_feature = self.reply_lstm_encoder(reply_feature, reply_seq_lens)
    
        batch_size, review_len, hidden_dim = review_lstm_feature.shape 
        _, reply_len, _ = reply_lstm_feature.shape
        
        cross_mask = reply_input_mask.unsqueeze(1) * review_input_mask.unsqueeze(2)
        
        # heterogenous graph convolution network  
        sentence_review_list = []
        sentence_reply_list = []
        features = {"rev": review_lstm_feature.reshape(-1, hidden_dim), "rep": reply_lstm_feature.reshape(-1, hidden_dim)}
        for gcn_layer in self.hgcn:
            features = gcn_layer(graphs, features)
            sentence_review_list.append(features['rev'])
            sentence_reply_list.append(features['rep'])
        # # mean pooling
        review_gcn_embedding = torch.mean(torch.stack(sentence_review_list), 0).reshape(batch_size, review_len, hidden_dim)
        reply_gcn_embedding = torch.mean(torch.stack(sentence_reply_list), 0).reshape(batch_size, reply_len, hidden_dim)
            
        # Layer Norm
        review_gcn_embedding = self.review_ln(review_lstm_feature + review_gcn_embedding)
        reply_gcn_embedding = self.reply_ln(reply_lstm_feature + reply_gcn_embedding)
        
        grid_feature = self.tfg(review_gcn_embedding, reply_gcn_embedding)
        
        review_crf_input = self.hidden2biotag(review_gcn_embedding)
        reply_crf_input = self.hidden2biotag(reply_gcn_embedding)
        review_crf_loss = self.review_crf(review_crf_input, review_seq_lens, review_bio_tags, review_input_mask)
        reply_crf_loss = self.reply_crf(reply_crf_input, reply_seq_lens, reply_bio_tags, reply_input_mask)

        pair_output = self.hidden2tag(grid_feature)
        
        return pair_output, review_crf_loss + reply_crf_loss
    
    def decode(self, review_embedder_input, reply_embedder_input, review_input_mask, reply_input_mask, review_seq_lens, reply_seq_lens, graphs):
        
        review_feature = self.embedder(*review_embedder_input)
        reply_feature = self.embedder(*reply_embedder_input)

        if self.args.lstm_share_param:
            review_lstm_feature = self.review_lstm_encoder(review_feature, review_seq_lens)
            reply_lstm_feature = self.reply_lstm_encoder(reply_feature, reply_seq_lens)
        else:
            review_lstm_feature = self.review_lstm_encoder(review_feature, review_seq_lens)
            reply_lstm_feature = self.reply_lstm_encoder(reply_feature, reply_seq_lens)
    

        batch_size, review_len, hidden_dim = review_lstm_feature.shape 
        _, reply_len, _ = reply_lstm_feature.shape
        
        # heterogenous graph convolution network  
        sentence_review_list = []
        sentence_reply_list = []
        features = {"rev": review_lstm_feature.reshape(-1, hidden_dim), "rep": reply_lstm_feature.reshape(-1, hidden_dim)}
        for gcn_layer in self.hgcn:
            features = gcn_layer(graphs, features)
            sentence_review_list.append(features['rev'])
            sentence_reply_list.append(features['rep'])
        # # mean pooling
        review_gcn_embedding = torch.mean(torch.stack(sentence_review_list), 0).reshape(batch_size, review_len, hidden_dim)
        reply_gcn_embedding = torch.mean(torch.stack(sentence_reply_list), 0).reshape(batch_size, reply_len, hidden_dim)
            
        # Layer Norm
        review_gcn_embedding = self.review_ln(review_lstm_feature + review_gcn_embedding)
        reply_gcn_embedding = self.reply_ln(reply_lstm_feature + reply_gcn_embedding)
        
        grid_feature = self.tfg(review_gcn_embedding, reply_gcn_embedding)
        
        review_crf_input = self.hidden2biotag(review_gcn_embedding)
        reply_crf_input = self.hidden2biotag(reply_gcn_embedding)
        
        _, review_decode_idx = self.review_crf.decode(review_crf_input, review_seq_lens)
        _, reply_decode_idx = self.reply_crf.decode(reply_crf_input, reply_seq_lens)
        pair_output = self.hidden2tag(grid_feature)

        return pair_output, review_decode_idx, reply_decode_idx

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.args.device)
        
        return embeddings
    
    def rotary_position_embedding(self, input_feature):
        batch_size, seq_len, hidden_dim = input_feature.shape
        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, hidden_dim)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
        input_feature1 = torch.stack([-input_feature[..., 1::2], input_feature[..., ::2]], -1)
        input_feature1 = input_feature1.reshape(input_feature.shape)
        input_feature = input_feature * cos_pos + input_feature1 * sin_pos
        
        return input_feature


class PositionWisedTableEncoder(nn.Module):
    def __init__(self, args):
        super(PositionWisedTableEncoder, self).__init__()
        self.args = args
        self.table_transform = nn.Linear(args.hidden_dim * 2, args.hidden_dim)

    def forward(self, review, reply):
        _, num_review_sents, _ = review.size()
        _, num_reply_sents, _ = reply.size()
        review = self.rotary_position_embedding(review)
        reply = self.rotary_position_embedding(reply)

        expanded_review_embedding = review.unsqueeze(2).expand([-1, -1, num_reply_sents, -1])
        expanded_reply_embedding = reply.unsqueeze(1).expand([-1, num_review_sents, -1, -1])


        cross_feature = torch.cat((expanded_review_embedding, expanded_reply_embedding), dim=3)
        cross_feature = self.table_transform(cross_feature)
        cross_feature = F.relu(cross_feature)
        
        return cross_feature
    
    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.args.device)
        
        return embeddings
    
    def rotary_position_embedding(self, input_feature):
        batch_size, seq_len, hidden_dim = input_feature.shape
        pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, hidden_dim)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
        input_feature1 = torch.stack([-input_feature[..., 1::2], input_feature[..., ::2]], -1)
        input_feature1 = input_feature1.reshape(input_feature.shape)
        input_feature = input_feature * cos_pos + input_feature1 * sin_pos
        
        return input_feature
    
    
class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False, )
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}

    
