import torch
import torch.nn as nn
import dgl.nn as dglnn
import math
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer
from termcolor import colored

from module.rnn import BiLSTMEncoder
from module.inferencer import LinearCRF
from module.embedder import Embedder, TokenEmbedder

from data import START_TAG, STOP_TAG, PAD_TAG, label2idx, idx2labels, iobes_idx2labels, iobes_label2idx



class JointModule(nn.Module):
    
    def __init__(self, args, input_dim, hidden_dim, lstm_share_param):
        super(JointModule, self).__init__()
        self.args = args
        self.lstm_share_param = lstm_share_param
        if lstm_share_param:
            self.lstm_encoder = BiLSTMEncoder(input_dim, hidden_dim)
        else:
            self.review_lstm_encoder = BiLSTMEncoder(input_dim, hidden_dim)
            self.reply_lstm_encoder = BiLSTMEncoder(input_dim, hidden_dim)

        # hgcn
        rel_name = ['v2v', 'b2b', 'v2b', 'b2v', 'sl-rep', 'sl-rev']
        self.activation = nn.ReLU()
        self.hgcn = nn.ModuleList([RelGraphConvLayer(
            args.hidden_dim,
            args.hidden_dim, 
            rel_name,
            num_bases=len(rel_name),
            activation=self.activation,
            self_loop=True,
            dropout=0.2)
            for _ in range(1)])
        
        self.tfg = PositionWisedTableEncoder(args)
        self.dropout = nn.Dropout(args.dropout)
        self.ln = nn.LayerNorm(hidden_dim)

    
    def forward(self, review_input, reply_input, table_input, review_seq_lens, reply_seq_lens, graphs):
        """
        Encoding the input with RNNs
        param:
        review_input: (batch_size, num_review_sents, input_dim)
        reply_input: (batch_size, num_reply_sents, input_dim)
        table_input: (batch_size, num_review_sents, num_reply_sents, hidden_dim*2)
        review_seq_lens: (batch_size, )
        reply_seq_lens: (batch_size, )
        review_input_mask: (batch_size, num_review_sents)
        reply_input_mask: (batch_size, num_reply_sents)
        return: 
        review_feature_out: (batch_size, num_review_sents, hidden_dim)
        reply_feature_out: (batch_size, num_reply_sents, hidden_dim)
        table_feature_out: (batch_size, num_review_sents, num_reply_sents, hidden_dim)
        """
        if self.args.lstm_share_param:
            review_lstm_feature = self.lstm_encoder(review_input, review_seq_lens)
            reply_lstm_feature = self.lstm_encoder(reply_input, reply_seq_lens)
        else:
            review_lstm_feature = self.review_lstm_encoder(review_input, review_seq_lens)
            reply_lstm_feature = self.reply_lstm_encoder(reply_input, reply_seq_lens)

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

        
        cross_feature = self.tfg(review_gcn_embedding, reply_gcn_embedding)
        grid_feature = self.ln(self.dropout(cross_feature) + table_input)

        return review_gcn_embedding, reply_gcn_embedding, grid_feature
    
    
    
class CrossModel(nn.Module):
    def __init__(self, args, bertModel):
        super(CrossModel, self).__init__()
        if args.token_embedding:
            self.embedder = TokenEmbedder(bertModel, args)
        else:
            self.embedder = Embedder(bertModel)
        self.encoder = nn.ModuleList([JointModule(args, args.bert_feature_dim, args.hidden_dim, args.lstm_share_param) if i == 0 \
                                      else JointModule(args, args.hidden_dim, args.hidden_dim, args.lstm_share_param) for i in range(args.layers)])
        self.share_crf_param = args.share_crf_param

        if args.encoding_scheme == 'IOBES':
            crf_label2idx, crf_idx2labels, iobes = iobes_label2idx, iobes_idx2labels, True
        else:
            crf_label2idx, crf_idx2labels, iobes = label2idx, idx2labels, False
        if args.share_crf_param:
            self.review_crf = LinearCRF(crf_label2idx, crf_idx2labels, START_TAG=START_TAG, STOP_TAG=STOP_TAG, PAD_TAG=PAD_TAG, iobes=iobes)
            self.reply_crf = self.review_crf
        else:
            self.review_crf = LinearCRF(crf_label2idx, crf_idx2labels, START_TAG=START_TAG, STOP_TAG=STOP_TAG, PAD_TAG=PAD_TAG, iobes=iobes)
            self.reply_crf = LinearCRF(crf_label2idx, crf_idx2labels, START_TAG=START_TAG, STOP_TAG=STOP_TAG, PAD_TAG=PAD_TAG, iobes=iobes)

        classifier = [nn.Linear(args.hidden_dim, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 2)]
        self.hidden2tag = nn.Sequential(*classifier)
        self.pair2tag = nn.Sequential(*classifier)
        self.hidden2biotag = nn.Linear(args.hidden_dim, len(crf_label2idx))
        self.initial_table_input = torch.zeros((1, 1, 1, args.hidden_dim)).to(args.device)
        
        self.args = args
    
    def forward(self, review_embedder_input, reply_embedder_input, review_input_mask, reply_input_mask, review_seq_lens, reply_seq_lens, review_bio_tags, reply_bio_tags, graphs, pair_tag):
        
        review_feature = self.embedder(*review_embedder_input)
        reply_feature = self.embedder(*reply_embedder_input)
        batch_size, review_num_sents, _ = review_feature.size()
        _, reply_num_sents, _ = reply_feature.size()

        grid_feature = self.initial_table_input.expand(batch_size, review_num_sents, reply_num_sents, -1)

        for encoder_module in self.encoder:
            review_feature, reply_feature, grid_feature = encoder_module(review_feature, reply_feature, grid_feature, review_seq_lens, reply_seq_lens, graphs)
            

        review_crf_input = self.hidden2biotag(review_feature)
        reply_crf_input = self.hidden2biotag(reply_feature)
        review_crf_loss = self.review_crf(review_crf_input, review_seq_lens, review_bio_tags, review_input_mask)
        reply_crf_loss = self.reply_crf(reply_crf_input, reply_seq_lens, reply_bio_tags, reply_input_mask)
        
        pair_output = self.hidden2tag(grid_feature)
        

        return pair_output, review_crf_loss + reply_crf_loss
    
    def decode(self, review_embedder_input, reply_embedder_input, review_input_mask, reply_input_mask, review_seq_lens, reply_seq_lens, graphs):
        
        review_feature = self.embedder(*review_embedder_input)
        reply_feature = self.embedder(*reply_embedder_input)
        batch_size, review_num_sents, _ = review_feature.size()
        _, reply_num_sents, _ = reply_feature.size()

        grid_feature = self.initial_table_input.expand(batch_size, review_num_sents, reply_num_sents, -1)
        dev_num = review_feature.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")


        for encoder_module in self.encoder:
             review_feature, reply_feature, grid_feature = encoder_module(review_feature, reply_feature, grid_feature, review_seq_lens, reply_seq_lens, graphs)
            
        
        review_crf_input = self.hidden2biotag(review_feature)
        reply_crf_input = self.hidden2biotag(reply_feature)
        _, review_decode_idx = self.review_crf.decode(review_crf_input, review_seq_lens)
        _, reply_decode_idx = self.reply_crf.decode(reply_crf_input, reply_seq_lens)
        pair_output = self.hidden2tag(grid_feature)

        return pair_output, review_decode_idx, reply_decode_idx
    
    
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
    