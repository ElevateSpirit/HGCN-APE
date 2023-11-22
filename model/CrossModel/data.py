import dgl
import math
# from matplotlib.cbook import maxdict
import torch
import numpy as np
from transformers import *
from utils import context_models, START_TAG, STOP_TAG, PAD_TAG, sentiment2id, label2idx, idx2labels, semi_label2idx, semi_idx2labels, O
from utils import iobes_label2idx, iobes_idx2labels, convert_bio_to_iobes
import itertools
import re
import os
from collections import Counter, defaultdict
from tqdm import tqdm

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9

def get_spans(tags):
    """
    for spans
    """
    tags = tags.strip().split('<tag>')
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans

class Instance(object):
    
    def __init__(self, tokenizer, sentence_pack, args, is_train, graph):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.last_review = sentence_pack['split_idx']
        self.sents = self.sentence.strip().split(' <sentsep> ')
        self.review = self.sents[:self.last_review+1]
        self.reply = self.sents[self.last_review+1:]
        self.graph_dic = construct_graph(self.review, self.reply)
        self.sen_length = len(self.sents)
        self.review_length = self.last_review + 1
        self.reply_length = self.sen_length - self.last_review - 1
        self.graph = graph
        self.review_bert_tokens = []
        self.reply_bert_tokens = []
        self.review_num_tokens = []
        self.reply_num_tokens = []

        dist = np.zeros((self.review_length, self.reply_length), dtype=np.int)
        # compute relative position
        for i in range(self.review_length):
            dist[i, :] += i
        for j in range(self.reply_length):
            dist[:, j] -= j
        dist = np.abs(dist)
        # convert relative position to index
        for i in range(self.review_length):
            for j in range(self.reply_length):
                if dist[i, j] < 0:
                    dist[i, j] = dis2idx[-dist[i, j]] + 9
                else:
                    dist[i, j] = dis2idx[dist[i, j]]
        dist[dist == 0] = 19
        self.dist_input = dist

        for i, sent in enumerate(self.review):
            word_tokens = tokenizer.tokenize(" " + sent)
            input_ids = tokenizer.convert_tokens_to_ids(
                [tokenizer.cls_token] + word_tokens + [tokenizer.sep_token])
            self.review_bert_tokens.append(input_ids)
            self.review_num_tokens.append(min(len(word_tokens), args.max_bert_token-1))
        for i, sent in enumerate(self.reply):
            word_tokens = tokenizer.tokenize(" " + sent)
            input_ids = tokenizer.convert_tokens_to_ids(
                [tokenizer.cls_token] + word_tokens + [tokenizer.sep_token])
            self.reply_bert_tokens.append(input_ids)
            self.reply_num_tokens.append(min(len(word_tokens), args.max_bert_token-1))
        self.length = len(self.sents)
        if is_train:
            self.tags = torch.full((self.review_length, self.reply_length), -1, dtype=torch.long)
        else:
            self.tags = torch.zeros(self.review_length, self.reply_length).long()
        
        review_bio_list = [O] * self.review_length
        reply_bio_list = [O] * self.reply_length

        for triple in sentence_pack['triples']:
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            for l, r in aspect_span:
                for i in range(l, r+1):
                    if i == l:
                        review_bio_list[i] = 'B'
                    else:
                        review_bio_list[i] = 'I'

            for l, r in opinion_span:
                for i in range(l, r+1):
                    if i == l:
                        reply_bio_list[i-self.review_length] = 'B'
                    else:
                        reply_bio_list[i-self.review_length] = 'I'

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            self.tags[i][j-self.review_length] = 1
        
        if args.encoding_scheme == 'BIO' or not is_train:
            review_bio_list = [label2idx[label] for label in review_bio_list]
            reply_bio_list = [label2idx[label] for label in reply_bio_list]
            self.review_bio = torch.LongTensor(review_bio_list)
            self.reply_bio = torch.LongTensor(reply_bio_list)
        elif args.encoding_scheme == 'IOBES' and is_train:
            review_bio_list = [iobes_label2idx[label] for label in convert_bio_to_iobes(review_bio_list)]
            reply_bio_list = [iobes_label2idx[label] for label in convert_bio_to_iobes(reply_bio_list)]
            self.review_bio = torch.LongTensor(review_bio_list)
            self.reply_bio = torch.LongTensor(reply_bio_list)

def load_data_instances(sentence_packs, args,  mode, is_train):
    if not os.path.exists("saved_data"):
        os.mkdir("saved_data")
    
    if not os.path.exists(f"saved_data/instance_{args.dataset}_{mode}.pt"):
        print(f"saving instance_{mode}.pt")
        instances = list()
        tokenizer = context_models[args.bert_tokenizer_path]['tokenizer'].from_pretrained(args.bert_tokenizer_path)
        
        for sentence_pack in tqdm(sentence_packs, total=len(sentence_packs)):
            instances.append(Instance(tokenizer, sentence_pack, args, is_train, 0))
        
        torch.save(instances, f"saved_data/instance_{args.dataset}_{mode}.pt")
        if args.num_instances != -1:
            return instances[:args.num_instances]
        else:
            return instances
    else:
        instances = torch.load(f"saved_data/instance_{args.dataset}_{mode}.pt")
        if args.num_instances != -1:
            return instances[:args.num_instances]
        else:
            return instances

class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)
        self.max_bert_token = args.max_bert_token

    def __len__(self):
        return len(self.instances)

    def get_batch(self, index):
        sentence_ids = []
        reviews = []
        replies = []
        sens_lens = []
        lengths = []
        review_lengths = []
        reply_lengths = []
        dists = []
        graphs = []

        batch_size = min((index + 1) * self.args.batch_size, len(self.instances)) - index * self.args.batch_size
        max_review_num_sents = max([self.instances[i].review_length for i in range(index * self.args.batch_size,
                            min((index + 1) * self.args.batch_size, len(self.instances)))])
        max_reply_num_sents = max([self.instances[i].reply_length for i in range(index * self.args.batch_size,
                            min((index + 1) * self.args.batch_size, len(self.instances)))])
        max_review_sent_length = min(max([max(map(len, self.instances[i].review_bert_tokens)) for i in range(index * self.args.batch_size,
                            min((index + 1) * self.args.batch_size, len(self.instances)))]), self.max_bert_token)
        max_reply_sent_length = min(max([max(map(len, self.instances[i].reply_bert_tokens)) for i in range(index * self.args.batch_size,
                            min((index + 1) * self.args.batch_size, len(self.instances)))]), self.max_bert_token)

        review_bert_tokens = torch.zeros(batch_size, max_review_num_sents, max_review_sent_length, dtype=torch.long)
        reply_bert_tokens = torch.zeros(batch_size, max_reply_num_sents, max_reply_sent_length, dtype=torch.long)
        review_attn_masks = torch.zeros(batch_size, max_review_num_sents, max_review_sent_length, dtype=torch.long)
        reply_attn_masks = torch.zeros(batch_size, max_reply_num_sents, max_reply_sent_length, dtype=torch.long)
        review_masks = torch.zeros(batch_size, max_review_num_sents, dtype=torch.long)
        reply_masks = torch.zeros(batch_size, max_reply_num_sents, dtype=torch.long)
        tags = -torch.ones(batch_size, max_review_num_sents, max_reply_num_sents).long()
        review_biotags = torch.full((batch_size, max_review_num_sents), label2idx[PAD_TAG]).long()
        reply_biotags = torch.full((batch_size, max_reply_num_sents), label2idx[PAD_TAG]).long()
        review_num_tokens = torch.ones(batch_size, max_review_num_sents).long()
        reply_num_tokens = torch.ones(batch_size, max_reply_num_sents).long()

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            reviews.append(self.instances[i].review)
            replies.append(self.instances[i].reply)
            dists.append(self.instances[i].dist_input)
            # graphs.append(self.instances[i].graph)
            graph_dic = self.instances[i].graph_dic
            graph_dic[('rev', 'v2b', 'rep')].append((max_review_num_sents-1, max_reply_num_sents-1))
            graph_dic[('rep', 'b2v', 'rev')].append((max_reply_num_sents-1, max_review_num_sents-1))
            graph = dgl.heterograph(graph_dic)
            graphs.append(graph)
            sens_lens.append(self.instances[i].sen_length)
            lengths.append(self.instances[i].length)
            review_lengths.append(self.instances[i].review_length)
            reply_lengths.append(self.instances[i].reply_length)
            review_masks[i-index * self.args.batch_size, :self.instances[i].review_length] = 1
            reply_masks[i-index * self.args.batch_size, :self.instances[i].reply_length] = 1
            tags[i-index * self.args.batch_size, :self.instances[i].review_length, :self.instances[i].reply_length] = self.instances[i].tags
            review_biotags[i-index * self.args.batch_size, :self.instances[i].review_length] = self.instances[i].review_bio
            reply_biotags[i-index * self.args.batch_size, :self.instances[i].reply_length] = self.instances[i].reply_bio
            review_num_tokens[i-index * self.args.batch_size, :self.instances[i].review_length] = torch.LongTensor(self.instances[i].review_num_tokens)
            reply_num_tokens[i-index * self.args.batch_size, :self.instances[i].reply_length] = torch.LongTensor(self.instances[i].reply_num_tokens)

            for j in range(self.instances[i].review_length):
                length_filled = min(self.max_bert_token, len(self.instances[i].review_bert_tokens[j]))
                review_bert_tokens[i-index * self.args.batch_size, j, :length_filled] = \
                    torch.LongTensor(self.instances[i].review_bert_tokens[j][:length_filled])
                review_attn_masks[i-index * self.args.batch_size, j, :length_filled] = 1
            for k in range(self.instances[i].reply_length):
                length_filled = min(self.max_bert_token, len(self.instances[i].reply_bert_tokens[k]))
                reply_bert_tokens[i-index * self.args.batch_size, k, :length_filled] = \
                    torch.LongTensor(self.instances[i].reply_bert_tokens[k][:length_filled])
                reply_attn_masks[i-index * self.args.batch_size, k, :length_filled] = 1

        review_bert_tokens = review_bert_tokens.to(self.args.device)
        reply_bert_tokens = reply_bert_tokens.to(self.args.device)
        review_attn_masks = review_attn_masks.to(self.args.device)
        reply_attn_masks = reply_attn_masks.to(self.args.device)
        tags = tags.to(self.args.device)
        review_masks = review_masks.to(self.args.device)
        reply_masks = reply_masks.to(self.args.device)
        review_biotags = review_biotags.to(self.args.device)
        reply_biotags = reply_biotags.to(self.args.device)
        lengths = torch.tensor(lengths).to(self.args.device)
        review_lengths = torch.tensor(review_lengths).to(self.args.device)
        reply_lengths = torch.tensor(reply_lengths).to(self.args.device)
        # dists = torch.tensor(dists).to(self.args.device)
        dists = None
        review_num_tokens = review_num_tokens.to(self.args.device)
        reply_num_tokens = reply_num_tokens.to(self.args.device)
        graphs = dgl.batch(graphs).to(self.args.device)

        if self.args.token_embedding:
            return sentence_ids, reviews, replies, (review_bert_tokens, review_attn_masks, review_num_tokens), (reply_bert_tokens, reply_attn_masks, reply_num_tokens), lengths, sens_lens, tags, review_biotags, reply_biotags, review_masks, reply_masks, reply_lengths, review_lengths, dists, graphs
        else:
            return sentence_ids, reviews, replies, (review_bert_tokens, review_attn_masks), (reply_bert_tokens, reply_attn_masks), lengths, sens_lens, tags, review_biotags, reply_biotags, review_masks, reply_masks, reply_lengths, review_lengths, dists, graphs


def clean_sent(sent):
    sent = sent.lower()
    sent = sent.replace('[line_break_token]', ' ')
    sent = sent.replace('[tab_token]', ' ')
    sent = re.sub(', ', ' , ', sent).strip()
    sent = re.sub(': ', ' : ', sent).strip()
    sent = re.sub('; ', ' ; ', sent).strip()
    sent = re.sub('\*', ' ', sent).strip()
    sent = re.sub('" ', ' " ', sent).strip()
    sent = re.sub(' "', ' " ', sent).strip()
    sent = re.sub(" '", " ' ", sent).strip()
    sent = re.sub("' ", " ' ", sent).strip()
    sent = re.sub("\) ", " ) ", sent).strip()
    sent = re.sub(" \(", " ( ", sent).strip()

    sent = re.sub(' +', ' ', sent).strip()
    return sent


def remove_stopwords(word_list):
    from nltk.corpus import stopwords
    stopwords = set(stopwords.words('english'))
    for w in ['!',',','.','?', '(', ')', '"', "'", ';', ':']:
        stopwords.add(w)
    filtered_words = [word for word in word_list if word not in stopwords]
    return filtered_words
    

def construct_graph(reviews, replys):
    d = defaultdict(list)
    # review to review
    if len(reviews) == 1:
        d[('rev', 'sl-rev', 'rev')].append((0, 0))
        d[('rev', 'v2v', 'rev')].append((0, 0))
    else:
        for i in range(len(reviews)-1):
            d[('rev', 'v2v', 'rev')].append((i, i+1))
            d[('rev', 'v2v', 'rev')].append((i+1, i))
            d[('rev', 'sl-rev', 'rev')].append((i, i))
        d[('rev', 'sl-rev', 'rev')].append((i+1, i+1))
        
    # reply to reply
    if len(replys) == 1:
        d[('rep', 'sl-rep', 'rep')].append((0, 0))
        d[('rep', 'b2b', 'rep')].append((0, 0))
    else:
        for i in range(len(replys)-1):
            d[('rep', 'b2b', 'rep')].append((i, i+1))
            d[('rep', 'b2b', 'rep')].append((i+1, i))
            d[('rep', 'sl-rep', 'rep')].append((i, i))
        d[('rep', 'sl-rep', 'rep')].append((i+1, i+1))
    
    # review to reply
    for i in range(len(reviews)):
        review = reviews[i]
        sent_rev = clean_sent(review)
        sent_rev_tokens = sent_rev.split(' ')
        sent_rev_tokens = remove_stopwords(sent_rev_tokens)
        words_rev_dict = dict(Counter(sent_rev_tokens))
        for j in range(len(replys)):
            reply = replys[j]
            sent_rep = clean_sent(reply)
            sent_rep_tokens = sent_rep.split(' ')
            sent_rep_tokens = remove_stopwords(sent_rep_tokens)
            words_rep_dict = dict(Counter(sent_rep_tokens))
            co_occur_words = words_rev_dict.keys() & words_rep_dict.keys()
            co_occur_words_dict = {word: words_rev_dict[word] + words_rep_dict[word] for word in co_occur_words}
            if len(co_occur_words_dict) >= 2 and i < len(reviews) and j < len(replys):
                # review to reply or reply to review
                d[('rev', 'v2b', 'rep')].append((i, j))
                d[('rep', 'b2v', 'rev')].append((j, i))
 
    # graph = dgl.heterograph(d)
    
    return d