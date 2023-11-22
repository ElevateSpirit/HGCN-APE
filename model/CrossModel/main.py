import json, os
import random
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import dgl
from tqdm import trange

from data import load_data_instances, DataIterator, idx2labels
from HGAT import CrossModel
from utils import get_huggingface_optimizer_and_scheduler, context_models, Metric, Writer, plot_attention_weights, plot_attn_loss
import math
from termcolor import colored


def setup_seed(seed):
    dgl.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def train(args):

    random.seed(args.random_seed)
    
    if not args.test_code:
        train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))
        random.seed(args.random_seed)
        random.shuffle(train_sentence_packs)
        random.seed(args.random_seed)
        
        dev_sentence_packs = json.load(open(args.prefix  + args.dataset + '/dev.json'))
        test_sentence_packs = json.load(open(args.prefix  + args.dataset +'/test.json'))
    else:
        train_sentence_packs = json.load(open(args.prefix  + args.dataset + '/test.json'))
        dev_sentence_packs = json.load(open(args.prefix  + args.dataset + '/dev.json'))
        test_sentence_packs = json.load(open(args.prefix  + args.dataset + '/test.json'))

    train_batch_count = math.ceil(len(train_sentence_packs)/args.batch_size)

    instances_train = load_data_instances(train_sentence_packs, args, "train", is_train=True)
    instances_dev = load_data_instances(dev_sentence_packs, args, "dev", is_train=False)
    instances_test = load_data_instances(test_sentence_packs, args, "test", is_train=False)
    del train_sentence_packs
    del dev_sentence_packs
    del test_sentence_packs
    random.seed(args.random_seed)
    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)
    testset = DataIterator(instances_test, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, args.model_name)):
        os.makedirs(os.path.join(args.output_dir, args.model_name))
    bertModel = context_models[args.bert_model_path]['model'].from_pretrained(args.bert_model_path, return_dict=False)

    model = CrossModel(args, bertModel).to(args.device)
    
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer, scheduler = get_huggingface_optimizer_and_scheduler(args, model, num_training_steps=train_batch_count * args.epochs,
                                                                    weight_decay=0.0, eps = 1e-8, warmup_step=0)

    best_joint_f1 = -1
    best_joint_epoch = 0
    best_joint_precision = 0
    joint_precision = 0
    best_joint_recall = 0
    
    for epoch in range(1, args.epochs+1):
        
        model.zero_grad()
        model.train()
        print('Epoch:{}'.format(epoch))
        losses = []
        pair_losses = []
        crf_losses = []
        
        for j in trange(trainset.batch_count):
            
            _, _, _, review_embedder_input, reply_embedder_input, _, _, tags, review_biotags, reply_biotags, review_mask, reply_mask, reply_lengths, review_lengths, dists, graphs = trainset.get_batch(j)
            """
            fill in -1 for attention template
            """
            attn_template = tags.clone().detach()
            for idx, (review_length, reply_length) in enumerate(zip(review_lengths, reply_lengths)):    
                for i in range(review_length):
                    attn_template[idx][i][:reply_length][attn_template[idx][i][:reply_length] == -1] = 0

            """
            dynamic negative sampling
            """
            for idx, (review_length, reply_length) in enumerate(zip(review_lengths, reply_lengths)):    
                for i in range(review_length):
                    if (tags[idx][i][:reply_length] == -1).sum().item() >= args.negative_sample:
                        neg_idx = (tags[idx][i][:reply_length] == -1).nonzero(as_tuple=False).view(-1)
                        choice = torch.multinomial(neg_idx.float(), args.negative_sample)
                        tags[idx][i][neg_idx[choice]] = 0
                    else:
                        tags[idx][i][:reply_length][tags[idx][i][:reply_length] == -1] = 0
            
            pair_logits, crf_loss = model(review_embedder_input, reply_embedder_input, review_mask, reply_mask, review_lengths, reply_lengths, review_biotags, reply_biotags, graphs, tags)
            logits_flatten = pair_logits.reshape(-1, pair_logits.size()[-1])
            tags_flatten = tags.reshape([-1])
            pair_loss = F.cross_entropy(logits_flatten, tags_flatten, ignore_index=-1, reduction='sum')
            attn_template += 1
            attn_template[attn_template==2] = -1
            loss = args.lamda * (args.pair_weight * pair_loss) + (1 - args.lamda) * crf_loss
            losses.append(loss.item())
            pair_losses.append(pair_loss.item()*args.pair_weight)
            crf_losses.append(crf_loss.item())
            loss.backward()

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if args.optimizer == 'adamw':
                scheduler.step()
            model.zero_grad()

        print('average loss {:.4f}'.format(np.average(losses)))
        print('average pairing loss {:.4f}'.format(np.average(pair_losses)))
        print('average crf loss {:.4f}'.format(np.average(crf_losses)))
        print(colored('Evaluating dev set: ', color='red'))
        joint_precision, joint_recall, joint_f1 = eval(model, devset, args)
        print(colored('Evaluating test set: ', color='red'))
        _, _, _ = eval(model, testset, args)


        if joint_f1 > best_joint_f1:
            model_path = args.model_dir + args.model_name + '.pt'
            torch.save(model, model_path)
            best_joint_precision = joint_precision
            best_joint_recall = joint_recall
            best_joint_f1 = joint_f1
            best_joint_epoch = epoch
    
    print(colored('Final evluation on dev set: ', color='red'))
    print('best epoch: {}\tbest dev precision: {:.5f}\tbest dev recall: {:.5f}\tbest dev f1: {:.5f}\n\n'.format(best_joint_epoch, best_joint_precision, best_joint_recall, best_joint_f1))


def eval(model, dataset, args, output_results=False):
    
    model.eval()
    
    with torch.no_grad():
        
        all_ids = []
        all_preds = []
        all_labels = []
        all_review_lengths = []
        all_reply_lengths = []
        all_review_bio_preds = []
        all_reply_bio_preds = []
        all_review_bio_golds = []
        all_reply_bio_golds = []
        
        for i in trange(dataset.batch_count):
            sentence_ids, _, _, review_embedder_input, reply_embedder_input, _, _, tags, review_biotags, reply_biotags, review_mask, reply_mask, reply_lengths, review_lengths, dists, graphs = dataset.get_batch(i)
            pair_logits, review_decode_idx, reply_decode_idx = model.decode(review_embedder_input, reply_embedder_input, review_mask, reply_mask, review_lengths, reply_lengths, graphs)
            pair_preds = torch.argmax(pair_logits, dim=3)
            all_ids.extend(sentence_ids)
            all_preds.extend(pair_preds.cpu().tolist())
            all_labels.extend(tags.cpu().tolist())
            all_review_lengths.extend(review_lengths.cpu().tolist())
            all_reply_lengths.extend(reply_lengths.cpu().tolist())
            all_review_bio_golds.extend(review_biotags.cpu().tolist())
            all_reply_bio_golds.extend(reply_biotags.cpu().tolist())
            all_review_bio_preds.extend(review_decode_idx.cpu().tolist())
            all_reply_bio_preds.extend(reply_decode_idx.cpu().tolist())


            attn_template = tags.clone().detach()
            attn_template += 1
            attn_template[attn_template==2] = -1

        metric = Metric(args, all_preds, all_labels, all_review_lengths, all_reply_lengths, all_review_bio_preds,
                        all_reply_bio_preds, all_review_bio_golds, all_reply_bio_golds)

        precision, recall, f1 = metric.score_uniontags()
        review_results = metric.score_review()
        reply_results = metric.score_reply()
        bio_results = metric.score_bio(review_results, reply_results)
        pair_results = metric.score_pair()

        print('Review\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(review_results[3], review_results[4],
                                                                review_results[5]))
        print('Reply\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(reply_results[3], reply_results[4],
                                                                reply_results[5]))
        print('Argument\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(bio_results[0], bio_results[1],
                                                                bio_results[2]))
        print('Pairing\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(pair_results[0], pair_results[1],
                                                            pair_results[2]))
        print('Overall\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

        if output_results:
            writer = Writer(args, all_preds, all_labels, all_review_lengths, all_reply_lengths, all_review_bio_preds,
                        all_reply_bio_preds, all_review_bio_golds, all_reply_bio_golds)
            writer.output_results()

    model.train()
    return precision, recall, f1,



def test(args):
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir, args.model_name)):
        os.makedirs(os.path.join(args.output_dir, args.model_name))

    print(colored('Final evluation on test set: ', color='red'))
    model_path = args.model_dir + args.model_name + '.pt'
    if args.device == 'cpu':
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path).to(args.device)
    model.eval()

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    instances = load_data_instances(sentence_packs, args, "test", is_train=False)
    testset = DataIterator(instances, args)
    eval(model, testset, args, output_results=True)
    
    """
    count number of parameters
    """
    num_param = 0
    for layer, weights in model.state_dict().items():
        if layer.startswith('embedder.bert'):
            continue
        prod = 1
        for dim in weights.size():
            prod *= dim
        num_param += prod
    print(colored('There are in total {} parameters within the model'.format(num_param), color='yellow'))

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="saved_models/",
                        help='model path prefix')
    parser.add_argument('--model_name', type=str, default="model1",
                        help='model name')
    parser.add_argument('--output_dir', type=str, default="./outputs",
                        help='test dataset outputs directory')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--dataset', type=str, default="rr-passage",
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=210,
                        help='max length of a sentence')
    parser.add_argument('--max_bert_token', type=int, default=200,
                        help='max length of bert tokens for one sentence')

    parser.add_argument('--random_seed', type=int, default=42,
                        help='set random seed')
    parser.add_argument('--encoding_scheme', type=str, default='BIO', choices=['BIO', 'IOBES'],
                        help='encoding scheme for linear CRF')
    parser.add_argument('--test_code', type=bool, default=False)

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_tokenizer_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--token_embedding', type=bool, default=True,
                        help='additional lstm embedding over pre-trained bert token embeddings')
    parser.add_argument('--freeze_bert', type=bool, default=True,
                        help='whether to freeze parameters of pre-trained bert model')
    parser.add_argument('--num_embedding_layer', type=int, default=1,
                        help='number of layers for token LSTM')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--hidden_dim', type=int, default=200,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--position_embeeding', type=bool, default=False)
    parser.add_argument('--layer_norm', type=bool, default=False,
                        help='whether apply layer normalization to RNN model')

    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate') 
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='maximum gradient norm used during backprop') 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam'],
                        help='optimizer choice') 
    parser.add_argument('--pair_weight', type=float, default=1,
                        help='pair loss weight coefficient for loss computation') 
    parser.add_argument('--attn_weight', type=float, default=1,
                        help='attention loss weight coefficient for loss computation')
    parser.add_argument('--ema', type=float, default=0.9,
                        help='EMA coefficient alpha')
    parser.add_argument('--lstm_share_param', type=bool, default=True,
                        help='whether to share same LSTM layer for review&reply embedding')
    parser.add_argument('--share_crf_param', type=bool, default=True,
                        help='whether to share same CRF layer for review&reply decoding')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--negative_sample', type=int, default=1000,
                        help='number of negative samples, 1000 means all') 

    parser.add_argument('--pair_threshold', type=float, default=0.5,
                        help='pairing threshold during evaluation')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=25,
                        help='training epoch number')
    parser.add_argument('--lamda', type=float, default=0.5)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument('--num_instances', type=int, default=-1,
                        help='number of instances to read')
    parser.add_argument('--device', type=str, default="cuda:1",
                        help='gpu or cpu')
    args = parser.parse_args()
    setup_seed(args.random_seed)
    if args.mode == 'train':
        train(args)
        test(args)
    else:
        test(args)
