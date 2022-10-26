import argparse
import json
import time
from unittest import result

import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
import data_loader
import utils
from model import Model
from fastapi import FastAPI
import uvicorn

app = FastAPI()


class Trainer(object):
    def __init__(self, mode, device):
        self.model = model
        self.model = self.model.to(device)
        self.device = device

    def predict(self, predict_loader):
        self.model.eval()
        predict_result = []
        with torch.no_grad():
            s_s = time.time()
            bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length,_ = predict_loader
            m_s = time.time()
            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            m_e = time.time()
            outputs = torch.argmax(outputs, -1)
            print(outputs)
            print(f'data_batch:{m_s - s_s}\n model:{m_e - m_s}')
            return outputs, sent_length, _
                
                

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path,map_location='cpu'))


def predict_decode(outputs, length, texts):
    start_t = time.time()
    entities = []
    for index, (instance, l, text) in enumerate(zip(outputs, length, texts)):
        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(l):
            for j in range(i + 1, l):
                if instance[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        def convert_index_to_text(index, type):
            text = "-".join([str(i) for i in index])
            text = text + "-#-{}".format(type)
            return text

        for head in head_dict:
            find_entity(head, [], head_dict[head])
        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])
        tmp = (text,)
        entity_type_list = []
        for pre in predicts:
            pre = pre.split('-#-')
            print(pre)
            print(text)
            ind = pre[0].split('-')
            entity = text[int(ind[0]):int(ind[-1]) + 1]
            entity_type = config.vocab.id2label[int(pre[1])]
            if entity_type == '4':
                entity_type_list.append(entity)
            tmp += ((entity, entity_type, int(ind[0]), int(ind[-1])),)
        entities.append(tmp)
    result_dic = {'sentence':text,'ner':entities[-1],'4':entity_type_list}
    str_json = json.dumps(result_dic,ensure_ascii=False)
    print(entities)
    end_t = time.time()
    print(f"时长：{end_t - start_t}")
    logger.info(result_dic)
    return result_dic

@app.get("/title={text}")
def sentence_api(text):
    if isinstance(text, str):
        texts = [text]
    set_s = time.time()
    predict_dataset = data_loader.load_data_bert_predict(texts, config)
    set_e = time.time()
    predict_loader = data_loader.collate_fn_predict(predict_dataset)
    load_e = time.time()
    outputs, sent_length, texts_= trainer.predict(predict_loader)
    pred_e = time.time()
    print(f'dataset:{set_e - set_s}\n loader:{load_e - set_e}\n predict:{pred_e - load_e}')
    return predict_decode(outputs.cpu().numpy(), sent_length.cpu().numpy(), texts_)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/jd.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger
    config.label_num = 54
    config.tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    vocab = data_loader.Vocabulary()
    label2id = {}
    for i in range(0,config.label_num):
        label2id[f'{i+1}'] = i
    id2label = {v:k for k,v in label2id.items()}
    vocab.label2id = label2id
    vocab.id2label = id2label
    print(dict(vocab.label2id))
    print("=============================")
    #   config.label_num = len(vocab.label2id)
    config.vocab = vocab
    print(config)



    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)


    logger.info("Loading Data")
    updates_total = 0
    logger.info("Building Model")
    model = Model(config)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    trainer = Trainer(model, device)
    trainer.load("model.pt")
    # sentence_api('佳能NPG-67粉盒iRC3330C3325C3320C3520墨粉iR3020低容单色-青90GA')
    uvicorn.run(app, host="0.0.0.0", port=8077)
