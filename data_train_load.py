import json
import pandas as pd
import jieba
from tqdm import tqdm
import random
def read_data(data_path):
    sentence = []
    ner = []
    types = []
    word = []
    all_data = []
    sentence_str = ''
    index_str = ''
    start_i = 0
    word_list = []
    type_list = []
    with open(data_path,'r',encoding='utf-8') as f:
        for i in tqdm(f.readlines()):
            if i == '\n':
                if index != []:
                    ner.append({'index':index,'type':r_l[-1]})
                start_w = 0
                start_i = 0
                sentence_word = jieba.lcut(sentence_str)
                for w in sentence_word:
                    word.append(list(range(start_w,start_w+len(w))))
                    start_w += len(w)
                all_data.append({'sentence':sentence,'ner':ner,'word':word})
                sentence = []
                types = []
                ner = []
                word = []
                sentence_str = ''
                index_str = ''
                
                
                
            else:
                i_l = i.split(' ')
                sentence_str += i_l[0]
                sentence.append(i_l[0])
                t_l = i_l[-1].replace('\n','')
                types.append(t_l)
                r_l = index_str.split('-')
                t_l = t_l.split('-')

                if t_l[0] == 'B':
                    if r_l[0] != '':
                        ner.append({'index':index,'type':r_l[-1]})
                        word_list.append(''.join(sentence[index[0]:index[-1]+1]))
                        type_list.append(r_l[-1])
                        
                    index = []
                    index.append(start_i)
                    index_str = i_l[-1].replace('\n','')
                elif t_l[0] == 'I' and t_l[-1] == r_l[-1]:
                    index.append(start_i)

                start_i += 1
        
        # wtype_dic = {'words':word_list,'types':type_list}
        # word_type = pd.DataFrame(wtype_dic)
        # word_type.to_csv('word_type.csv')
        random.shuffle(all_data)
        train_data = all_data[:int(0.8*len(all_data))]
        dev_data = all_data[int(0.8*len(all_data)):]
        with open('./data/jd/train.json','a',encoding='utf-8') as df:
            json.dump(train_data,df,ensure_ascii=False)
        df.close()
        with open('./data/jd/dev.json','a',encoding='utf-8') as df:
            json.dump(dev_data,df,ensure_ascii=False)
        df.close()


if __name__ == "__main__":
    read_data('train.txt')
