import json
import os
from random import randint

import spacy
from tqdm import tqdm

PER_list_set = ['Jennifer', 'Damien', 'Paula', 'Wexell', 'Rainwater', 'Bill', 'Bob', 'Munroe','Mike','John','Marry','Jeney','Hanson']
GPE_list_set = ['Mexico City', 'KingdomG', 'Kingdom F', 'Kingdom', 'Caladia', 'Panama City', 'Hong Kong']
NORP_list_set = ['Cantonese', 'North American','Fujianese','Hananese','South American','South Africa']
ORG_list_set = ['Funston', 'the Hall of Fame', 'Antibiotics', 'Eagles', 'Funston Stores', 'The Calex Telecommunications Company', 'air express']
def replace(dict_temp,type,corpus,key,i):
    flag=0
    if type=='PERSON':
        # for i, key in enumerate(dict_temp):
        marge = 0
        for j in range(len(dict_temp[key])):
            start = dict_temp[key][j][0]+marge
            end = start + len(key)
            corpus = corpus[:start] + PER_list_set[i] + corpus[end:]
            marge+=(len(PER_list_set[i])-len(key))
            flag+=1
    elif type=='GPE':
        # for i,key in enumerate(dict_temp):
        marge = 0
        for j in range(len(dict_temp[key])):
            start = dict_temp[key][j][0]+marge
            end = start+len(key)
            corpus = corpus[:start] + GPE_list_set[i] + corpus[end:]
            marge+=(len(GPE_list_set[i])-len(key))
            flag += 1
    elif type=='NORP':
        # for i,key in enumerate(dict_temp):
        marge = 0
        for j in range(len(dict_temp[key])):
            start = dict_temp[key][j][0]+marge
            end = start+len(key)
            corpus = corpus[:start] + NORP_list_set[i] + corpus[end:]
            marge+=(len(NORP_list_set[i])-len(key))
            flag += 1
    elif type=='ORG':
        # for i,key in enumerate(dict_temp):
        marge = 0
        for j in range(len(dict_temp[key])):
            start = dict_temp[key][j][0]+marge
            end = start+len(key)
            corpus = corpus[:start] + ORG_list_set[i] + corpus[end:]
            marge+=(len(ORG_list_set[i])-len(key))
            flag += 1
    return corpus,flag
def findKeyword(tokens,type,set_ex=None):
    PER_set = set()
    GPE_set = set()
    NORP_set = set()
    ORG_set = set()
    PER_list = []
    GPE_list = []
    NORP_list = []
    ORG_list = []
    PER_dict = dict()
    GPE_dict = dict()
    NORP_dict = dict()
    ORG_dict = dict()
    if type == 'PERSON':
        for tok in tokens.ents:
            if tok.label_ == 'PERSON':
                if '@@' not in tok.text:
                    # print('to',tok.start_char)
                    PER_set.add(tok.text)
                    PER_list.append((tok.text, (tok.start_char, tok.__len__())))
        for key, value in PER_list:
            PER_dict.setdefault(key, []).append(value)
        return PER_set,PER_dict
    elif type == 'GPE':
        for tok in tokens.ents:

            if tok.label_ == 'GPE':
                if '@@' not in tok.text:

                    GPE_set.add(tok.text)
                    GPE_list.append((tok.text, (tok.start_char, tok.__len__())))
        for key, value in GPE_list:
            GPE_dict.setdefault(key, []).append(value)
        return GPE_set,GPE_dict
    elif type == 'NORP':
        for tok in tokens.ents:
            if tok.label_ == 'NORP':
                if '@@' not in tok.text:

                    NORP_set.add(tok.text)
                    NORP_list.append((tok.text, (tok.start_char, tok.__len__())))
        for key, value in NORP_list:
            NORP_dict.setdefault(key, []).append(value)
        return NORP_set,NORP_dict
    elif type == 'ORG':
        for tok in tokens.ents:
            if tok.label_=='ORG':
                if '@@' not in tok.text:

                    ORG_set.add(tok.text)
                    ORG_list.append((tok.text,(tok.start_char,tok.__len__())))
        for key,value in ORG_list:
            ORG_dict.setdefault(key,[]).append(value)
        return ORG_set,ORG_dict

def updataKeyword(tokens,type,set_ex,key,dict_temp):
    PER_list = []
    GPE_list = []
    NORP_list = []
    ORG_list = []
    PER_dict = dict()
    GPE_dict = dict()
    NORP_dict = dict()
    ORG_dict = dict()

    temp= dict_temp[key]

    del dict_temp[key]

    flag= 0

    if type == 'PERSON':
        for tok in tokens.ents:
            if tok.label_ == 'PERSON' and tok.text==key:

                if '@@' not in tok.text:
                    flag = 1
                    PER_list.append((tok.text, (tok.start_char, tok.__len__())))
        if flag == 1:
            for key, value in PER_list:
                PER_dict.setdefault(key, []).append(value)
            if PER_dict[key]!=None:
                dict_temp[key] = PER_dict[key]
        else:
            dict_temp[key]=temp

    elif type == 'GPE':
        for tok in tokens.ents:
            if tok.label_ == 'GPE' and tok.text==key:
                if '@@' not in tok.text:
                    flag = 1
                    GPE_list.append((tok.text, (tok.start_char, tok.__len__())))
        if flag == 1:
            for key, value in GPE_list:
                GPE_dict.setdefault(key, []).append(value)
            if GPE_dict[key]!=None:
                dict_temp[key] = GPE_dict[key]
        else:
            dict_temp[key]=temp

    elif type == 'NORP':
        for tok in tokens.ents:
            if tok.label_ == 'NORP' and tok.text==key:
                if '@@' not in tok.text:
                    flag = 1
                    NORP_list.append((tok.text, (tok.start_char, tok.__len__())))
        if flag == 1:
            for key, value in NORP_list:
                NORP_dict.setdefault(key, []).append(value)
            if NORP_dict[key]!=None:
                dict_temp[key] = NORP_dict[key]
        else:
            dict_temp[key]=temp


    elif type == 'ORG':
        for tok in tokens.ents:
            if tok.label_=='ORG' and tok.text==key:
                if '@@' not in tok.text :
                    flag = 1
                    ORG_list.append((tok.text,(tok.start_char,tok.__len__())))
        if flag == 1:
            for key,value in ORG_list:
                ORG_dict.setdefault(key,[]).append(value)

            if ORG_dict[key]!=None:
                dict_temp[key] = ORG_dict[key]
        else:
            dict_temp[key]=temp
    return dict_temp

class InputExample(object):
    def __init__(self, example_id, question, contexts, endings, label=None):
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label
def d_read_json(input_file):
    with open(input_file, "r") as f:
        lines = json.load(f)
    return lines
def create_examples(lines):
    examples = []
    for d in lines:
        context = d['context']
        question = d['question']
        answers = d['answers']
        label = 0 if type == "test" else d['label'] # for test set, there is no label. Just use 0 for convenience.
        id_string = int(d['id_string'][6:])
        examples.append(
            InputExample(
                example_id = id_string,
                question = question,
                contexts=[context, context, context, context],  # this is not efficient but convenient
                endings=[answers[0], answers[1], answers[2], answers[3]],
                label = label
                )
            )
    return examples
def _read_txt( input_file):
    with open(input_file, "r",encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def get_labels(self):
    return [0, 1, 2, 3]


def _create_examples( lines):
    label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    assert len(lines) % 8 == 0, 'len(lines)={}'.format(len(lines))
    n_examples = int(len(lines) / 8)
    examples = []
    # for i, line in enumerate(examples):
    for i in range(n_examples):

        label_str = lines[i * 8 + 1].strip()
        context = lines[i * 8 + 2].strip()
        question = lines[i * 8 + 3].strip()
        answers = lines[i * 8 + 4: i * 8 + 8]
        examples.append(
            InputExample(
                example_id=" ",  # no example_id in LogiQA.
                question=question,
                contexts=[context, context, context, context],
                endings=[item.strip() for item in answers],
                label=label_str
            )
        )
    assert len(examples) == n_examples
    return examples

def main():
    file = os.path.join('data_reclor/', 'Train' + ".txt")
    lines = _read_txt(file)
    examples=_create_examples(lines)
    nlp = spacy.load("en_core_web_sm")
    ls = list()
    types = ['PERSON','GPE','NORP','ORG']
    for example in tqdm(examples):

        question = example.question
        contexts = example.contexts[0]
        endings_a = example.endings[0]
        endings_b = example.endings[1]
        endings_c = example.endings[2]
        endings_d = example.endings[3]

        corpus =  question + '@@' + contexts +'@@' + endings_a + '@@' + endings_b + '@@' + endings_c + '@@'+ endings_d
        sum=0
        for type in types:
            tok = nlp(corpus)
            set_temp,dict_temp = findKeyword(tok,type)
            for i, key in enumerate(set_temp):
                # print('i',i)
                tok = nlp(corpus)
                # print('key',key)
                # print('dict_temp[key]_hziq', dict_temp[key])
                dict_temp = updataKeyword(tok, type, set_temp, key,dict_temp)
                corpus,flag = replace(dict_temp,type,corpus,key,i)
                sum+=flag

            # print('set_temp','key',set_temp)

            if type== 'PERSON':
                if len(set_temp) != 0:
                    # print(len(set_temp))
                    if len(PER_list_set)<20:
                        PER_list_set.append(list(set_temp)[0])
                    else:
                        k = randint(0, 19)
                        PER_list_set[k] = list(set_temp)[0]
            elif type == 'GPE':
                if len(set_temp) != 0:
                    if len(GPE_list_set) < 20:
                        GPE_list_set.append(list(set_temp)[0])
                    else:
                        k = randint(0, 19)
                        GPE_list_set[k] = list(set_temp)[0]
            elif type == 'NORP':
                if len(set_temp) != 0:
                    # print(len(NORP_list_set))
                    if len(NORP_list_set) < 20:
                        NORP_list_set.append(list(set_temp)[0])
                    else:
                        k = randint(0, 19)
                        NORP_list_set[k] = list(set_temp)[0]
            elif type == 'ORG':
                if len(set_temp) != 0:
                    if len(ORG_list_set) < 20:
                        ORG_list_set.append(list(set_temp)[0])
                    else:
                        k = randint(0, 19)
                        ORG_list_set[k] = list(set_temp)[0]




        corpus_all = corpus.split('@@')
        dic = dict()
        if sum>=2:
            dic['context'] = corpus_all[1]
            dic["question"] = corpus_all[0]
            dic["answers"] = [corpus_all[2],corpus_all[3],corpus_all[4],corpus_all[5]]
            dic["label"] = example.label
            dic["id_string"] = 'train_'+str(example.example_id)
            ls.append(dic)
    f = open(' content.txt', 'w', encoding='utf-8')  # 第一个参数是路径，第二个参数‘w’代表写入的意思
    for i in ls:  # content into txt
        # print(i["label"])
        f.writelines('\n')
        f.writelines(i["label"]+'\n')
        f.writelines(i["context"]+'\n')
        f.writelines(i["question"]+'\n')
        f.writelines(i["answers"][0]+'\n')
        f.writelines(i["answers"][1]+'\n')
        f.writelines(i["answers"][2]+'\n')
        f.writelines(i["answers"][3]+'\n')

if __name__ == "__main__":
    main()