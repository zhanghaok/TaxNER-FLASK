import os
import re
import torch
import json
from transformers import BertTokenizer,BertConfig
from model import BERTforNER_CRF
from data_loader import sequence_padding_for_demo
import warnings
warnings.filterwarnings("ignore")
here = os.path.dirname(os.path.abspath(__file__))
# from main import tab,idx2label#写对应的标签集

TOKEN_PATH = os.path.join(here,'pretrained_model/bert-base-chinese')
# TOKEN_PATH = 'bert-base-chinese'
max_len = 128

def split_str(s):
    return [ch for ch in s]

def extract(chars,tags):
    """
    chars：一句话 "CLS  张    三   是我们  班    主   任   SEP"
    tags：标签列表[O   B-LOC,I-LOC,O,O,O,B-PER,I-PER,i-PER,O]
    返回一段话中的实体
    """
    ne_set = set()

    result = []
    pre = ''
    w = []
    position=[]
    for idx,tag in enumerate(tags):
        if not pre:
            if tag.startswith('B'):
                pre = tag.split('-')[1] #pre LOC
                w.append(chars[idx])#w 张
                position.append(idx)
                ne_set.add(pre)
        else:
            if tag == f'I-{pre}': #I-LOC True
                w.append(chars[idx]) #w 张三
                position.append(idx)
            else:
                result.append([w,pre,(position[0],position[-1])])
                position = []
                w = []
                pre = ''
                if tag.startswith('B'):
                    pre = tag.split('-')[1]
                    w.append(chars[idx])
                    position.append(idx)
                    ne_set.add(pre)
    # print(result)
    return [[''.join(x[0]),x[1],x[-1]] for x in result], ne_set

def extractBatch(str_list,tags_list):
    """
    chars：一句话 "CLS  张    三   是我们  班    主   任   SEP"
    tags：标签列表[O   B-LOC,I-LOC,O,O,O,B-PER,I-PER,i-PER,O]
    返回一段话中的实体
    """
    result = []
    ne_sets = []
    pre = ''
    w = []
    for str,tag in zip(str_list,tags_list):
        pred_entity,ne_set = extract(str,tag)
        result.append(pred_entity)
        ne_sets.append(ne_set)
    return result,ne_sets



def testModel():
    scheme = input("请输入标注模式，总共有4种,请你输入0，1，2或者3\n")
    scheme = int(scheme)
    if scheme == 1:
        tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action']
    if scheme == 2:
        tab = ['Industry', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount', 'KWordAmount', 'Buyer',
               'TaxRate']
    if scheme == 3:
        tab = ['PayerDecorate', 'ObjDecorate', 'TaxDecorate', 'ActionDecorate']
    if scheme == 0:
        tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount',
               'KWordAmount', 'Buyer', 'TaxRate', 'PayerDecorate', 'ObjDecorate', 'TaxDecorate', 'ActionDecorate']

    labels = ['O']
    for l in tab:
        for seg in ['B', 'I']:
            token = seg + '-' + l
            labels.append(token)
    idx2label = labels
    num_labels = len(labels)

    tokenizer = BertTokenizer.from_pretrained(TOKEN_PATH)
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    config = BertConfig.from_pretrained(
        TOKEN_PATH,
        num_labels=num_labels,
        hidden_dropout_prob=0.2)
    model = BERTforNER_CRF.from_pretrained(TOKEN_PATH,
                                           config=config,
                                           use_crf=True)
    # scheme = 0 #标注模式
    if scheme == 0:
        model.load_state_dict(torch.load("./save_model_0/bert_crf.pt", map_location="cpu"))
    if scheme == 1:
        model.load_state_dict(torch.load("./save_model_1/bert_crf.pt", map_location="cpu"))
    if scheme == 2:
        model.load_state_dict(torch.load("./save_model_2/bert_crf.pt", map_location="cpu"))
    if scheme == 1:
        model.load_state_dict(torch.load("./save_model_3/bert_crf.pt", map_location="cpu"))
    model.to(device)
    model.eval()

    os.chdir('./sentences')
    for ftxt in os.listdir():
        if not os.path.isfile(ftxt) or ftxt[-4:] != '.txt':
            continue

        text = ''
        with open(ftxt, 'r', encoding='utf-8') as fin:
            for l in fin.readlines():
                text += l.strip()

        # text = input("请输入：")
        text = re.sub('\s', '', text)
        X = [split_str(text)]

        try:
            input_ids,attention_mask,pred_mask = sequence_padding_for_demo(X,tokenizer,max_len)
        except Exception:
            # with open('/home/cjr/Chinese-NER/error.list', 'a', encoding='utf-8') as fout:
            #     fout.write(ftxt+'\n')
            print(ftxt)
            continue


        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pred_mask = pred_mask.to(device)
        res = model(input_ids,attention_mask,pred_mask = pred_mask)
        res = res[0].tolist()

        pred_labels = [idx2label[ix] for ix in res[0]]

        chars = tokenizer.convert_ids_to_tokens(input_ids[0])
        # print([f'{w}_{s}' for w, s in zip(text, pred_labels)])
        pred_entities, ne_set = extract(text,pred_labels)
        # print(pred_entities)
        # print(ne_set)
        if len(ne_set) == 3:
            with open('/home/cjr/Chinese-NER/equal3.txt', 'a', encoding='utf-8') as fout:
                fout.write(ftxt + '\n')
        elif len(ne_set) > 3:
            # print(ftxt)
            # print("预测的实体：%s" % pred_entities)
            with open('/home/cjr/Chinese-NER/morethan3.txt', 'a', encoding='utf-8') as fout:
                fout.write(ftxt + '\n')
        # for w, s in zip(text, pred_labels):
        #     print(f"{w}_{s}")


def testBatch(unlabeld_path,labeled_path):

    #税友内部的映射表
    entity_to_id = {
        'TaxPayer': '1542037826984517632',
        'Taxobj': '1542037850803970048',
        'Tax': '1542037908907663360',
        'Action': '1542037885960626176',
        'Industry': '1542038051539165184',
        'Loc': '1542038083495567360',
        'StartTime': '1542037949072318464',
        'EndTime': '1542037971763503104',
        'UpperAmount': '1542038034921332736',
        'LowerAmount': '1542038016764190720',
        'KWordAmount': '1542037998355390464',
        'Buyer': '1542038066865152000',
        'TaxRate': '1542038098062385152',
        'PayerDecorate': '1542038113275125760',
        'ObjDecorate': '1542038128206848000',
        'TaxDecorate': '1542038159957729280',
        'ActionDecorate': '1542038143323119616'
    }


    entity_dict = {
        'TaxPayer':'纳税人',
        'Taxobj':'征税对象',
        'Tax':'税种',
        'Action':'动作',
        'Industry':'行业',
        'Loc':'地点',
        'StartTime':'起始时间',
        'EndTime':'终止时间',
        'UpperAmount':'金额上界',
        'LowerAmount':'金额下界',
        'KWordAmount':'金额中心词',
        'Buyer':'购买方',
        'TaxRate':'税率',
        'PayerDecorate':'纳税人修饰',
        'ObjDecorate':'对象修饰',
        'TaxDecorate':'税种修饰',
        'ActionDecorate':'行为修饰'
    }
    # scheme = input("请输入标注模式，总共有4种,请你输入0，1，2或者3\n")
    # scheme = int(scheme)
    scheme=0
    if scheme == 1:
        tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action']
    if scheme == 2:
        tab = ['Industry', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount', 'KWordAmount', 'Buyer',
               'TaxRate']
    if scheme == 3:
        tab = ['PayerDecorate', 'ObjDecorate', 'TaxDecorate', 'ActionDecorate']
    if scheme == 0:
        tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount',
               'KWordAmount', 'Buyer', 'TaxRate', 'PayerDecorate', 'ObjDecorate', 'TaxDecorate', 'ActionDecorate']

    labels = ['O']
    for l in tab:
        for seg in ['B', 'I']:
            token = seg + '-' + l
            labels.append(token)
    idx2label = labels
    num_labels = len(labels)
    tokenizer = BertTokenizer.from_pretrained(TOKEN_PATH)
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    config = BertConfig.from_pretrained(
        TOKEN_PATH,
        num_labels=num_labels,
        hidden_dropout_prob=0.2)
    model = BERTforNER_CRF.from_pretrained(TOKEN_PATH,
                                           config=config,
                                           use_crf=True)
    # scheme = 0 #默认标注模式为0
    if scheme == 0:
        model.load_state_dict(torch.load("./save_model_0/bert_crf.pt", map_location="cpu"))
    if scheme == 1:
        model.load_state_dict(torch.load("./save_model_1/bert_crf.pt", map_location="cpu"))
    if scheme == 2:
        model.load_state_dict(torch.load("./save_model_2/bert_crf.pt", map_location="cpu"))
    if scheme == 1:
        model.load_state_dict(torch.load("./save_model_3/bert_crf.pt", map_location="cpu"))
    model.to(device)
    model.eval()

    os.chdir(unlabeld_path)
    for ftxt in os.listdir():
        if not os.path.isfile(ftxt) or ftxt[-4:] != '.txt':
            continue

        text = []
        with open(ftxt, 'r', encoding='utf-8') as fin:
            for l in fin.readlines():
                text.append(l.strip())
        # text = input("请输入：")
        # 一定要要去掉存在的空字符等，不然会报错
        text = [re.sub('\s', '', string) for string in text if string !='']
        X = [split_str(_) for _ in text]
        try:
            input_ids,attention_mask,pred_mask = sequence_padding_for_demo(X,tokenizer,max_len)
        except Exception:
            print(ftxt)
            continue
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pred_mask = pred_mask.to(device)
        step=32
        for i in range(0,len(input_ids),step):#
            """
            做一个批量数据，以32为一个批量大小。
            防止文件过大，内存爆炸。
            """
            input_ids_=input_ids[i:min(i+step,len(input_ids))]
            attention_mask_=attention_mask[i:min(i+step,len(input_ids))]
            pred_mask_=pred_mask[i:min(i+step,len(input_ids))]
            text_ = text[i:min(i+step,len(input_ids))]
            res = model(input_ids_,attention_mask_,pred_mask = pred_mask_)#(pred,loss)
            res = res[0].tolist() # pred [[]m]
            pred_labels = [[idx2label[ix] for ix in j] for j in res]
            # chars = tokenizer.convert_ids_to_tokens(input_ids[0])
            pred_entities, ne_sets = extractBatch(text_,pred_labels)
            # print(f'pred_entities:{pred_entities}')
            # print(f'ne_sets:{ne_sets}')

            # for item in pred_entities:
            #     if len(item)>=3:
            #         print(f'item:{item}')
            # print(f'pred_entities:{pred_entities}')
            # print(f'ne_sets:{ne_sets}')

            for i, item in enumerate(pred_entities):#[32,]
                if len(item) >= 3:
                    """
                    挑选一些显著的预测结果，并将其回标进标注工具中（即生成BRAT格式的标注文件），方便人工修正。
                    """
                    # for idx, line in enumerate(item):
                    #     print(f'line:{line}') #line:['嵌入式软件产品和信息', 'Taxobj', (116, 125)]
                    txt_filename = f'{ftxt[:-4]}_{str(i)}.txt'
                    with open(os.path.join(labeled_path,txt_filename),'w',encoding="utf-8") as txtf:
                        txtf.write(text[i])
                        ann_filename = f'{ftxt[:-4]}_{str(i)}.ann'
                        with open(os.path.join(labeled_path,ann_filename),'w',encoding="utf-8") as annf:#讲结果写入标注文件
                            # tmp={"content":text_[i],"labels":[]}
                            for idx,line in enumerate(item):
                                """
                                BART标注文件格式：T6	税种 32 35	增值税
                                """
                                # print(f'line:{line}')
                                item = f"T{idx+1}	{entity_dict.get(line[1])} {line[2][0]} {line[2][1]+1}	{line[0]}"
                                # tmp["labels"].append({"id":idx,"categoryId":entity_to_id[line[1]],"startIndex":line[2][0],"endIndex":line[2][1]+1})
                                # f.write(f'"content::"{text[i]},"labels":{entity_dict[line[1]]} {line[2][0]} {line[2][1]+1}	{line[0]}\n')
                                annf.write(item+"\n")

if __name__ == '__main__':
    file_path = './unlabled_data' #未标注文章路径
    file_path = os.path.join(here,file_path)
    labeled_path = 'labeled_data' #存放回标结果的文件路径
    labeled_path = os.path.join(here,labeled_path)
    testBatch(file_path,labeled_path)