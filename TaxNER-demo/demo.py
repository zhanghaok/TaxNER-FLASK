# import sys
# print(sys.path)
import torch
import re
from transformers import BertTokenizer, BertConfig
from bert import BERTforNER_CRF
import data_loader
import warnings
import os
here = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore")


# from main import tab,idx2label#写对应的标签集


def split_str(s):
    return [ch for ch in s]


def extract(chars, tags):
    """
    chars：一句话 "CLS  张    三   是我们  班    主   任   SEP"
    tags：标签列表[O   B-LOC,I-LOC,O,O,O,B-PER,I-PER,i-PER,O]
    返回一段话中的实体
    """
    result = []
    pre = ''
    w = []
    for idx, tag in enumerate(tags):
        if not pre:
            if tag.startswith('B'):
                pre = tag.split('-')[1]  # pre LOC
                w.append(chars[idx])  # w 张
        else:
            if tag == f'I-{pre}':  # I-LOC True
                w.append(chars[idx])  # w 张三
            else:
                result.append([w, pre])
                w = []
                pre = ''
                if tag.startswith('B'):
                    pre = tag.split('-')[1]
                    w.append(chars[idx])
    return [[''.join(x[0]), x[1]] for x in result]


def NER_predict(model, tokenizer, text):
    max_len = 128
    scheme = 0
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

    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    model.to(device)
    model.eval()

    text = re.sub('\s', '', text)
    X = [split_str(text)]

    input_ids, attention_mask, pred_mask = data_loader.sequence_padding_for_demo(X, tokenizer, max_len)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pred_mask = pred_mask.to(device)
    res = model(input_ids, attention_mask, pred_mask=pred_mask)
    res = res[0].tolist()

    pred_labels = [idx2label[ix] for ix in res[0]]

    pred_entities = extract(text, pred_labels)

    pred_labels_ = pred_labels[:len(text)]

    return pred_entities, pred_labels_


if __name__ == '__main__':
    TOKEN_PATH = os.path.join(here, 'pretrained_model/bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained(TOKEN_PATH)
    config = BertConfig.from_pretrained(
        TOKEN_PATH,
        num_labels=33,#标签数量总共33个
        hidden_dropout_prob=0.2)
    model = BERTforNER_CRF.from_pretrained(TOKEN_PATH,
                                           config=config,
                                           use_crf=True)

    saved_model_path = os.path.join(here, "save/save_model_0/bert_crf.pt")
    model.load_state_dict(torch.load(saved_model_path, map_location="cpu"))

    text = "自2020年10月1日至2023年12月31日，对注册在广州市的保险企业向注册在南沙自贸片区的企业提供国际航运保险业务取得的收入，免征增值税。"
    res, pred_labels_ = NER_predict(model,tokenizer,text)
    print(text)
    print(res)
    print(pred_labels_)