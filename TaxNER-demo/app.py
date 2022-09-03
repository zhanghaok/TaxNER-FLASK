#! -*- coding:utf-8 -*-
from transformers import BertTokenizer, BertConfig
from bert import BERTforNER_CRF
import warnings
import torch
import os

warnings.filterwarnings("ignore")
from flask import Flask, render_template, request
from demo import NER_predict

# *****************模型准备*******************#
scheme = 0
if scheme == 1:
    tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action']
if scheme == 2:
    tab = ['Industry', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount', 'KWordAmount', 'Buyer', 'TaxRate']
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
here = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(here, 'pretrained_model/bert-base-chinese')

tokenizer = BertTokenizer.from_pretrained(TOKEN_PATH)
config = BertConfig.from_pretrained(
    TOKEN_PATH,
    num_labels=num_labels,
    hidden_dropout_prob=0.2)
model = BERTforNER_CRF.from_pretrained(TOKEN_PATH,
                                       config=config,
                                       use_crf=True)

saved_model_path = os.path.join(here, "save/save_model_0/bert_crf.pt")
model.load_state_dict(torch.load(saved_model_path, map_location="cpu"))

app = Flask(__name__)


@app.route('/', methods=['GET', ])
def index():
    return render_template('index.html')


@app.route('/nerapi', methods=['POST', ])
def nerapi():
    if request.method == 'GET':
        return "<h1>error</h1>"
    else:
        text = request.form.get('sentences')
        result, pred_labels_ = NER_predict(model, tokenizer, text)
        print(result)
        # result[0]===>[B-PER,I-PER,O,O,O]
    return render_template('index.html', result=zip(pred_labels_, list(text)))  # TODO:这个地方要改一下就欧克了


if __name__ == '__main__':
    #     app.debug = True
    app.run()  #
