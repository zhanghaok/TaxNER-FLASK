from re import L
from framework import Framework, set_seed
from data_loader import NERDataset
from model import BERTforNER_CRF, BiLSTM_CRF
from transformers import BertConfig, BertTokenizer
import argparse
import torch
import json
import os


def main(args):

    # tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action']
    # tab = ['Industry', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount', 'KWordAmount', 'Buyer', 'TaxRate']
    # tab = ['PayerDecorate', 'ObjDecorate', 'TaxDecorate', 'ActionDecorate']
    tab = ['TaxPayer', 'Taxobj', 'Tax', 'Action', 'Loc', 'StartTime', 'EndTime', 'UpperAmount', 'LowerAmount', 'KWordAmount', 'Buyer', 'TaxRate', 'PayerDecorate', 'ObjDecorate', 'TaxDecorate', 'ActionDecorate']
    labels = ['O']
    for l in tab:
        for seg in ['B', 'I']:
            token = seg + '-' + l
            labels.append(token)

    args.num_labels = len(labels)

    tokenizer = None
    word2id = None
    print("选择的模型是：",args.model)
    if args.model == 'bert':
        is_BERT = True
        # use 'bert-base-chinese' model
        pretrained_model_name = 'bert-base-chinese'
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        config = BertConfig.from_pretrained(
            pretrained_model_name,
            num_labels=args.num_labels,
            hidden_dropout_prob=args.hidden_dropout_prob)
        model = BERTforNER_CRF.from_pretrained(pretrained_model_name,
                                               config=config,
                                               use_crf=args.crf)
    else:
        is_BERT = False
        print("使用BiLSTM")
        word2id = json.load(open(args.word2id_file, "r", encoding="utf8"))
        model = BiLSTM_CRF(len(word2id), args.embedding_dim, args.hidden_dim,
                           args.num_labels, args.hidden_dropout_prob, args.crf)

    framework = Framework(args)

    if args.mode == "train":
        print("loading training dataset...")
        train_dataset = NERDataset(file_path=args.train_file,
                                   labels=labels,
                                   word2id=word2id,
                                   tokenizer=tokenizer,
                                   max_len=args.max_len,
                                   is_BERT=is_BERT)

        print("loading dev datasets...")
        dev_dataset = NERDataset(file_path=args.dev_file,
                                 labels=labels,
                                 word2id=word2id,
                                 tokenizer=tokenizer,
                                 max_len=args.max_len,
                                 is_BERT=is_BERT)

        framework.train(train_dataset, dev_dataset, model, labels)

    print("\Testing ...")
    print("loading testing datasets...")
    test_dataset = NERDataset(file_path=args.test_file,
                              labels=labels,
                              word2id=word2id,
                              tokenizer=tokenizer,
                              max_len=args.max_len,
                              is_BERT=is_BERT)

    model.load_state_dict(torch.load(args.save_model))
    framework.test(test_dataset, model, labels)


if __name__ == "__main__":

    set_seed(2020)

    parser = argparse.ArgumentParser()

    # task setting
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'test'])
    parser.add_argument('--model',
                        type=str,
                        default='bert',
                        # default='bilstm',
                        choices=['bilstm', 'bert'])
    parser.add_argument('--crf', action='store_true')

    # train setting
    parser.add_argument('--evaluate_step', type=int, default=1000)
    parser.add_argument('--max_len', type=int, default=256)

    parser.add_argument('--train_batch_size', type=int, default=12)
    parser.add_argument('--dev_batch_size', type=int, default=6)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--crf_lr', type=float, default=1e-2)
    # parser.add_argument('--train_batch_size', type=int, default=64)
    # parser.add_argument('--dev_batch_size', type=int, default=32)
    # parser.add_argument('--num_train_epochs', type=int, default=20)
    # parser.add_argument('--learning_rate', type=float, default=1e-3)

    # for BiLSTM
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=200)

    # file path
    parser.add_argument('--train_file', type=str, default='./data/train.txt')
    parser.add_argument('--dev_file', type=str, default='./data/dev.txt')
    parser.add_argument('--test_file', type=str, default='./data/test.txt')
    parser.add_argument('--word2id_file',
                        type=str,
                        default='./data/word2id.json')

    parser.add_argument('--save_model', type=str, default='./save_model/')
    parser.add_argument('--output_dir', type=str, default='./output/')

    # others
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.2)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    args = parser.parse_args()

    # device
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for file_dir in [args.save_model, args.output_dir]:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

    if args.crf:
        save_name = args.model + "_crf"
    else:
        save_name = args.model

    args.save_model = os.path.join(args.save_model, save_name + ".pt")
    args.output_dir = os.path.join(args.output_dir, save_name + ".result")

    print(args)
    main(args)

# nohup python main.py --model bert --crf --train_file ./data/train_0.txt --dev_file ./data/dev_0.txt --test_file ./data/test_0.txt --num_train_epochs 400 --evaluate_step 500 --save_model ./save_model_0/ --output_dir ./output_0/ 1>L0_400epoch.out 2>&1 &
# nohup python main.py --model bert --crf --train_file ./data/train_1.txt --dev_file ./data/dev_1.txt --test_file ./data/test_1.txt --num_train_epochs 400 --evaluate_step 500 --save_model ./save_model_1/ --output_dir ./output_1/ 1>L1_400epoch.out 2>&1 &