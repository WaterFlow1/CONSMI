# coding = utf-8
import numpy as np
import os
import datetime
from utils.util import augment
import torch
import torch.nn as nn
from model.contrastive_model import TextEmbedding, NTXentLoss
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import argparse
from tqdm import tqdm
import re
import random

dataset_path = '../dataset/zinc0220.csv'
parser = argparse.ArgumentParser(description='Text embedding')
parser.add_argument('-max-length', type=str, default=100)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-batch-size', type=int, default=168)
parser.add_argument('-epoch', type=int, default=300)
parser.add_argument('-dropout', type=float, default=0.4)
parser.add_argument('-device', type=str, default='cuda:3')
parser.add_argument('-vocab', type=bool, default=False)
parser.add_argument('-early-stopping', type=int, default=10)
parser.add_argument('-save_best', type=bool, default=True)
parser.add_argument('-save_dir', type=str, default='./results/')

dataset_index = 0


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)


class Seq:
    def __init__(self, name):
        self.name = name
        self.word2index = {'<': 0, 'SOS': 1, 'EOS': 2}  # vocabulary
        self.word2count = {}
        self.index2word = {0: '<', 1: 'SOS', 2: "EOS"}
        self.n_words = 3  # PAD, SOS, EOS

    def fill_vocab(self, word2index):
        self.word2index.clear()
        self.index2word.clear()
        self.word2count.clear()
        self.n_words = 0
        for word, index in word2index.items():
            self.word2index[word] = index
            self.index2word[index] = word
            if word in self.word2count.keys():
                self.word2count[word] += 1
            else:
                self.word2count[word] = 1
            self.n_words += 1

    def add_sentence(self, sentence):
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        sentence = regex.findall(sentence)
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class MyDataSet(Data.Dataset):
    def __init__(self, smiles, src_vocab):
        super(MyDataSet, self).__init__()
        self.smiles = smiles
        self.src_vocab = src_vocab
        self.pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        global dataset_index
        smiles = self.smiles[idx]
        smiles_aug = augment(smiles)[0]
        smiles_tensor = self.SMILES2Tensor(smiles)
        smiles_aug_tensor = self.SMILES2Tensor(smiles_aug)
        target = torch.LongTensor([dataset_index])
        dataset_index += 1
        return smiles_tensor, smiles_aug_tensor, target

    def SMILES2Tensor(self, smiles):
        smiles = self.regex.findall(smiles)
        if len(smiles) > args.max_length:
            smiles = smiles[:args.max_length]
        smiles += str('<') * (args.max_length + 1 - len(smiles))
        tensor = []
        for n in smiles:
            if n in self.src_vocab:
                tensor.append(self.src_vocab[n])
            else:
                tensor.append(self.src_vocab['unk'])
        return torch.LongTensor(tensor)


def process(language, args):
    print('Reading lines...')
    lines = open(dataset_path, encoding='utf-8').read().strip().split('\n')
    smiles = []

    for l in lines[1:]:
        temp = l.split(',')[0]
        if args.max_length > len(temp) > 0 and '.' not in temp:
            smiles.append(temp)
    input_l = Seq(language)

    for smile in tqdm(smiles):
        input_l.add_sentence(smile)
    print(len(smiles))
    if args.vocab:
        word2index = {}
        with open('config/vocab_copy.txt', 'r+') as f:
            lines = f.readlines()
            for line in lines:
                word2index[line.split('\t')[0]] = int(line.split('\t')[1])
        input_l.fill_vocab(word2index)
    else:
        with open('config/vocab.txt', 'w+') as f:
            for key, value in input_l.word2index.items():
                f.write(key + '\t' + str(value) + '\n')
    src_vocab_size = input_l.n_words
    src_vocab = input_l.word2index
    print(src_vocab_size)
    print(input_l.index2word)

    x_train, x_test = train_test_split(smiles, test_size=0.05, random_state=1)
    return Data.DataLoader(MyDataSet(x_train, src_vocab), args.batch_size, True, num_workers=4), Data.DataLoader(
        MyDataSet(x_test, src_vocab),
        args.batch_size,
        True, num_workers=4), src_vocab_size


def train(args):
    set_seed(3407)
    train_loader, dev_loader, args.vocab_size = process('input', args)
    model = TextEmbedding(args.vocab_size)
    if 'cuda' in args.device: model.cuda(3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = 10000
    last_epoch = 0
    model.train()
    conloss = NTXentLoss(temperature=1.0)
    for epoch in range(1, args.epoch + 1):
        tqdm_bar = tqdm(train_loader, desc="Training")
        sum_loss = []
        for smiles, smiles_aug, target in tqdm_bar:
            temp_target = target
            smiles = torch.cat([smiles, smiles_aug], 0)
            target = torch.cat([target, temp_target], 0).squeeze()
            smiles, target = smiles.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            logits = model(smiles)
            loss = conloss(logits, target)
            loss.backward()
            optimizer.step()
            sum_loss.append(loss.detach().cpu().numpy())
        print('epoch_{}_train_loss:{}'.format(epoch, np.average(sum_loss)))

        dev_loss = []
        dev_tqdm_bar = tqdm(dev_loader, desc="Training")
        with torch.no_grad():
            for dev_smiles, dev_smiles_aug, dev_target in dev_tqdm_bar:
                temp_target = dev_target
                dev_smiles = torch.cat([dev_smiles, dev_smiles_aug], 0)
                dev_target = torch.cat([dev_target, temp_target], 0).squeeze()
                dev_smiles, dev_target = dev_smiles.to(args.device), dev_target.to(args.device)
                dev_logits = model(dev_smiles)
                loss = conloss(dev_logits, dev_target)
                dev_loss.append(loss.detach().cpu().numpy())
            avg_loss = np.average(dev_loss)
        print('epoch_{}_dev_loss:{}'.format(epoch, avg_loss))
        if avg_loss - best_loss < -0.0001:
            best_loss = avg_loss
            last_epoch = epoch
            if args.save_best:
                print('Saving best model, loss: {:.4f}\n'.format(best_loss))
                save(model, args.save_dir, args.device, epoch)
        else:
            if epoch - last_epoch >= args.early_stopping:
                print('\nearly stop by {} steps, loss: {:.4f}'.format(args.early_stopping, best_loss))
                raise KeyboardInterrupt


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_checkpoint.pt'.format(save_prefix)
    print(save_path)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    torch.save(model.state_dict(), save_path)


args = parser.parse_args()
train(args)
