# coding = utf-8
import numpy as np

import torch
import torch.nn.functional as F
from model.unified_transformer_classifier import Transformer
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import argparse
import re

dataset_path = 'dataset/human_cpi.txt'

parser = argparse.ArgumentParser(description='Text embedding')
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-batch-size', type=int, default=16)
parser.add_argument('-epoch', type=int, default=100)
parser.add_argument('-embedding-dim', type=int, default=128)
parser.add_argument('-dropout', type=float, default=0.4)
parser.add_argument('-device', type=str, default='cuda:1')
parser.add_argument('-vocab', type=bool, default=False)
parser.add_argument('-early-stopping', type=int, default=10)
parser.add_argument('-save_best', type=bool, default=True)
parser.add_argument('-save_dir', type=str, default='./results/')


class MyDataSet(Data.Dataset):
    def __init__(self, smiles_protein, target, s2t, p2t, smiles_max_length, protein_max_length):
        super(MyDataSet, self).__init__()
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(pattern)
        self.smiles_protein = smiles_protein
        self.target = target
        self.s2t = s2t
        self.p2t = p2t
        self.smiles_max_len = smiles_max_length
        self.protein_max_len = protein_max_length

    def __len__(self):
        return len(self.smiles_protein)

    def __getitem__(self, idx):
        smiles = self.smiles_protein[idx][0]
        smiles = self.regex.findall(smiles)
        smiles += str('<') * (self.smiles_max_len - len(smiles))
        s_tensor = torch.LongTensor([self.s2t.get(s, self.s2t['unk']) for s in smiles])

        protein = self.smiles_protein[idx][1]
        if len(protein) > self.protein_max_len:
            protein = protein[:self.protein_max_len]
        else:
            protein += str('<') * (self.protein_max_len - len(protein))
        p_tensor = torch.LongTensor([self.p2t[p] for p in protein])

        target = torch.LongTensor([self.target[idx]])
        return s_tensor, p_tensor, target


def process(args):
    print('Reading lines...')
    lines = open(dataset_path, encoding='utf-8').read().strip().split('\n')
    smiles_protein, label = [], []

    for l in lines:
        smiles_protein.append((l.split(' ')[0], l.split(' ')[1]))
        label.append(int(l.split(' ')[2]))

    max_smiles_length = max([len(s[0]) for s in smiles_protein])
    max_protein_length = max([len(s[1]) for s in smiles_protein])
    max_protein_length = 1000
    s2t, p2t = {}, {}
    with open('config/vocab_copy.txt', 'r+') as f:
        lines = f.readlines()
        for line in lines:
            s2t[line.split('\t')[0]] = int(line.split('\t')[1])

    with open('config/protein_vocab.txt', 'r+') as f:
        lines = f.readlines()
        for line in lines:
            p2t[line.split('\t')[0]] = int(line.split('\t')[1])

    # idx = 0
    # for p in smiles_protein:
    #     for i in p[1]:
    #         if not p2t.get(i):
    #             p2t[i] = idx
    #             idx += 1
    print(s2t)
    print(p2t)

    x_train, x_test, y_train, y_test = train_test_split(smiles_protein, label, test_size=0.10,
                                                        random_state=1)

    x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.10, random_state=1)

    train_loader = Data.DataLoader(MyDataSet(x_train, y_train, s2t, p2t, max_smiles_length, max_protein_length),
                                   args.batch_size, True)
    dev_loader = Data.DataLoader(MyDataSet(x_dev, y_dev, s2t, p2t, max_smiles_length, max_protein_length),
                                 args.batch_size, True)
    test_loader = Data.DataLoader(MyDataSet(x_test, y_test, s2t, p2t, max_smiles_length, max_protein_length),
                                  args.batch_size, True)
    return train_loader, dev_loader, test_loader, s2t, p2t, max_protein_length, max_smiles_length


def train(args):
    print(args)
    train_loader, dev_loader, test_loader, s2t, p2t, max_pro_len, max_smiles_len = process(args)
    model = Transformer(len(s2t), 2, len(p2t), 16, 40)
    if 'cuda:1' in args.device: model.cuda(1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    best_loss = 10
    last_epoch = 0

    # pre_model = TextEmbedding(src_vocab_size=95)
    # pre_model.load_state_dict(torch.load('./results/best_steps_checkpoint20230326.pt'))
    # model.mol_encoder.src_emb.load_state_dict(pre_model.tok_emb.state_dict())
    # model.mol_encoder.src_emb.requires_grad_(False)

    # model.mol_encoder.pos_emb = pre_model.pos_emb
    # model.type_emb = pre_model.type_emb
    # model.blocks = pre_model.blocks

    model.train()
    for epoch in range(1, args.epoch + 1):
        sum_loss = []
        for smiles, protein, target in train_loader:
            smiles, protein, target = smiles.to(args.device), protein.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            logits, _ = model(smiles, protein)
            loss = F.cross_entropy(logits, target.view(-1))
            loss.backward()
            optimizer.step()
            sum_loss.append(loss.detach().cpu().numpy())
        print('epoch_{}_train_loss:{}'.format(epoch, np.average(sum_loss)))

        dev_loss = []
        with torch.no_grad():
            for dev_smiles, dev_protein, dev_target in dev_loader:
                smiles, protein, target = dev_smiles.to(args.device), dev_protein.to(args.device), dev_target.to(
                    args.device)
                dev_logits, _ = model(smiles, protein)
                loss = F.cross_entropy(dev_logits, target.view(-1))
                dev_loss.append(loss.detach().cpu().numpy())
            avg_loss = np.average(dev_loss)
        print('epoch_{}_dev_loss:{}'.format(epoch, avg_loss))
        if avg_loss - best_loss < -0.0001:
            best_loss = avg_loss
            last_epoch = epoch
            if args.save_best:
                print('Saving best model, loss: {:.4f}\n'.format(best_loss))
                save(model, args.save_dir, 'best', epoch)
        else:
            if epoch - last_epoch >= args.early_stopping:
                print('\nearly stop by {} steps, loss: {:.4f}'.format(args.early_stopping, best_loss))
                raise KeyboardInterrupt


import os
import datetime


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_cpi_checkpoint.pt'.format(save_prefix)
    print(save_path)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    torch.save(model.state_dict(), save_path)


args = parser.parse_args()
train(args)
