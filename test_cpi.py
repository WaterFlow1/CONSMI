# coding = utf-8
import numpy as np

import torch
import torch.nn as nn
from model.unified_transformer_classifier import Transformer
import argparse
from train_cpi import process

dataset_path = 'dataset/human_cpi.txt'

parser = argparse.ArgumentParser(description='Text embedding')
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-batch-size', type=int, default=16)
parser.add_argument('-epoch', type=int, default=100)
parser.add_argument('-embedding-dim', type=int, default=128)
parser.add_argument('-dropout', type=float, default=0.4)
parser.add_argument('-device', type=str, default='cuda:1')
parser.add_argument('-vocab', type=bool, default=False)
parser.add_argument('-test-interval', type=int, default=1)
parser.add_argument('-early-stopping', type=int, default=20)
parser.add_argument('-save_best', type=bool, default=True)
parser.add_argument('-save_dir', type=str, default='./results/')


def test(args):
    print(args)
    _, _, test_loader, s2t, p2t, max_pro_len, max_smiles_len = process(args)

    model = Transformer(len(s2t), 2, len(p2t), 16, 40)
    model.load_state_dict(torch.load('./results/best_cpi_checkpoint.pt'))
    model.eval()

    with torch.no_grad():
        test_acc = []
        test_recal = []
        tp = fn = 0
        for test_smiles, test_protein, test_target in test_loader:
            smiles, protein, target = test_smiles.to(args.device), test_protein.to(args.device), test_target.to(
                args.device)
            test_logits, _ = model(smiles, protein)
            softmax = nn.Softmax()
            test_output = (softmax(test_logits) > 0.5).float().to(args.device)
            target = expand_binary_output(target, 2).to(args.device)
            tp += ((test_output == 1) & (target == 1)).sum().item()

            fn += ((test_output == 0) & (target == 1)).sum().item()

            test_recal.append(tp / (tp + fn))
            test_acc.append((test_output == target).float().mean().detach().cpu().numpy())

        print('acc:', np.mean(test_acc))
        print('recal', np.mean(test_recal))


def expand_binary_output(binary_output, target):
    expanded_output = torch.zeros((binary_output.shape[0], target))
    expanded_output[:, 0] = 1 - binary_output.squeeze()
    expanded_output[:, 1] = binary_output.squeeze()
    return expanded_output


args = parser.parse_args()
test(args)
