# coding = utf-8
import re
import unicodedata
from rdkit import Chem


def augment(smile):
    mols = set()
    mol = Chem.MolFromSmiles(smile)
    while len(mols) < 1:
        mols.add(Chem.MolToSmiles(mol, doRandom=True))
    return list(mols)
