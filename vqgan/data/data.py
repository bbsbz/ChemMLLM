import random
from tqdm import tqdm
import json
import os
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

def split_dataset(input_file, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, shuffle=True, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "The sum of the proportions must equal 1"

    with open(input_file, 'r') as f:
        file_paths = [line.strip() for line in f.readlines() if line.strip()]

    if shuffle:
        random.seed(seed)
        random.shuffle(file_paths)
        
    total = len(file_paths)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_files = file_paths[:train_size]
    val_files = file_paths[train_size : train_size + val_size]
    test_files = file_paths[train_size + val_size:]

    os.makedirs('smiles',exist_ok=True)
    with open('smiles/train.txt', 'w') as f:
        f.write('\n'.join(train_files))
    with open('smiles/val.txt', 'w') as f:
        f.write('\n'.join(val_files))
    with open('smiles/test.txt', 'w') as f:
        f.write('\n'.join(test_files))

    print(f"done：\n"
          f"- train: {len(train_files)},（{100 * train_ratio:.0f}%）\n"
          f"- val: {len(val_files)},（{100 * val_ratio:.0f}%）\n"
          f"- test: {len(test_files)},（{100 * test_ratio:.0f}%）")

def data_gen():
    files = ['smiles/train.txt','smiles/val.txt','smiles/test.txt']
    for file in files:
        with open(file,'r') as f:
            data = [line.strip() for line in f.readlines() if line.strip()]
        _, file_name = os.path.split(file)
        file_name = file_name.split('.')[0]
        os.makedirs(os.path.join('image',file_name),exist_ok=True)
        cnt = 1
        for smiles in tqdm(data):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                img = Draw.MolToImage(molecule, size=(256, 256))
                img_path = os.path.join('image',file_name,f'{cnt}.png')
                img.save(img_path)
                cnt +=1
        

if __name__ =='__main__':
    input_file = 'smiles.txt'
    split_dataset(input_file)
    data_gen()