import json
import os
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

def get_all_folders(path):
    folders = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folders.append(entry.path)
    return folders

def get_all_files(path):
    files = []
    for entry in os.scandir(path):
        if entry.is_file():
            files.append(entry.path)
    return files

def count_files(path):
    total_files = 0
    for root, dirs, files in os.walk(path):
        total_files += len(files)
    return total_files

def count_smiles(raw_data):
    raw = raw_data[0]
    smiles = raw["SMILES"]
    if isinstance(smiles, list):
        return len(smiles) * len(raw_data)
    else: 
        return len(raw_data)

                    
def data_gen(raw_path):
    foldirs = get_all_folders(raw_path)
    for dirs in foldirs:
        _, spl = os.path.split(dirs)
        os.makedirs(os.path.join('data',spl),exist_ok=True)
        os.makedirs(os.path.join('data',spl,'image'),exist_ok=True)
        files = get_all_files(dirs)
        for file in files:
            cnt = 1
            _, file_name = os.path.split(file)
            file_name = file_name.split('.')[0]
            os.makedirs(os.path.join('data',spl,'image',file_name),exist_ok=True)
            with open(file,'r') as f:
                raw_data = json.load(f)
            print(file,"generating...")
            if count_files(os.path.join('data',spl,'image',file_name)) == count_smiles(raw_data):
                print(file,"generated")  
                continue
            new_data = []
            for raw in tqdm(raw_data):
                smiles = raw["SMILES"]
                image = []
                if isinstance(smiles, list):
                    for smi in smiles:
                        molecule = Chem.MolFromSmiles(smi)
                        if molecule is not None:
                            img = Draw.MolToImage(molecule, size=(256, 256))
                            img_path = os.path.join('data',spl,'image',file_name,f'{cnt}.png')
                            img.save(img_path)
                            image.append(img_path)
                            cnt +=1
                else:
                    molecule = Chem.MolFromSmiles(smiles)
                    if molecule is not None:
                        img = Draw.MolToImage(molecule, size=(256, 256))
                        img_path = os.path.join('data',spl,'image',file_name,f'{cnt}.png')
                        img.save(img_path)
                        image.append(img_path)
                        cnt +=1
                raw["image"] = image
                new_data.append(raw)
            with open(os.path.join('data',spl,f'{file_name}.json'),'w') as w:
                json.dump(new_data,w,indent=4)  
            print(file,"generated")  
    print('all done')

if __name__ == '__main__':
    path = 'raw_text'
    data_gen(path)