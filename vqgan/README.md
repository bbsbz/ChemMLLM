# Mol-VQGAN

## Getting Started
```bash
cd vqgan
```

## Data Generation
Due to the large size of the image data, we provide the smiles in text format along with a script to generate the images. The script requires RDKit in your Conda environment. Please install it first.

```bash
conda activate <your_env>
conda install -c conda-forge rdkit
```
Then run the following code to reconstruct the complete dataset:
```bash
cd data
python data.py
```
After running the script, a new image folder and smiles folder will be generated with the following structure, containing training set, validation set and test set, and corresponding images.
```bash
image/
├── test/
├── train/
└── val/
smiles/
├── test.txt
├── train.txt
└── val.txt
```

## Training

Environment named `molvqgan` can be created and activated by:

```
conda env create -f environment.yaml
conda activate molvqgan
```

And then run the following code for training:

1. Create 2 text files that point to the image files
```bash
cd ..
find data/image/train -name "*.png" > data/train.txt
find data/image/val -name "*.png" > data/val.txt
```
2. adapt configs/mol_vqgan.yaml to point to these 2 files

3. run the following code
```bash
python main.py
```

## Implementation of CORE
This repo is adapted from [Taming Transformers for High-Resolution Image Synthesis](https://github.com/CompVis/taming-transformers).
