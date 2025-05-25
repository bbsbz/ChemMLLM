import importlib
import yaml
import os
import sys
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from taming.data.utils import custom_collate

def load_config_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
    yaml_path = "configs/mol_vqgan.yaml" 
    config = load_config_from_yaml(yaml_path)
    
    config = OmegaConf.create(config)
    
    model = instantiate_from_config(config['model'])
    bs, base_lr = config['data']['params']['batch_size'], config['model']['base_learning_rate']
    num_gpu = config['train']['num_gpu']
    model.learning_rate = num_gpu * bs * base_lr
    data = instantiate_from_config(config['data'])
    data.prepare_data()
    data.setup()
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['train']['save_dir'],
        save_weights_only=config['train']['save_weights_only'],
        monitor=config['train']['monitor'], 
        every_n_train_steps=config['train']['every_n_train_steps'], 
        filename=config['train']['filename'],         
        mode=config['train']['mode'],
        save_top_k=config['train']['save_top_k'],
    )
    
    trainer = Trainer(
        strategy="ddp_find_unused_parameters_true",
        devices=num_gpu,
        num_nodes=config['train']['num_nodes'],
        max_epochs=config['train']['max_epochs'],
        val_check_interval=config['train']['val_check_interval'],
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, data)