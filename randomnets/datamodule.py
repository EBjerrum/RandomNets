import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning
from rdkit.Chem import PandasTools

from scikit_mol.fingerprints import MorganFingerprintTransformer
from sklearn.model_selection import train_test_split


class FpsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fp1 = torch.tensor(self.data.fps.iloc[idx], dtype=torch.float)
        y1 = torch.tensor(self.data.pXC50.iloc[idx], dtype=torch.float)
        sample_mask = torch.tensor(self.data.sample_mask.iloc[idx], dtype=torch.float)

        return fp1, y1, sample_mask


class FpsDatamodule(pytorch_lightning.LightningDataModule):
    def __init__(self, batch_size, n_nns=25, sample_mask_thr=0.5):
        super().__init__()
        self.n_nns = n_nns
        self.sample_mask_thr = sample_mask_thr
        self.batch_size = batch_size
        self.csv_file = "SLC6A4_active_excape_export.csv"

    def ensure_data(self):
        if not os.path.exists(self.csv_file):
            import urllib.request

            url = "https://ndownloader.figshare.com/files/25747817"
            urllib.request.urlretrieve(url, self.csv_file)

    def setup(self, stage):
        if hasattr(self, "data"):
            return
        self.ensure_data()
        self.data = pd.read_csv(self.csv_file)

        PandasTools.AddMoleculeColumnToFrame(self.data, smilesCol="SMILES")
        print(
            f"{self.data.ROMol.isna().sum()} out of {len(self.data)} SMILES failed in conversion"
        )

        trf = MorganFingerprintTransformer(nBits=4096, radius=2)
        fps = trf.transform(self.data.ROMol)
        self.data["fps"] = [arr for arr in fps]

        # prepare model_mask
        self.data["sample_mask"] = [
            np.random.random(self.n_nns) > self.sample_mask_thr
            for i in range(len(self.data))
        ]

        # Initialize Weights
        self.data_train, self.data_test = train_test_split(self.data, random_state=0)

        # self.data_train, self.data_val = train_test_split(data_train, random_state=0)
        # With mask reversal, we can get the cross_val score
        self.data_val = self.data_train.sample(1000, random_state=0)

    def train_dataloader(self, shuffle=True):
        dataset = FpsDataset(self.data_train)
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=16, shuffle=shuffle
        )

    def val_dataloader(self):
        dataset = FpsDataset(self.data_val)
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=16, shuffle=False
        )

    def test_dataloader(self):
        dataset = FpsDataset(self.data_test)
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=16, shuffle=False
        )
