import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning
from rdkit.Chem import PandasTools

from scikit_mol.fingerprints import MorganFingerprintTransformer
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class FpsDataset(Dataset):
    def __init__(
        self,
        data,
        target_column="pXC50",
        feature_column="fps",
        sample_mask_column="sample_mask",
        invert_mask=False,
    ):
        self.data = data
        self.target_column = target_column
        self.feature_column = feature_column
        self.sample_mask_column = sample_mask_column
        self.invert_mask = invert_mask
        self._max_n = np.max(np.stack(self.data.sample_mask.values)) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fp1 = torch.tensor(self.data[self.feature_column].iloc[idx], dtype=torch.float)
        y1 = torch.tensor(self.data[self.target_column].iloc[idx], dtype=torch.float)

        sample_mask = self.data[self.sample_mask_column].iloc[idx]

        full_set = set(range(self._max_n))

        if self.invert_mask:
            sample_mask = list(full_set - set(sample_mask))

        sample_mask = torch.tensor(sample_mask, dtype=torch.int64) + 0

        return fp1, y1, sample_mask


class FpsDatamodule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        batch_size,
        csv_file="SLC6A4_active_excape_export.csv",
        n_nns=25,
        sample_mask_thr=0.0,
        dedicated_val=True,
        val_sample_size=0.25,
        scikit_mol_transformer=MorganFingerprintTransformer(nBits=4096, radius=2),
        smiles_column="SMILES",
        target_column="pXC50",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.n_nns = n_nns
        self.sample_mask_fraction = sample_mask_thr
        self.csv_file = csv_file
        self.dedicated_val = dedicated_val
        self.skmol_trf = scikit_mol_transformer
        self.val_sample_size = val_sample_size
        self.target_column = target_column
        self.smiles_column = smiles_column
        self.features_column = "fps"
        self.sample_mask_column = "sample_mask"
        self.num_worker = 4

        self.save_hyperparameters()

    def setup(self, stage):
        if hasattr(self, "data"):
            return
        assert os.path.exists(self.csv_file), f"CSV file {self.csv_file} not found"
        self.data = pd.read_csv(self.csv_file)

        PandasTools.AddMoleculeColumnToFrame(self.data, smilesCol=self.smiles_column)
        mol_conv_errors = self.data.ROMol.isna().sum()
        if mol_conv_errors > 0:
            logger.warning(
                f"{mol_conv_errors} out of {len(self.data)} SMILES failed in conversion"
            )
            self.data = self.data[self.data.ROMol.notna()]

        fps = self.skmol_trf.transform(self.data.ROMol)
        self.data[self.features_column] = [arr for arr in fps]

        # prepare model_mask, TODO: is there another way to prepare the mask? it preempts knowledge about the number in the ensemble, which is the models responsibility!
        # This is the indices of the ensemble ids to associate with each sample
        self.data[self.sample_mask_column] = [
            np.random.choice(
                range(self.n_nns),
                max(
                    int(self.n_nns * (1 - self.sample_mask_fraction)), 1
                ),  # Ensure always one selected
                replace=False,
            )
            for i in range(len(self.data))
        ]

        # dataset splits
        self.data_train, self.data_test = train_test_split(self.data, random_state=0)

        if self.dedicated_val:
            self.data_train, self.data_val = train_test_split(
                self.data_train, test_size=self.val_sample_size, random_state=0
            )
        else:
            # With sample mask reversal, we can get the pseudo cross_val loss
            if isinstance(self.val_sample_size, float):
                val_sample_size = int(self.val_sample_size * len(self.data_train))
            else:
                val_sample_size = self.val_sample_size
            self.data_val = self.data_train.sample(val_sample_size, random_state=0)

    def train_dataloader(self, shuffle=True):
        dataset = FpsDataset(
            self.data_train,
            feature_column=self.features_column,
            target_column=self.target_column,
            sample_mask_column=self.sample_mask_column,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            shuffle=shuffle,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.dedicated_val:
            invert_mask = False
        else:
            invert_mask = True
            # print("Inverting Mask on dataset")

        dataset = FpsDataset(
            self.data_val,
            feature_column=self.features_column,
            target_column=self.target_column,
            sample_mask_column=self.sample_mask_column,
            invert_mask=invert_mask,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        dataset = FpsDataset(
            self.data_test,
            feature_column=self.features_column,
            target_column=self.target_column,
            sample_mask_column=self.sample_mask_column,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_worker,
            shuffle=False,
            drop_last=True,
        )

    def mols_to_fp(self, mol_list):
        fps = self.skmol_trf.transform(mol_list)
        return torch.tensor(fps, dtype=torch.float)
