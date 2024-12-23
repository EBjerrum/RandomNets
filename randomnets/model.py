import pytorch_lightning

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np


def create_balanced_mask(n_features, n_ensemble_members, mask_thr=0.5):
    mask = torch.zeros((n_features, n_ensemble_members), dtype=bool)
    n_features_per_member = int(n_features * (1 - mask_thr))

    # Initialize pool of available indices
    available_indices = list(range(n_features))

    # For each ensemble member
    for member in range(n_ensemble_members):
        # Simple if we have more available than needed to sample
        if len(available_indices) >= n_features_per_member:
            sampled_indices = np.random.choice(
                available_indices, size=n_features_per_member, replace=False
            )
            for idx in sampled_indices:
                if idx in available_indices:
                    available_indices.remove(idx)
        else:
            # Otherwise we need to sample more cleverly
            sampled_indices = available_indices.copy()  # Empty the available indices
            # Refill the available indices.
            available_indices = list(range(n_features))

            # Make a sample buffer to sample from
            sample_buffer = available_indices.copy()

            # Remove the already sampled indices from the sample buffer
            for idx in sampled_indices:
                if idx in sample_buffer:
                    sample_buffer.remove(idx)

            # Sample from the sample buffer
            sampled_indices_fillup = np.random.choice(
                sample_buffer,
                size=n_features_per_member - len(sampled_indices),
                replace=False,
            )

            # remove the fillup indices from the available indices
            for idx in sampled_indices_fillup:
                if idx in available_indices:
                    available_indices.remove(idx)

            # Concatenate the two sets of sampled indices
            sampled_indices = np.concatenate([sampled_indices, sampled_indices_fillup])

        # Refill if completely empty
        if len(available_indices) == 0:
            available_indices = list(range(n_features))
        mask[sampled_indices, member] = True

    return mask


class RandomNetsModel(pytorch_lightning.LightningModule):
    def __init__(
        self,
        mask_thr: float = 0.5,  # 0 is no mask, 1.0 is drop all input!
        dim: int = 1024,
        in_dim: int = 4096,
        n_nns: int = 25,
        max_lr: float = 1e-3,
        dropout: float = 0.25,
        n_hidden_layers: int = 1,
        max_epochs: int = 10,
        balanced_mask: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.mask_thr = mask_thr
        self.dim = dim
        self.in_dim = in_dim
        self.n_nns = n_nns
        self.max_lr = max_lr
        self.dropout = dropout
        self.n_hidden_layers = n_hidden_layers
        self.max_epochs = max_epochs
        self.balanced_mask = balanced_mask

        self.create_input_mask()
        self.create_layers()
        self.float()

    def create_input_mask(self):
        if self.balanced_mask:
            input_mask = create_balanced_mask(self.in_dim, self.n_nns, self.mask_thr)
        else:
            random_tensor = torch.rand(
                (
                    self.in_dim,
                    self.n_nns,
                ),
                requires_grad=False,
            )
            input_mask = (
                random_tensor > self.mask_thr
            )  # If mask_the is 0, all is included
        self.register_buffer(
            "input_mask",
            input_mask,
            persistent=True,
        )  # TODO Some features may not be used at all? Can this be done more "intelligent?", e.g. chance of getting selected for next tree is related to if feature is already used?

    def create_layers(self):
        self.dropout_fn = nn.Dropout(self.dropout)
        self.activation_fn = nn.LeakyReLU()  # nn.ReLU()

        # Conv1D is (N,Cin,L), N is a batch size, C denotes a number of channels, L is a length of signal sequence.
        # So if FP is channels, hidden is Cout, L is n_nns, and stride and kernel_size should be 1.
        self.embedding_nn = nn.Conv1d(
            self.in_dim,  # + self.id_embedding_dim,
            self.dim,
            stride=1,
            kernel_size=1,
            padding=0,
        )
        self.embedding = nn.Sequential(
            self.embedding_nn, self.activation_fn, self.dropout_fn
        )  # Out is samples, hidden_dim, n_nns

        hidden_stack = []
        for i in range(self.n_hidden_layers - 1):  # Embedding counts for one
            hidden_nn = nn.Conv1d(
                self.dim, self.dim, stride=1, kernel_size=1, padding=0
            )  # samples, hidden_dim, n_nns
            hidden_stack.extend([hidden_nn, self.activation_fn, self.dropout_fn])
        self.FF = nn.Sequential(*hidden_stack)

        self.predict_nn = nn.Conv1d(self.dim, 1, kernel_size=1, padding=0, stride=1)

    def get_fp_masks(self, sample_mask):
        sample_mask_reshaped = sample_mask.unsqueeze(1).expand(
            -1, self.in_dim, -1
        )  # -> samples, fp_size, n_sel_nns

        input_mask_reshaped = self.input_mask.unsqueeze(0).expand(
            sample_mask.shape[0], -1, -1
        )  # -> samples, fp_size, n_nns

        fp_mask_gathered = torch.gather(
            input_mask_reshaped, 2, sample_mask_reshaped
        )  # Gathers according to indices -> samples, fp_size, n_sel_nns

        return fp_mask_gathered

    def forward(self, fp, sample_mask):
        fp_mask = self.get_fp_masks(sample_mask)

        n_ensemble_sel = sample_mask.shape[1]
        fp_reshaped = fp.unsqueeze(2).expand(
            -1, -1, n_ensemble_sel
        )  # -> samples, fp_size, n_nns_sel
        fp_masked = fp_reshaped * fp_mask

        emb_o = self.embedding(fp_masked)

        ff_o = self.FF(emb_o)
        y_hats = self.predict_nn(ff_o)

        return y_hats.squeeze(
            dim=1
        )  # Dims are then samples, n_nns_sel #This fails if n_nns_sel is one!

    def get_loss(self, batch):
        fp, y, sample_mask = batch

        # Add n_nns dim and repeat y n_nns_sel times
        ys = torch.unsqueeze(y, 1).expand(-1, sample_mask.shape[1]).detach()

        y_hats = self.forward(fp, sample_mask)

        loss = torch.nn.functional.mse_loss(y_hats, ys)

        y_hat_ensemble = (y_hats).mean(axis=1)
        ensemble_loss = torch.nn.functional.mse_loss(y_hat_ensemble, y).detach()
        ensemble_std = (y_hats).std(axis=1).mean().detach()

        return loss, ensemble_loss, ensemble_std

    def training_step(self, batch, batch_idx):
        self.train()
        loss, ensemble_loss, ensemble_std = self.get_loss(batch)
        self.log("train_mse_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        loss, ensemble_loss, ensemble_std = self.get_loss(batch)
        self.log("val_mse_loss", loss)
        self.log("val_mse_ensemble_loss", ensemble_loss)
        self.log("val_mse_ensemble_std", ensemble_std)
        self.log("hp_metric", ensemble_loss)
        return loss

    def test_step(self, batch, batch_idx):
        self.eval()
        loss, ensemble_loss, ensemble_std = self.get_loss(batch)
        self.log("mse_loss", loss)
        self.log("mse_ensemble_loss", ensemble_loss)
        self.log("mse_ensemble_std", ensemble_std)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        dataloader = (
            # self.trainer._data_connector._train_dataloader_source.dataloader()
            self.trainer._data_connector._datahook_selector.datamodule.train_dataloader()
            # dm.train_dataloader()  # TODO, this needs to be better integrated!
        )  # = self.train_dataloader()

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=None,
            epochs=self.max_epochs,
            steps_per_epoch=len(
                dataloader
            ),  # We call train_dataloader, just to get the length, is this necessary?
            pct_start=0.1,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=1e3,
            final_div_factor=1e3,
            last_epoch=-1,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]

    def predict(self, fp, std=False):
        self.eval()

        sample_mask = torch.tensor(
            [range(self.n_nns)] * fp.shape[0], dtype=torch.int64
        )  # Use Full Ensemble

        y_hats = self.forward(fp, sample_mask=sample_mask).detach()

        # take the mean and std along n_nns
        if std:
            return y_hats.mean(axis=1), y_hats.std(axis=1)
        else:
            return y_hats.mean(axis=1)

    def predict_dataloader(self, dataloader, std=False):
        self.eval()
        y_hats = []
        y_hat_stds = []
        for batch in dataloader:
            y_hat_batch, y_hat_std_batch = self.predict(batch[0], std=std)
            y_hats.append(y_hat_batch)
            y_hat_stds.append(y_hat_std_batch)
        if std:
            return torch.cat(y_hats), torch.cat(y_hat_stds)
        else:
            return torch.cat(y_hats)
