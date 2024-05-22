import pytorch_lightning

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR


class RandomNetsModel(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        self.mask_thr = 0.5
        self.dim = 1024
        self.in_dim = 4096
        self.n_nns = 25
        self.learning_rate = 1e-3
        self.max_lr = 1e-3
        self.dropout = 0.25
        self.n_hidden_layers = 2
        self.max_epochs = 10

        self.create_input_mask()
        self.create_layers()
        self.float()

    def create_input_mask(self):
        random_tensor = torch.rand(
            (
                self.in_dim,
                self.n_nns,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "input_mask", random_tensor > self.mask_thr, persistent=True
        )  # Some features may not be used at all? Can this be done more "intelligent?", e.g. chance of getting selected for next tree is related to if feature is already used?

    def create_layers(self):
        self.dropout_fn = nn.Dropout(self.dropout)
        self.activation_fn = nn.LeakyReLU()  # nn.ReLU()

        # Conv1D is (N,Cin,L), N is a batch size, C denotes a number of channels, L is a length of signal sequence.
        # So if FP is channels, hidden is Cout, L is n_nns, and stride and kernel_size should be 1.
        self.embedding_nn = nn.Conv1d(
            self.in_dim, self.dim, stride=1, kernel_size=1, padding=0
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

    def forward(self, fp):
        # fp dim is samples, in_dim
        fp_reshaped = torch.unsqueeze(fp, 2)  # Add the n_nn dim

        masked_fp = fp_reshaped * self.input_mask

        emb_o = self.embedding(masked_fp)
        ff_o = self.FF(emb_o)
        y_hats = self.predict_nn(ff_o)
        return y_hats.squeeze()  # Dims are then samples, n_nns

    def get_loss(self, batch, revert_mask=False):
        fp, y, mask = batch

        if revert_mask:
            mask = 1 - mask

        # Add n_nns dim and repeat y five times
        ys = torch.unsqueeze(y, 1).repeat(1, self.n_nns)

        y_hats = self.forward(fp)

        # These lines will come in handy for the sample masking
        # mse_loss = torch.nn.functional.mse_loss(y_hat, y, reduce=False)
        # weighted_loss = mse_loss * weights

        loss = torch.nn.functional.mse_loss(y_hats, ys, reduce=False)
        loss = (
            (loss * mask).sum() / mask.sum()
        )  # Normalize so that only the one that are not masked are used in calculation!
        return loss

    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.get_loss(batch)
        self.log("train_mse_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO, double check the masking and eval score, its way lower than with a dedicated validation set
        self.eval()
        loss = self.get_loss(batch, revert_mask=True)
        self.log("val_mse_loss", loss)
        return loss

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    #     return optimizer

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
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,  # These need to be tuned?
            div_factor=1e3,
            final_div_factor=1e3,
            last_epoch=-1,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]

    def predict(self, fp, std=False):
        self.eval()
        y_hats = self.forward(fp).detach()
        if std:
            return y_hats.mean(axis=1), y_hats.std(axis=1)  # take the mean along n_nns
        else:
            return y_hats.mean(axis=1)
