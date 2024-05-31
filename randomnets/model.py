import pytorch_lightning

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR


class RandomNetsModel(pytorch_lightning.LightningModule):
    def __init__(
        self,
        mask_thr: float = 0.5,
        dim: int = 1024,
        in_dim: int = 4096,
        n_nns: int = 25,
        learning_rate: float = 1e-3,
        max_lr: float = 1e-3,
        dropout: float = 0.25,
        n_hidden_layers: int = 2,
        max_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # TODO, substitute with calls to hparams
        self.mask_thr = mask_thr
        self.dim = dim
        self.in_dim = in_dim
        self.n_nns = n_nns
        self.max_lr = max_lr
        self.dropout = dropout
        self.n_hidden_layers = n_hidden_layers
        self.max_epochs = max_epochs

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

    def forward(self, masked_fp):
        emb_o = self.embedding(masked_fp)
        ff_o = self.FF(emb_o)
        y_hats = self.predict_nn(ff_o)
        return y_hats.squeeze()  # Dims are then samples, n_nns_sel

    def get_loss(self, batch):
        fp, y, sample_mask = batch

        n_ensemble_sel = sample_mask.shape[1]

        fp_mask = self.get_fp_masks(sample_mask)
        fp_reshaped = fp.unsqueeze(2).expand(
            -1, -1, n_ensemble_sel
        )  # -> samples, fp_size, n_nns_sel
        fp_masked = fp_reshaped * fp_mask

        # Add n_nns dim and repeat y n_nns_sel times
        ys = torch.unsqueeze(y, 1).expand(-1, n_ensemble_sel).detach()

        y_hats = self.forward(fp_masked)

        loss = torch.nn.functional.mse_loss(y_hats, ys)

        # loss = torch.nn.functional.mse_loss(y_hats, ys, reduction="none")
        # loss = (
        #     (loss * mask).sum() / mask.sum()
        # )  # Normalize so that only the one that are not masked are used in calculation!

        # y_hat_ensemble = (y_hats * mask).sum(axis=1) / mask.sum(axis=1).detach()
        # ensemble_loss = torch.nn.functional.mse_loss(y_hat_ensemble, y).detach()

        return loss  # , ensemble_loss

    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.get_loss(batch)
        self.log("train_mse_loss", loss)
        # self.log("train_mse_ensemble_loss", ensemble_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # TODO, double check the masking and eval score, its way lower than with a dedicated validation set
        self.eval()
        loss = self.get_loss(batch)
        self.log("val_mse_loss", loss)
        # self.log("val_mse_ensemble_loss", ensemble_loss)
        self.log("hp_metric", loss)
        return loss

    def test_step(self, batch, batch_idx):
        self.eval()
        loss, ensemble_loss = self.get_loss(batch)
        self.log("mse_loss", loss)
        self.log("mse_ensemble_loss", ensemble_loss)
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
            max_momentum=0.95,  # These need to be tuned?
            div_factor=1e3,
            final_div_factor=1e3,
            last_epoch=-1,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]

    def predict(self, fp, std=False):
        self.eval()
        input_mask = self.input_mask
        n_ensemble = input_mask.shape[1]

        input_mask_reshaped = input_mask.unsqueeze(0).expand(
            fp.shape[0], -1, -1
        )  # -> samples, fp_size, n_nns

        fp_reshaped = fp.unsqueeze(2).expand(
            -1, -1, n_ensemble
        )  # -> samples, fp_size, n_nns

        fp_masked = fp_reshaped * input_mask_reshaped

        y_hats = self.forward(fp_masked).detach()

        # take the mean and std along n_nns
        if std:
            return y_hats.mean(axis=1), y_hats.std(axis=1)
        else:
            return y_hats.mean(axis=1)
