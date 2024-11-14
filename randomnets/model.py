import pytorch_lightning

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR


class RandomNetsModel(pytorch_lightning.LightningModule):
    def __init__(
        self,
        mask_thr: float = 0.5,  # 0 is no mask, 1.0 is drop all input!
        dim: int = 1024,
        in_dim: int = 4096,
        n_nns: int = 50,
        max_lr: float = 1e-3,
        dropout: float = 0.25,
        n_hidden_layers: int = 1,
        max_epochs: int = 10,
        # id_embedding_dim=8,  # Put for zero to disable
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
        # self.id_embedding_dim = id_embedding_dim

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
        )  # TODO Some features may not be used at all? Can this be done more "intelligent?", e.g. chance of getting selected for next tree is related to if feature is already used?

    def create_layers(self):
        self.dropout_fn = nn.Dropout(self.dropout)
        self.activation_fn = nn.LeakyReLU()  # nn.ReLU()

        # self.id_embedding = nn.Embedding(
        #     num_embeddings=self.n_nns, embedding_dim=self.id_embedding_dim
        # )

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

    def forward(
        self, fp, sample_mask
    ):  # TODO make a default to use ALL input masks, if none provided
        fp_mask = self.get_fp_masks(sample_mask)

        n_ensemble_sel = sample_mask.shape[1]
        fp_reshaped = fp.unsqueeze(2).expand(
            -1, -1, n_ensemble_sel
        )  # -> samples, fp_size, n_nns_sel
        fp_masked = fp_reshaped * fp_mask

        # if self.id_embedding_dim:  # Set to zero to inactivate
        #     emb_id = self.id_embedding(sample_mask).transpose(1, 2)
        #     fp_masked = torch.concat((fp_masked, emb_id), dim=1)

        emb_o = self.embedding(fp_masked)

        ff_o = self.FF(emb_o)
        y_hats = self.predict_nn(ff_o)
        # print(y_hats.shape)
        # raise Exception(y_hats.shape)
        return y_hats.squeeze(
            dim=1
        )  # Dims are then samples, n_nns_sel #This fails if n_nns_sel is one!

    def get_loss(self, batch):
        fp, y, sample_mask = batch

        # Add n_nns dim and repeat y n_nns_sel times
        ys = torch.unsqueeze(y, 1).expand(-1, sample_mask.shape[1]).detach()

        y_hats = self.forward(fp, sample_mask)

        loss = torch.nn.functional.mse_loss(y_hats, ys)

        # loss = torch.nn.functional.mse_loss(y_hats, ys, reduction="none")
        # loss = (
        #     (loss * mask).sum() / mask.sum()
        # )  # Normalize so that only the one that are not masked are used in calculation!

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


class RandomNetsModelSeperate(pytorch_lightning.LightningModule):
    def __init__(
        self,
        mask_thr: float = 0.5,  # 0 is no mask, 1.0 is drop all input!
        dim: int = 1024,
        in_dim: int = 4096,
        n_nns: int = 50,
        max_lr: float = 1e-3,
        dropout: float = 0.25,
        n_hidden_layers: int = 1,
        max_epochs: int = 10,
        # id_embedding_dim=8,  # Put for zero to disable
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
        # self.id_embedding_dim = id_embedding_dim

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
        )  # TODO Some features may not be used at all? Can this be done more "intelligent?", e.g. chance of getting selected for next tree is related to if feature is already used?

    def create_layers(self):
        self.dropout_fn = nn.Dropout(self.dropout)
        self.activation_fn = nn.LeakyReLU()

        # Create separate networks for each ensemble member
        self.embedding_nn = nn.ModuleList(
            [nn.Linear(self.in_dim, self.dim) for _ in range(self.n_nns)]
        )

        # Create hidden layers for each ensemble member
        self.hidden_layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.dim, self.dim)
                        for _ in range(self.n_hidden_layers - 1)
                    ]
                )
                for _ in range(self.n_nns)
            ]
        )

        # Create output layer for each ensemble member
        self.predict_nn = nn.ModuleList(
            [nn.Linear(self.dim, 1) for _ in range(self.n_nns)]
        )

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
        """
        Args:
            fp: Input features [batch_size, in_dim]
            sample_mask: Which networks to use for each sample [batch_size, n_sel_nns]
                        Contains indices from 0 to n_nns-1
        """
        # if torch.cuda.is_available():
        #     print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        #     print(f"Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        batch_size, n_sel_nns = sample_mask.shape

        # 1. Apply input masking specific to each selected network
        # get_fp_masks uses the fixed input_mask specific to each network
        fp_mask = self.get_fp_masks(sample_mask)  # [batch_size, in_dim, n_sel_nns]

        # 2. Prepare input features
        # Expand features to match each selected network
        fp_reshaped = fp.unsqueeze(2).expand(
            -1, -1, n_sel_nns
        )  # [batch_size, in_dim, n_sel_nns]
        # Apply network-specific input masks
        fp_masked = fp_reshaped * fp_mask  # [batch_size, in_dim, n_sel_nns]

        # 3. Reshape for processing each sample-network pair independently
        # Transpose to get features as last dimension
        fp_masked = fp_masked.transpose(1, 2)  # [batch_size, n_sel_nns, in_dim]
        # Combine batch and network dimensions
        fp_masked = fp_masked.reshape(
            -1, self.in_dim
        )  # [batch_size * n_sel_nns, in_dim]

        # 4. Get which network to use for each combined batch item
        network_indices = sample_mask.reshape(-1)  # [batch_size * n_sel_nns]

        # 5. Process through networks
        outputs = []
        for i in range(len(fp_masked)):
            # Get the network index for this sample
            net_idx = network_indices[i]
            # Get the masked features for this sample
            x = fp_masked[i]

            # Embedding layer
            x = self.embedding_nn[net_idx](x)
            x = self.activation_fn(x)
            x = self.dropout_fn(x)

            # Hidden layers
            for hidden in self.hidden_layers[net_idx]:
                x = hidden(x)
                x = self.activation_fn(x)
                x = self.dropout_fn(x)

            # Output layer
            x = self.predict_nn[net_idx](x)
            outputs.append(x)

        # 6. Combine results
        outputs = torch.stack(outputs)  # [batch_size * n_sel_nns, 1]
        # Reshape back to separate batch and network dimensions
        outputs = outputs.reshape(batch_size, n_sel_nns)  # [batch_size, n_sel_nns]

        return outputs

    def get_loss(self, batch):
        fp, y, sample_mask = batch

        # Add n_nns dim and repeat y n_nns_sel times
        ys = torch.unsqueeze(y, 1).expand(-1, sample_mask.shape[1]).detach()

        y_hats = self.forward(fp, sample_mask)

        loss = torch.nn.functional.mse_loss(y_hats, ys)

        # loss = torch.nn.functional.mse_loss(y_hats, ys, reduction="none")
        # loss = (
        #     (loss * mask).sum() / mask.sum()
        # )  # Normalize so that only the one that are not masked are used in calculation!

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
