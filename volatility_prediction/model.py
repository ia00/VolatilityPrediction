import einops
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from .attention import StockAttention, TimeAttention

n_stocks = 112
coarsen = 3


class OptiverModel(pl.LightningModule):
    def __init__(
        self,
        mode="multi-stock",
        dim=32,
        conv1_kernel=3,
        rnn_layers=2,
        rnn_dropout=0.3,
        n_features=21,
        aux_loss_weight=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.stock_emb = nn.Embedding(n_stocks, dim)
        self.stock_emb.weight.data.normal_(0, 0.2)
        self.conv1 = nn.Conv1d(n_features, dim, conv1_kernel, conv1_kernel)
        self.conv2 = nn.Conv1d(dim, dim, 1, 1)
        self.norm1 = nn.LayerNorm([n_stocks, dim])
        self.norm2 = nn.LayerNorm([n_stocks, dim])
        self.rnn = nn.GRU(dim, dim, rnn_layers, batch_first=True, dropout=rnn_dropout)
        self.timesteps_attn = TimeAttention(600 // conv1_kernel // coarsen)
        self.timesteps_attn2 = TimeAttention(300 // conv1_kernel // coarsen)
        self.stock_attn = StockAttention(dim)
        self.fc_out1 = nn.Linear(dim, 1)
        self.fc_out2 = nn.Linear(dim, 1)
        self.history = pd.DataFrame()

    def forward(self, x, stock_ind=None):
        # x: (b, st, t, f)
        x = einops.rearrange(x, "b st t f -> (b st) f t")
        x = self.conv1(x)
        x = einops.rearrange(x, "(b st) f t -> b t st f", st=n_stocks)
        x = F.gelu(x)
        x = self.norm1(x)
        x = einops.rearrange(x, "b t st f -> (b st) f t")
        x = self.conv2(x)
        x = einops.rearrange(x, "(b st) f t -> b t st f", st=n_stocks)
        x = F.gelu(x)
        x = self.norm2(x)
        x = einops.rearrange(x, "b t st f -> b st t f")
        x = self.stock_attn(x)
        x = x + self.stock_emb.weight[None, :, None, :]
        if self.hparams.mode == "single-stock":
            x = x[torch.arange(len(x)), stock_ind][:, None]
        x = einops.rearrange(x, "b st t f -> (b st) t f")
        x = self.rnn(x)[0]
        x = einops.rearrange(
            x,
            "(b st) t f -> b st t f",
            st=n_stocks if self.hparams.mode == "multi-stock" else 1,
        )
        x1 = self.timesteps_attn(x)
        x2 = self.timesteps_attn2(x[:, :, : self.timesteps_attn2.steps, :])
        x1 = self.fc_out1(x1)
        x2 = self.fc_out2(x2)
        x1 = x1 * 0.63393 - 5.762331
        x2 = x2 * 0.67473418 - 6.098946
        x1 = torch.exp(x1)
        x2 = torch.exp(x2)
        if self.hparams.mode == "single-stock":
            return {"vol": x1[:, 0, 0], "vol2": x2[:, 0, 0]}  # (b,)  # (b,)
        else:
            return {"vol": x1[..., 0], "vol2": x2[..., 0]}  # (b, st)  # (b, st)

    def training_step(self, batch, batch_ind):
        out = self.common_step(batch, "train")
        return out

    def validation_step(self, batch, batch_ind):
        return self.common_step(batch, "valid")

    def common_step(self, batch, stage):
        out = self(
            batch["data"],
            batch["stock_ind"] if self.hparams.mode == "single-stock" else None,
        )
        mask1 = ~torch.isnan(batch["target"])
        target1 = torch.where(
            mask1, batch["target"], torch.tensor(1.0, device=self.device)
        )
        mask2 = batch["current_vol_2nd_half"] > 0
        target2 = torch.where(
            mask2, batch["current_vol_2nd_half"], torch.tensor(1.0, device=self.device)
        )
        vol_loss = (((out["vol"] - target1) / target1) ** 2)[mask1].mean() ** 0.5
        vol2_loss = (((out["vol2"] - target2) / target2) ** 2)[mask2].mean() ** 0.5
        loss = vol_loss + self.hparams.aux_loss_weight * vol2_loss
        self.log(f"{stage}/loss", loss.item(), on_step=False, on_epoch=True)
        self.log(f"{stage}/vol_loss", vol_loss.item(), on_step=False, on_epoch=True)
        self.log(f"{stage}/vol2_loss", vol2_loss.item(), on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "target": batch["target"],
            "vol": out["vol"].detach(),
            "time_id": batch["time_id"],
        }

    def common_epoch_end(self, outs, stage):
        target = torch.cat([x["target"] for x in outs])
        vol = torch.cat([x["vol"] for x in outs])
        mask = ~torch.isnan(target)
        target = torch.where(mask, target, torch.tensor(1.0, device=self.device))
        rmspe = (((vol - target) / target) ** 2)[mask].mean() ** 0.5
        self.log(f"{stage}/rmspe", rmspe, prog_bar=True, on_step=False, on_epoch=True)
        self.history.loc[self.trainer.current_epoch, f"{stage}/rmspe"] = rmspe.item()

    def training_epoch_end(self, outs):
        self.common_epoch_end(outs, "train")

    def validation_epoch_end(self, outs):
        self.common_epoch_end(outs, "valid")

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=0.001)
        #         opt = Adam(self.parameters(), lr=0.0005) # single-stock
        sched = {
            "scheduler": ExponentialLR(opt, 0.93),
            #             'scheduler': ExponentialLR(opt, 0.9), #  single-stock
            "interval": "epoch",
        }
        return [opt], [sched]

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, "test")
