"""Training script for the Optiver volatility prediction model."""

import os

import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from volatility_prediction.dataset import OptiverDataset
from volatility_prediction.model import OptiverModel
from volatility_prediction.utils import load_and_prepare_data


def setup_dvc():
    """Setup DVC and pull required data."""
    try:
        # Pull data from DVC storage
        os.system("dvc pull data/optiver-realized-volatility-prediction")
    except Exception as e:
        print(f"Warning: Could not pull data from DVC storage: {e}")
        print("Continuing with local data if available...")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run the training pipeline including data preparation, training and evaluation."""
    # Setup DVC and pull data
    setup_dvc()

    # Load and prepare data
    train_data, train_extra, train_norm = load_and_prepare_data(
        data_dir=cfg.data.data_dir, coarsen=cfg.data.coarsen, n_jobs=cfg.data.n_jobs
    )

    # Setup cross-validation
    cv = KFold(
        cfg.data.dataset.cv_folds,
        shuffle=True,
        random_state=cfg.data.dataset.random_seed,
    )
    time_ids = train_extra.indexes["time_id"].values
    split_index = int(len(time_ids) * cfg.data.dataset.train_val_split)
    train_val_time_ids = time_ids[:split_index]
    test_time_ids = time_ids[split_index:]
    train_time_ids, val_time_ids = next(cv.split(train_val_time_ids))
    test_time_ids_indices = np.where(np.isin(time_ids, test_time_ids))[0]

    # Create datasets
    train_ds = OptiverDataset(
        train_data,
        train_extra,
        cfg.data.dataset.mode,
        time_ids[train_time_ids],
        cfg.data.n_stocks,
    )
    val_ds = OptiverDataset(
        train_data,
        train_extra,
        cfg.data.dataset.mode,
        time_ids[val_time_ids],
        cfg.data.n_stocks,
    )
    test_ds = OptiverDataset(
        train_data,
        train_extra,
        cfg.data.dataset.mode,
        time_ids[test_time_ids_indices],
        cfg.data.n_stocks,
    )

    # Create dataloaders
    batch_sizes = (
        cfg.data.dataset.batch_sizes.multi_stock
        if cfg.data.dataset.mode == "multi-stock"
        else cfg.data.dataset.batch_sizes.single_stock
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_sizes.train,
        shuffle=True,
        num_workers=cfg.data.dataset.num_workers,
        pin_memory=cfg.data.dataset.pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_sizes.val,
        shuffle=False,
        num_workers=cfg.data.dataset.num_workers,
        pin_memory=cfg.data.dataset.pin_memory,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_sizes.test,
        shuffle=False,
        num_workers=cfg.data.dataset.num_workers,
        pin_memory=cfg.data.dataset.pin_memory,
    )

    # Initialize model
    model = OptiverModel(
        mode=cfg.data.dataset.mode,
        dim=32,
        conv1_kernel=1,
        aux_loss_weight=1 if cfg.data.dataset.mode == "multi-stock" else 0,
    )

    # Setup trainer
    trainer = pl.Trainer(
        accelerator=cfg.model.training.accelerator,
        precision=cfg.model.training.precision,
        max_epochs=cfg.model.training.max_epochs.multi_stock
        if cfg.data.dataset.mode == "multi-stock"
        else cfg.model.training.max_epochs.single_stock,
        limit_train_batches=cfg.model.training.limit_train_batches,
    )

    # Train
    trainer.fit(model, train_dl, val_dl)

    # Test
    trainer.test(model, test_dl)


if __name__ == "__main__":
    main()
