"""Training script for the Optiver volatility prediction model."""

import os

import numpy as np
import pytorch_lightning as pl
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


def main():
    """Run the training pipeline including data preparation, training and evaluation."""
    # Setup DVC and pull data
    setup_dvc()

    # Load and prepare data
    # (3830, 112, 200, 21)
    train_data, train_extra, train_norm = load_and_prepare_data(
        data_dir="data/optiver-realized-volatility-prediction"
    )
    # Setup cross-validation
    cv = KFold(5, shuffle=True, random_state=1)
    time_ids = train_extra.indexes["time_id"].values
    split_index = int(len(time_ids) * 0.9)
    train_val_time_ids = time_ids[:split_index]
    test_time_ids = time_ids[split_index:]
    train_time_ids, val_time_ids = next(cv.split(train_val_time_ids))
    test_time_ids_indices = np.where(np.isin(time_ids, test_time_ids))[0]

    # Create datasets
    mode = "multi-stock"  # or 'single-stock'
    train_ds = OptiverDataset(
        train_data, train_extra, mode, time_ids[train_time_ids], n_stocks=112
    )
    val_ds = OptiverDataset(
        train_data, train_extra, mode, time_ids[val_time_ids], n_stocks=112
    )
    test_ds = OptiverDataset(
        train_data, train_extra, mode, time_ids[test_time_ids_indices], n_stocks=112
    )

    # Create dataloaders
    if mode == "multi-stock":
        train_dl = DataLoader(
            train_ds, batch_size=8, shuffle=True, num_workers=1, pin_memory=True
        )
        val_dl = DataLoader(
            val_ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=True
        )
        test_dl = DataLoader(
            test_ds, batch_size=32, shuffle=False, num_workers=1, pin_memory=True
        )
    else:
        train_dl = DataLoader(
            train_ds, batch_size=128, shuffle=True, num_workers=1, pin_memory=True
        )
        val_dl = DataLoader(
            val_ds, batch_size=128, shuffle=False, num_workers=1, pin_memory=True
        )
        test_dl = DataLoader(
            test_ds, batch_size=128, shuffle=False, num_workers=1, pin_memory=True
        )
    # Initialize model
    model = OptiverModel(
        mode=mode,
        dim=32,
        conv1_kernel=1,
        aux_loss_weight=1 if mode == "multi-stock" else 0,
    )

    # Setup trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        precision=16,
        max_epochs=1 if mode == "multi-stock" else 25,
        limit_train_batches=10,  # None if mode == 'multi-stock' else 10
    )

    # Train
    trainer.fit(model, train_dl, val_dl)

    # Test
    trainer.test(model, test_dl)


if __name__ == "__main__":
    main()
