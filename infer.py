"""Module for making predictions using trained OptiverModel.

This module provides functionality to:
- Load trained model checkpoints
- Make volatility predictions on input data
- Save predictions to CSV format

The main entry point is the `infer` function, which can be called directly
or through the command line interface.
"""

import glob
import os

import fire
import pandas as pd
import torch

from volatility_prediction.model import OptiverModel


def get_latest_checkpoint():
    """Get path to the most recent model checkpoint file.

    Returns:
        str: Path to the latest checkpoint file in the lightning_logs directory.

    Raises:
        ValueError: If no checkpoint files are found.
    """
    # Get the latest version directory
    version_dirs = glob.glob("lightning_logs/version_*")
    latest_version = max(version_dirs, key=os.path.getctime)

    # Get the latest checkpoint file
    checkpoints = glob.glob(f"{latest_version}/checkpoints/*.ckpt")
    if not checkpoints:
        raise ValueError("No checkpoint files found!")

    return max(checkpoints, key=os.path.getctime)


def load_model(checkpoint_path=None):
    """Load a trained OptiverModel from a checkpoint file.

    Args:
        checkpoint_path (str, optional): Path to specific checkpoint file.
            If None, uses the latest checkpoint. Defaults to None.

    Returns:
        OptiverModel: Loaded model in evaluation mode.
    """
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()

    model = OptiverModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def predict(model, data):
    """Make predictions using the model.

    Args:
        model: Loaded OptiverModel
        data: Input data tensor of shape (batch_size, n_stocks, timesteps, features)

    Returns:
        Dictionary containing predictions for both volatility windows
    """
    with torch.no_grad():
        predictions = model(data)
    return predictions


def infer(
    input_path: str, checkpoint_path: str = None, output_path: str = "predictions.csv"
):
    """Load model and make predictions on input data.

    Args:
        input_path: Path to input data file
        checkpoint_path: Path to specific model checkpoint (optional)
        output_path: Path to save predictions (default: predictions.csv)
    """
    # Load model
    model = load_model(checkpoint_path)

    # Load and preprocess data
    # Note: Modify this according to your data format
    data = torch.load(input_path)

    # Make predictions
    predictions = predict(model, data)

    # Convert predictions to DataFrame
    results = pd.DataFrame(
        {
            "volatility": predictions["vol"].cpu().numpy().flatten(),
            "volatility_2nd_half": predictions["vol2"].cpu().numpy().flatten(),
        }
    )

    # Save predictions
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return results


if __name__ == "__main__":
    fire.Fire(infer)
