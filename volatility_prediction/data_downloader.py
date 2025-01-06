import os

from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(competition_name: str):
    # Initialize the API
    api = KaggleApi()
    api.authenticate()

    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)

    # Download files from competition
    print(f"Downloading {competition_name} competition files...")
    api.competition_download_files(competition_name, path="./data/")
    print("Download complete!")


if __name__ == "__main__":
    competition = "optiver-realized-volatility-prediction"
    download_kaggle_dataset(competition)
