from dotenv import load_dotenv

# load environment variables
load_dotenv()

import os
from pathlib import Path
from logger import setup_logger
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# setup logger
log = setup_logger(__name__, "get_data.log")


class GetData:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"
        self.api = None

    # creating data folder
    def create_folder(self):
        self.data_dir.mkdir(exist_ok=True)
        log.info(f"Created directory: {self.data_dir.name}")

    # kaggle api initialization
    def kaggle_api(self):
        kaggle_username = os.environ.get("KAGGLE_USERNAME")
        kaggle_key = os.environ.get("KAGGLE_KEY")
        if not kaggle_username or not kaggle_key:
            log.error("KAGGLE_USERNAME or KAGGLE_KEY not found in .env")
        self.api = KaggleApi()
        self.api.authenticate()
        log.info("Kaggle API authenticated")

    # downloading zip files
    def download_zip(self):
        log.info(f"Downloading data to {self.data_dir.name} ...")
        try:
            self.api.competition_download_files(
                "quora-question-pairs", path=self.data_dir, quiet=False
            )
            log.info("Download complete")
        except Exception as e:
            log.error(f"Failed to download data: {e}")
        return

    # extracting zip files
    def extract_zip(self):
        log.info("Unzipping files ...")
        for item in self.data_dir.iterdir():
            if item.suffix == ".zip":
                file_path = self.data_dir / item
        try:
            with zipfile.ZipFile(file_path, "r") as zip:
                zip.extractall(self.data_dir)
                log.info(f"Extracted: {item.name}")
        except zipfile.BadZipFile:
            log.error(f"Bad zip file: {item.name}")

    # handling final test dataset
    def rename_final(self):
        test_final = self.data_dir / "test.csv"
        if test_final.exists():
            test_final.rename(self.data_dir / "final.csv")
        log.info("Renamed initial test.csv to final.csv")

    # extracting training and testing datasets
    def extract_data(self):
        log.info("Extracting the datasets ...")
        for item in self.data_dir.iterdir():
            if item.suffix == ".zip":
                try:
                    with zipfile.ZipFile(item, "r") as zip:
                        zip.extractall(self.data_dir)
                    log.info(f"Extracted: {item.name}")
                except zipfile.BadZipFile:
                    log.error(f"Bad zip file: {item.name}")


if __name__ == "__main__":
    downloader = GetData()

    # download process
    downloader.kaggle_api()
    downloader.create_folder()
    downloader.download_zip()
    downloader.extract_zip()
    downloader.rename_final()
    downloader.extract_data()

    # download completed
    log.info("downloading process completed successfully")
