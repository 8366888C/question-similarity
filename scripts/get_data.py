from dotenv import load_dotenv
import os
from pathlib import Path
from logger import setup_logger
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# load environment variable
load_dotenv()

# setup logger
log = setup_logger(__name__, "get_data.log")


def get_data():
    # creating data folder
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    log.info(f"Created directory: {data_dir}")

    # kaggle api initialization
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        log.error("KAGGLE_API_TOKEN not found in .env")
    api = KaggleApi()
    api.authenticate()
    log.info("Kaggle API authenticated")

    # downloading zip files
    log.info(f"Downloading data to {data_dir} ...")
    try:
        api.competition_download_files(
            "quora-question-pairs", path=data_dir, quiet=False
        )
        log.info("Download complete")
    except Exception as e:
        log.error(f"Failed to download data: {e}")
        return

    # extracting zip files
    log.info("Unzipping files ...")
    for item in data_dir.iterdir():
        if item.suffix == ".zip":
            file_path = data_dir / item
            try:
                with zipfile.ZipFile(file_path, "r") as zip:
                    zip.extractall(data_dir)
                log.info(f"Extracted: {item.name}")
            except zipfile.BadZipFile:
                log.error(f"Bad zip file: {item.name}")

    # handling final test dataset
    test_final = data_dir / "test.csv"
    if test_final.exists():
        test_final.rename(data_dir / "final.csv")
        log.info("Renamed initial test.csv to final.csv")

    # extracting training and testing datasets
    log.info("Extracting the datasets ...")
    for item in data_dir.iterdir():
        if item.suffix == ".zip":
            try:
                with zipfile.ZipFile(item, "r") as zip:
                    zip.extractall(data_dir)
                log.info(f"Extracted: {item.name}")
            except zipfile.BadZipFile:
                log.error(f"Bad zip file: {item.name}")
