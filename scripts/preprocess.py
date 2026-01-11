from logger import setup_logger
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from utils.loader import load_contractions
import re
import inflect
import numpy as np

# setup logger
log = setup_logger(__name__, "preprocessor.log")

p = inflect.engine()


class Preprocessor:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"
        self.train_path = self.data_dir / "train.csv"
        self.data_path = self.data_dir / "data.csv"
        self.df = None
        self.contractions = load_contractions()

    def clean(self, text):
        text = text.lower()
        # remove html tags
        text = BeautifulSoup(str(text), "html.parser").get_text()
        # expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        # removes everything except letters
        text = re.sub(r"[^a-z\s]", " ", text)
        return " ".join(text.split())

    def load_data(self):
        try:
            self.df = pd.read_csv(self.train_path)
            log.info(f"Training data Loaded from {self.train_path.name}")
        except Exception as e:
            log.error(f"Failed to load data: {e}")

    def clean_data(self):
        # remove missing values
        self.df.dropna(inplace=True)
        log.info("Removed null values")
        # preprocess the text
        log.info("Data cleaning in process ...")
        self.df["question1"] = self.df["question1"].apply(self.clean)
        self.df["question2"] = self.df["question2"].apply(self.clean)
        log.info("Data cleaning successful")

    def save_data(self):
        self.df.to_csv(self.data_path, index=False)
        log.info(f"Processed data saved to {self.data_path.name}")


if __name__ == "__main__":
    processor = Preprocessor()

    # processing pipeline
    processor.load_data()
    processor.clean_data()
    processor.save_data()

    # preprocessing complete
    log.info("processing pipeline completed successfully")
