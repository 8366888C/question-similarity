from logger import setup_logger
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from utils.loader import load_contractions
import re
import inflect
import numpy as np

# setup logger
log = setup_logger(__name__, "preprocess.log")

p = inflect.engine()


class DataPreprocessor:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"
        self.train_path = self.data_dir / "train.csv"
        self.data_path = self.data_dir / "data.csv"
        self.df = None
        self.contractions = load_contractions()

    def preprocess(self, text):
        text = text.lower()
        # remove html tags
        text = BeautifulSoup(str(text), "html.parser").get_text()
        # expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        # numeric to text
        text = re.sub(r"(\d+)", lambda x: p.number_to_words(x.group(0)), text)
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
        self.df["question1"] = self.df["question1"].apply(self.preprocess)
        self.df["question2"] = self.df["question2"].apply(self.preprocess)
        log.info("Data cleaning successful")

    def feature_engineering(self):
        # count of words
        self.df["q1_len"] = self.df["question1"].apply(lambda x: len(str(x).split()))
        log.info("Added feature q1_len")
        self.df["q2_len"] = self.df["question2"].apply(lambda x: len(str(x).split()))
        log.info("Added feature q2_len")
        # common words
        self.df["common_words"] = self.df.apply(
            lambda x: len(
                set(str(x["question1"]).lower().split())
                & set(str(x["question2"]).lower())
            ),
            axis=1,
        )
        log.info("Added feature common_words")
        # unique words
        self.df["unique_words"] = self.df.apply(
            lambda x: len(
                set(str(x["question1"]).lower().split())
                | set(str(x["question2"]).lower().split())
            ),
            axis=1,
        )
        log.info("Added feature unique_words")
        # word share
        self.df["word_share"] = (
            (self.df["common_words"] / self.df["unique_words"]) * 100
        ).round(2)
        log.info("Added feature word_share")
        # question frequency in the whole dataset
        qid = pd.Series(self.df["qid1"].tolist() + self.df["qid2"].tolist())
        qid_counts = qid.value_counts()
        self.df["q1_freq"] = self.df["qid1"].map(qid_counts)
        log.info("Added feature q1_freq")
        self.df["q2_freq"] = self.df["qid2"].map(qid_counts)
        log.info("Added feature q2_freq")
        # sum of frequencies: how popular is the pair?
        self.df["freq_sum"] = self.df["q1_freq"] + self.df["q2_freq"]
        log.info("Added feature freq_sum")
        # difference of frequencies: is one question more popular than the other
        self.df["freq_diff"] = abs(self.df["q1_freq"] - self.df["q2_freq"])
        log.info("Added feature freq_diff")

    def handle_outliers(self):
        # features with outliers or high range
        uncapped_cols = ["q1_len", "q2_len", "unique_words", "word_share"]
        for col in uncapped_cols:
            q_limit = self.df[col].quantile(0.99)
            self.df[col] = self.df[col].clip(upper=q_limit)
            log.info(f"Removing outliers from {col}")
        # features with high skewness
        skewed_cols = ["q1_freq", "q2_freq", "freq_sum", "freq_diff"]
        for col in skewed_cols:
            self.df[col] = np.log1p(self.df[col])
            log.info(f"Performing logarithmic transform on {col}")

    def save_data(self):
        self.df.to_csv(self.data_path, index=False)
        log.info(f"Preprocessed data saved to {self.data_path.name} ")


if __name__ == "__main__":
    preprocessor = DataPreprocessor()

    # preprocessing pipeline
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.feature_engineering()
    preprocessor.handle_outliers()
    preprocessor.save_data()

    # preprocessing complete
    log.info("Preprocessing pipeline completed successfully")
