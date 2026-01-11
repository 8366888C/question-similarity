from logger import setup_logger
from pathlib import Path
import pandas as pd
import numpy as np

# setup logger
log = setup_logger(__name__, "feature_engineering.log")


class FeatureEngineering:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"
        self.data_path = self.data_dir / "data.csv"
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            log.info(f"Cleaned data loaded from {self.data_path.name}")
        except Exception as e:
            log.error(f"Failed to load data: {e}")

    def raw_features(self):
        # count of words
        self.df["q1_len"] = self.df["question1"].apply(lambda x: len(str(x).split()))
        log.info("Added feature q1_len")

        self.df["q2_len"] = self.df["question2"].apply(lambda x: len(str(x).split()))
        log.info("Added feature q2_len")

        # common words
        self.df["common_words"] = self.df.apply(
            lambda x: len(
                set(str(x["question1"]).lower().split())
                & set(str(x["question2"]).lower().split())
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

        # first word equal or not?
        self.df["first_word"] = self.df.apply(
            lambda x: (
                1
                if str(x["question1"]).split()[:1] == str(x["question2"]).split()[-1:]
                else 0
            ),
            axis=1,
        )

        # last word equal or not?
        self.df["last_word"] = self.df.apply(
            lambda x: (
                1
                if str(x["question1"]).split()[:1] == str(x["question2"]).split()[-1:]
                else 0
            ),
            axis=1,
        )

        # question frequency in the whole dataset
        qid = pd.Series(self.df["qid1"].tolist() + self.df["qid2"].tolist())
        qid_counts = qid.value_counts()

        self.df["q1_freq"] = self.df["qid1"].map(qid_counts)
        log.info("Added feature q1_freq")

        self.df["q2_freq"] = self.df["qid2"].map(qid_counts)
        log.info("Added feature q2_freq")

    def handle_distribution(self):
        # features with outliers or high range
        uncapped_cols = ["q1_len", "q2_len"]
        for col in uncapped_cols:
            upper_limit = self.df[col].quantile(0.99)
            self.df[col] = self.df[col].clip(upper=upper_limit)
            log.info(f"Removed outliers from {col}")

        # features with high skewness
        skewed_cols = ["q1_freq", "q2_freq"]
        for col in skewed_cols:
            self.df[col] = np.log1p(self.df[col])
            log.info(f"Performed logarithmic transform on {col}")

    def derived_features(self):
        # length based
        self.df["len_diff"] = abs(self.df["q1_len"] - self.df["q2_len"])
        log.info("Added feature len_diff")

        # sum of frequencies: how popular is the pair?
        self.df["freq_sum"] = self.df["q1_freq"] + self.df["q2_freq"]
        log.info("Added feature freq_sum")

        # difference of frequencies: is one question more popular than the other
        self.df["freq_diff"] = abs(self.df["q1_freq"] - self.df["q2_freq"])
        log.info("Added feature freq_diff")

    def reduce_multicollinearity(self):
        cols_to_drop = [
            "q1_len",
            "q2_len",
            "common_words",
            "unique_words",
            "q1_freq",
            "q2_freq",
        ]
        self.df = self.df.drop(columns=cols_to_drop)
        log.info(f"Removed features: {", ".join(cols_to_drop)}")

    def save_data(self):
        self.df.to_csv(self.data_path, index=False)
        log.info(f"Engineered data saved to {self.data_path.name}")


if __name__ == "__main__":
    engineer = FeatureEngineering()

    # feature engineering pipeline
    engineer.load_data()
    engineer.raw_features()
    engineer.handle_distribution()
    engineer.derived_features()
    engineer.reduce_multicollinearity()
    engineer.save_data()

    # feature engineering complete
    log.info("feature engineering pipeline completed succesfully")
