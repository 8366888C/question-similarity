from logger import setup_logger
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, SimilarityFunction
import torch

# setup logger
log = setup_logger(__name__, "embeddings.log")


class GenerateEmbeddings:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"
        self.data_path = self.data_dir / "data.csv"
        self.models_dir = self.base_dir / "app" / "models"
        self.sbert_path = self.models_dir / "sbert"
        self.embeddings_path = self.models_dir / "embeddings.pkl"
        self.df = None
        self.sbert = None
        if self.embeddings_path.exists():
            self.embeddings = joblib.load(self.embeddings_path)
            log.info("Embeddings loaded")
        else:
            self.embeddings = None
            log.error("Embeddings not found")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            log.info(f"Engineered data loaded from {self.data_path.name}")
        except Exception as e:
            log.error(f"Failed to load data: {e}")

    def sbert_init(self):
        self.sbert = SentenceTransformer(
            "all-miniLM-L6-v2", similarity_fn_name=SimilarityFunction.DOT_PRODUCT
        )
        self.sbert.save(self.sbert_path)
        log.info(f"Model initialized and saved to {self.sbert_path.name}")

    def sbert_encode(self):
        if self.embeddings is None:
            # getting all the unique questions only
            questions = (
                pd.concat([self.df["question1"], self.df["question2"]])
                .unique()
                .tolist()
            )
            log.info("All unique questions collected")
            # generating embeddings
            log.info("Encoding unique questions ...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            raw_vecs = self.sbert.encode(
                questions, device=device, show_progress_bar=True, batch_size=32
            )
            # creating cache
            self.embeddings = dict(zip(questions, raw_vecs))
            log.info(f"Encoded {len(questions)} unique questions on {device}")

    def save_cache(self):
        if self.embeddings is not None:
            joblib.dump(self.embeddings, self.embeddings_path)
            log.info(f"Embeddings cache saved in {self.embeddings_path.name}")

    def similarity_scores(self):
        # question 1 embeddings
        q1_embeddings = torch.from_numpy(
            np.stack(self.df["question1"].map(self.embeddings).values)
        )
        log.info("Mapping question 1 embeddings")
        # question 2 embeddings
        q2_embeddings = torch.from_numpy(
            np.stack(self.df["question2"].map(self.embeddings).values)
        )
        log.info("Mapping question 2 embeddings")
        # generating similarity scores
        similarity_scores = self.sbert.similarity_pairwise(q1_embeddings, q2_embeddings)
        self.df["sim_score"] = similarity_scores
        log.info("Generated similarity scores")

    def save_data(self):
        self.df.to_csv(self.data_path, index=False)
        log.info(f"Similarity scores saved to {self.data_path.name}")


if __name__ == "__main__":
    generator = GenerateEmbeddings()

    # generate embeddings pipeline
    generator.load_data()
    generator.sbert_init()
    generator.sbert_encode()
    generator.save_cache()
    generator.similarity_scores()
    generator.save_data()

    # generate embeddings complete
    log.info("GENERATE EMBEDDINGS PIPELINE COMPLETED SUCCESSFULLY")
    log.info("")
