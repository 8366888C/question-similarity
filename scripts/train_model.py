from logger import setup_logger
from pathlib import Path
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold,
)
import torch
import xgboost as xgb
import mlflow
from sklearn.metrics import (
    log_loss,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)

# setup logger
log = setup_logger(__name__, "train_model.log")


class ModelTraining:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / "data"
        self.data_path = self.data_dir / "data.csv"
        self.models_dir = self.base_dir / "app" / "models"
        self.xgboost_path = self.models_dir / "xgboost.json"
        self.df = None
        self.xgboost = None
        mlflow.xgboost.autolog(log_models=False)  # autolog mlflow runs
        self.model = None
        self.split = [None, None, None, None]
        self.random_search = None
        self.grid_search = None
        self.search_space = {
            "n_estimators": [500, 1000, 1500, 2000, 2500, 3000],
            "max_depth": [2, 3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9, 1],
            "gamma": [0, 0.1, 0.2, 0.3, 0.4],
            "reg_alpha": [0, 0.1, 0.15, 0.2, 0.25],
            "reg_lambda": [1.5, 1.75, 2, 2.25],
        }

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            log.info(f"Embedded data loaded from {self.data_path.name}")
        except Exception as e:
            log.error(f"Failed to load data: {e}")

    def data_split(self):
        # features and target class variable
        features = self.df.drop(
            columns=["id", "qid1", "qid2", "question1", "question2", "is_duplicate"]
        ).columns
        X = self.df[features]
        y = self.df["is_duplicate"]
        log.info("Features and target class variables set")
        # stratified split
        X = X.astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.split = [X_train, X_test, y_train, y_test]
        log.info("Stratified data split done")

    def random_space(self):
        # initialize the base model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.xgboost = xgb.XGBClassifier(
            tree_method="hist", device=device, random_state=42
        )
        log.info("Base xgboost model initialized")
        # setup random search
        self.random_search = RandomizedSearchCV(
            estimator=self.xgboost,
            param_distributions=self.search_space,
            n_iter=33,
            scoring="neg_log_loss",
            cv=StratifiedKFold(n_splits=3),
            random_state=42,
            n_jobs=1,
            verbose=2,
        )
        # run random search
        log.info("Random search running on search_space ...")
        self.random_search.fit(self.split[0], self.split[2])

    def grid_space(self):
        # define search space for grid
        random_best = self.random_search.best_params_
        grid_params = {
            "n_estimators": [
                random_best["n_estimators"] - 100,
                random_best["n_estimators"],
                random_best["n_estimators"] + 100,
            ],
            "max_depth": [
                random_best["max_depth"] - 1,
                random_best["max_depth"],
                random_best["max_depth"] + 1,
            ],
            "learning_rate": [random_best["learning_rate"]],
            "subsample": [random_best["subsample"]],
            "colsample_bytree": [random_best["colsample_bytree"]],
            "gamma": [random_best["gamma"]],
            "reg_alpha": [random_best["reg_alpha"]],
            "reg_lambda": [random_best["reg_lambda"]],
        }
        log.info("grid_params set based on random search's best params")
        # setup grid search
        self.grid_search = GridSearchCV(
            estimator=self.xgboost,
            param_grid=grid_params,
            scoring="neg_log_loss",
            cv=StratifiedKFold(n_splits=3),
            n_jobs=1,
            error_score="raise",
            verbose=2,
        )
        # run grid search
        log.info("Grid search running on grid_params ...")
        self.grid_search.fit(self.split[0], self.split[2])

    def model_evaluation(self):
        self.model = self.grid_search.best_estimator_
        y_pred = self.model.predict(self.split[1])
        y_proba = self.model.predict_proba(self.split[1])[:, 1]
        metrics = {
            "log_loss": log_loss(self.split[3], y_proba),
            "f1_score": f1_score(self.split[3], y_pred),
            "roc_auc": roc_auc_score(self.split[3], y_proba),
            "precision": precision_score(self.split[3], y_pred),
            "recall": recall_score(self.split[3], y_pred),
        }
        # log the final metrics and model to the active parent run
        mlflow.log_metrics(metrics)
        input_example = self.split[1].head(5)
        mlflow.xgboost.log_model(self.model, name="model", input_example=input_example)
        log.info(f"MLflow run complete. Metrics: {metrics}")

    def save_model(self):
        self.model.save_model(self.xgboost_path)
        log.info(f"Model saved locally to {self.xgboost_path.name}")


if __name__ == "__main__":
    trainer = ModelTraining()

    # model training pipeline
    trainer.load_data()
    trainer.data_split()

    mlflow.set_experiment("question_similarity")
    with mlflow.start_run(run_name="xgboost_pipeline"):
        trainer.random_space()
        trainer.grid_space()
        trainer.model_evaluation()

    trainer.save_model()

    # model training complete
    log.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    log.info("")
