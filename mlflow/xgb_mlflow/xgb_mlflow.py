import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import mlflow
import mlflow.sklearn
import click



@click.command()
@click.option("--max-depth", default=3, type=int)
@click.option("--learning-rate", default=0.1, type=float)
@click.option("--n-estimators", default=512, type=int)
@click.option("--booster", default="gbtree", type=str)
@click.option("--subsample", default=1.0, type=float)
@click.option("--min-child-weight", default=1.0, type=float)
def train_xgb(max_depth, learning_rate, n_estimators, booster, subsample, min_child_weight):
    np.random.seed(0)
    # データの用意
    X,y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    with mlflow.start_run():
        xgbClassifier = xgb.XGBClassifier(  max_depth=max_depth,
                                            learning_rate=learning_rate,
                                            n_estimators=n_estimators,
                                            booster=booster,
                                            subsample=subsample,
                                            min_child_weight=min_child_weight)
        pipeline = make_pipeline(PolynomialFeatures(),xgbClassifier)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        # 評価値の計算
        accuracy = accuracy_score(y_test, pred)
        recall = recall_score(y_test, pred, average="weighted")
        precision = precision_score(y_test, pred, average="weighted")
        f1 = f1_score(y_test, pred, average="weighted")
        # パラメータの保存
        mlflow.log_param("method_name",xgbClassifier.__class__.__name__)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("booster", booster)
        mlflow.log_param("subsample", subsample)
        mlflow.log_param("min_child_weight", min_child_weight)
        # 評価値の保存
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1", f1)
        # モデルの保存
        mlflow.sklearn.log_model(pipeline, "model")

if __name__ == "__main__":
    train_xgb()
