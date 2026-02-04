import mlflow
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlops.log"),
        logging.StreamHandler()
    ]
)

logging.info("Starting model training process.....")

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("iris-rf-experiment")

logging.info("Loading dataset.....")
iris = load_iris()

logging.info("Splitting dataset.....")
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2
)

logging.info("Starting MLflow run.....")
with mlflow.start_run():
    logging.info("Training model.....")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    logging.info("Predicting.....")
    y_pred = model.predict(X_test)

    logging.info("Calculating accuracy.....")
    acc = accuracy_score(y_test, y_pred)

    logging.info("Logging metrics.....")
    mlflow.log_metric("accuracy", acc)

    logging.info(f"Model accuracy: {acc}")
