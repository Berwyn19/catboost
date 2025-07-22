import pandas as pd
import numpy as np
import argparse
import joblib
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
import yaml
import os


def load_config(config_path):
    with open(config_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def load_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop("income", axis=1)
    y = df["income"]
    return X, y


def train_model(X_train, y_train, X_val, y_val, config, save_path):
    clf = CatBoostClassifier(
        learning_rate=config["params"]["learning_rate"],
        iterations=config["params"]["iterations"],
        depth=config["params"]["depth"],
        l2_leaf_reg=config["params"]["l2_leaf_reg"],
        random_seed=42,
        verbose=True,
        cat_features=config["cat_features"],
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    joblib.dump(clf, save_path)
    print(f"Model saved to {save_path}")
    return clf


def evaluate_model(X_test, y_test, model_path):
    clf = joblib.load(model_path)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(f"Accuracy Score: {accuracy}")

    classification_rep = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{classification_rep}")

    return accuracy, classification_rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="Mode: train or eval")
    parser.add_argument("--train", help="Path to training CSV")
    parser.add_argument("--eval", help="Path to validation CSV (for training)")
    parser.add_argument("--test", help="Path to test CSV (for eval)")
    parser.add_argument("--save", help="Path to save/load model")
    parser.add_argument("--config", help="Path to config YAML file")
    args = parser.parse_args()

    if args.mode == "train":
        if not all([args.train, args.eval, args.save, args.config]):
            raise ValueError("Training requires --train, --eval, --save, and --config arguments.")
        X_train, y_train = load_data(args.train)
        X_val, y_val = load_data(args.eval)
        config = load_config(args.config)
        train_model(X_train, y_train, X_val, y_val, config, args.save)

    elif args.mode == "eval":
        if not all([args.test, args.save]):
            raise ValueError("Evaluation requires --test and --save arguments (save is path to model).")
        X_test, y_test = load_data(args.test)
        evaluate_model(X_test, y_test, args.save)


if __name__ == "__main__":
    main()
