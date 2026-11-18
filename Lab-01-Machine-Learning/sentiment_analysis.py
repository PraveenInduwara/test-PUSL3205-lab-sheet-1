from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
DATA_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
DATA_FILES = (
    "amazon_cells_labelled.txt",
    "imdb_labelled.txt",
    "yelp_labelled.txt",
)


def download_dataset():
    data_dir = ROOT / "data"
    archive_path = data_dir / "sentiment_labelled_sentences.zip"
    if not archive_path.exists():
        response = requests.get(DATA_URL, timeout=60)
        response.raise_for_status()
        data_dir.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(response.content)
    return archive_path.read_bytes()


def load_dataset():
    rows = []
    archive_bytes = download_dataset()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        members = {Path(name).name.lower(): name for name in archive.namelist()}
        for filename in DATA_FILES:
            member = members.get(filename.lower())
            if member is None:
                raise FileNotFoundError(f"{filename} is missing from the UCI archive")
            text = archive.read(member).decode("utf-8-sig", errors="replace")
            pending = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                parts = line.rsplit("\t", 1)
                if len(parts) == 2 and parts[1].strip() in {"0", "1"}:
                    sentence = " ".join([*pending, parts[0].strip()]).strip()
                    label_text = parts[1].strip()
                    pending = []
                else:
                    pending.append(line.strip())
                    continue
                rows.append({
                    "text": sentence,
                    "label": int(label_text),
                    "source": filename.removesuffix("_labelled.txt"),
                })
            if pending:
                raise ValueError(f"Could not attach a sentiment label to every row in {filename}")
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("The dataset did not contain both expected sentiment labels")
    return frame


def clean_text(text):
    text = text.lower()
    text = text.replace("â€™", "'")
    text = re.sub(r"\bcan't\b", "cannot", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"\b([a-z]+)n't\b", r"\1 not", text)
    text = re.sub(r"\b([a-z]+)'re\b", r"\1 are", text)
    text = re.sub(r"\b([a-z]+)'ve\b", r"\1 have", text)
    text = re.sub(r"\b([a-z]+)'ll\b", r"\1 will", text)
    text = re.sub(r"\b([a-z]+)'m\b", r"\1 am", text)
    text = re.sub(r"\b([a-z]+)'d\b", r"\1 would", text)
    text = re.sub(
        r"\b(it|that|there|what|who|he|she|this|here|which|how|when|where)'s\b",
        r"\1 is",
        text,
    )
    text = re.sub(r"\b([a-z]+)'s\b", r"\1", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def top_words(frame, label, limit=10):
    excluded = (set(ENGLISH_STOP_WORDS) - {"no", "nor", "not"}) | {"s", "t"}
    words = " ".join(frame.loc[frame["label"] == label, "clean_text"]).split()
    counts = Counter(word for word in words if word not in excluded)
    return counts.most_common(limit)


def save_evidence(frame, y_test, predictions, metrics):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    examples = pd.concat(
        [frame.loc[frame["label"] == 1].head(5), frame.loc[frame["label"] == 0].head(5)]
    )
    examples.loc[:, ["text", "clean_text", "label", "source"]].to_csv(
        EVIDENCE / "preprocessing_examples.csv", index=False
    )

    frequent_rows = []
    for label, sentiment in ((0, "Negative"), (1, "Positive")):
        for word, count in top_words(frame, label):
            frequent_rows.append({"sentiment": sentiment, "word": word, "count": count})
    frequent_frame = pd.DataFrame(frequent_rows)
    frequent_frame.to_csv(EVIDENCE / "frequent_words.csv", index=False)

    pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predictions}).to_csv(
        EVIDENCE / "test_predictions.csv", index=False
    )

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics["confusion_matrix"] = matrix.tolist()
    metrics["true_negatives"] = int(matrix[0, 0])
    metrics["false_positives"] = int(matrix[0, 1])
    metrics["false_negatives"] = int(matrix[1, 0])
    metrics["true_positives"] = int(matrix[1, 1])
    pd.DataFrame([metrics]).to_csv(EVIDENCE / "model_metrics.csv", index=False)
    (EVIDENCE / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    matrix_frame = pd.DataFrame(
        matrix,
        index=["Actual negative", "Actual positive"],
        columns=["Predicted negative", "Predicted positive"],
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(matrix_frame, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Logistic Regression - Test Set Confusion Matrix")
    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("Actual sentiment")
    fig.tight_layout()
    fig.savefig(EVIDENCE / "confusion_matrix.png", dpi=180)
    fig.savefig(EVIDENCE / "confusion_matrix.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, sentiment, color) in zip(
        axes,
        ((0, "Negative", "#c44e52"), (1, "Positive", "#4c9f70")),
    ):
        pairs = top_words(frame, label)
        words = [word for word, _ in pairs][::-1]
        counts = [count for _, count in pairs][::-1]
        ax.barh(words, counts, color=color)
        ax.set_title(f"Frequent words - {sentiment}")
        ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(EVIDENCE / "frequent_words.png", dpi=180)
    fig.savefig(EVIDENCE / "frequent_words.svg")
    plt.close(fig)

    return matrix.tolist()


def main():
    frame = load_dataset()
    frame["clean_text"] = frame["text"].map(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        frame["clean_text"],
        frame["label"],
        test_size=0.2,
        random_state=42,
        stratify=frame["label"],
    )

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_bow, y_train)
    predictions = model.predict(X_test_bow)

    new_sentence = "The food was delicious and the service was excellent."
    new_prediction = int(model.predict(vectorizer.transform([clean_text(new_sentence)]))[0])
    metrics = {
        "dataset_rows": int(len(frame)),
        "positive_rows": int((frame["label"] == 1).sum()),
        "negative_rows": int((frame["label"] == 0).sum()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "bow_feature_count": int(len(vectorizer.get_feature_names_out())),
        "model": "LogisticRegression",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "new_sentence": new_sentence,
        "new_sentence_prediction": "Positive" if new_prediction == 1 else "Negative",
    }
    save_evidence(frame, y_test, predictions, metrics)

    output = [
        f"Dataset rows: {metrics['dataset_rows']}",
        f"Label counts: negative={metrics['negative_rows']}, positive={metrics['positive_rows']}",
        f"Train/test rows: {metrics['train_rows']}/{metrics['test_rows']}",
        f"Bag of Words features: {metrics['bow_feature_count']}",
        f"Model: {metrics['model']}",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1 score: {metrics['f1_score']:.4f}",
        f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}",
        f"New sentence: {new_sentence}",
        f"Prediction: {metrics['new_sentence_prediction']}",
        "",
        "Classification report:",
        classification_report(y_test, predictions, target_names=["Negative", "Positive"]),
    ]
    report = "\n".join(output)
    (EVIDENCE / "run_output.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

