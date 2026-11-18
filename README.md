# PUSL3205 Practical Portfolio

An organized portfolio of practical laboratory activities for PUSL3205. Each lab folder contains its implementation, outputs, evidence, observations, and learning outcomes.

## Laboratory progress

| Lab | Activity | Status |
|---|---|---|
| 01 | Sentiment Analysis Using Traditional Machine Learning | Complete; implementation and run evidence included |

## Repository layout

```text
PUSL3205-lab-sheet-1/
  README.md
  .gitignore
  requirements.txt
  Lab-01-Machine-Learning/
    README.md
    sentiment_analysis.py
    evidence/
      confusion_matrix.svg
      frequent_words.svg
      model_metrics.csv
      model_metrics.json
      preprocessing_examples.csv
      run_output.txt
      test_predictions.csv
```

## Run Lab 1

Install the dependencies with `pip install -r requirements.txt`, then run `python Lab-01-Machine-Learning/sentiment_analysis.py`. The script downloads the dataset from UCI and saves the run outputs under the lab's `evidence/` folder. The downloaded archive is cached locally and ignored by Git.

## Dataset reference

Kotzias, D. (2015). *Sentiment Labelled Sentences*. UCI Machine Learning Repository. https://doi.org/10.24432/C57604

