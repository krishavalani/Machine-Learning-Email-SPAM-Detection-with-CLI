
# Spam/Ham Classification CLI

## Overview
This Command Line Interface (CLI) application is designed to classify text messages as either "Spam" or "Ham" (non-Spam) using a Bernoulli Naive Bayes model. It allows for training the model on a dataset and predicting the classification of new messages.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- pickle

## Installation
Before running the application, ensure that you have Python installed on your system. You can then install the required packages using pip:
```bash
pip install pandas scikit-learn
```

## Usage

### Training the Model
To train the model, you need a CSV file with text data and labels. Run the following command:
```bash
python spam_ham_cli.py <file_path> <text_column> <label_column> --train
```
- `<file_path>`: Path to the CSV file containing the dataset.
- `<text_column>`: Name of the column containing the text data.
- `<label_column>`: Name of the column containing the label (1 for Spam, 0 for Ham).

### Predicting Using the Model
To predict new data, provide a text file with one message per line:
```bash
python spam_ham_cli.py <file_path> <text_column> <label_column> --predict <input_file>
```
- `<input_file>`: Path to the text file containing messages for prediction.

## Examples

### Command for Training
```bash
python spam_ham_cli.py spam_ham_dataset.csv text label_num --train
```

### Command for Prediction
```bash
python spam_ham_cli.py spam_ham_dataset.csv text label_num --predict input_for_prediction.txt
```

## Additional Information
- The model and vectorizer are saved after training and are loaded for prediction.
- Ensure the dataset and input files are formatted correctly for successful execution.
