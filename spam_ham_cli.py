import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data, text_column, vectorizer):
    X = vectorizer.fit_transform(data[text_column])
    y = data['label_num']
    return X, y

def train_model(X_train, y_train):
    model = BernoulliNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

def predict(model, vectorizer, input_texts):
    X = vectorizer.transform(input_texts)
    return model.predict(X)

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def main():
    parser = argparse.ArgumentParser(description='Spam/Ham Classification CLI')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    parser.add_argument('text_column', type=str, help='Name of the text column')
    parser.add_argument('label_column', type=str, help='Name of the label column')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', type=str, help='Predict using the model')
    args = parser.parse_args()

    vectorizer = CountVectorizer(binary=True)

    if args.train:
        data = load_data(args.file_path)
        X, y = preprocess_data(data, args.text_column, vectorizer)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
        print(f"Trained Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        save_model(model, 'bernoulli_nb_model.pkl')
        save_model(vectorizer, 'vectorizer.pkl')

    if args.predict:
        model = load_model('bernoulli_nb_model.pkl')
        vectorizer = load_model('vectorizer.pkl')
        with open(args.predict, 'r') as file:
            input_texts = [line.strip() for line in file.readlines()]
        predictions = predict(model, vectorizer, input_texts)
        for text, prediction in zip(input_texts, predictions):
            print(f"Text: {text}\nPrediction: {'Spam' if prediction == 1 else 'Ham'}\n")

if __name__ == "__main__":
    main()
    
#How to run the CLI
# command 1
# python spam_ham_cli.py spam_ham_dataset.csv text label_num --train
# command 2
# python spam_ham_cli.py spam_ham_dataset.csv text label_num --predict input_for_prediction.txt