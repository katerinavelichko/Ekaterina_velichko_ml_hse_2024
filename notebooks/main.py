import click
import pandas as pd
import re
import joblib
import os
import warnings

warnings.filterwarnings(action='ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub('[().,?!-:]', '', text)
    text = text.replace('"', '')
    return text


def preprocess_data(reviews):
    reviews = reviews.drop(columns=['type'])
    reviews['title'] = reviews['title'].apply(preprocess_text)
    reviews['text'] = reviews['text'].apply(preprocess_text)
    reviews['title_and_text'] = reviews['title'] + ' ' + reviews['text']
    return reviews


def splt(reviews, random_seed, test_size):
    return train_test_split(reviews, random_state=random_seed, test_size=test_size)


@click.group()
def train_and_predict():
    pass


@click.command()
@click.option('--data', required=True, help='path to file with data')
@click.option('--test', required=False, help='path to file with test data', default=None)
@click.option('--split', required=False, type=float, help='test size', default=None)
@click.option('--model', required=True, help='name of file to save model in pickle format')
def train(data, model, test, split):
    if not os.path.exists(data):
        raise FileNotFoundError(f"{data} does not found")
    df = pd.read_csv(data)
    reviews = preprocess_data(df)
    bow = TfidfVectorizer()
    m = LogisticRegression()

    if test is not None and split is not None:
        raise ValueError("--test and --split options cannot be provided together")


    elif test is not None:
        if not os.path.exists(test):
            raise FileNotFoundError(f"'{test}' does not exist")
        else:
            test = pd.read_csv(test)
            test = preprocess_data(test)
            x_train = bow.fit_transform(reviews['title_and_text'])
            x_test = bow.transform(test['title_and_text'])
            y_train = reviews['rating']
            y_test = test['rating']

            m.fit(x_train, y_train)
            y_pred = m.predict(x_test)
            f1 = f1_score(y_pred, y_test, average="weighted")
            click.echo(f"f1 result is {f1}")
            click.echo(f"{classification_report(y_test, y_pred)}")

    elif split is not None:
        if split < 0.0 or split > 1.0:
            raise ValueError("split must be > 0 and < 1 ")
        else:
            train, test = splt(reviews, 42, split)
            x_train = bow.fit_transform(train['title_and_text'])
            x_test = bow.transform(test['title_and_text'])
            y_train = train['rating']
            y_test = test['rating']

            m.fit(x_train, y_train)
            y_pred = m.predict(x_test)
            f1 = f1_score(y_pred, y_test, average="weighted")
            click.echo(f"f1 result is {f1}")
            click.echo(f"{classification_report(y_test, y_pred)}")

    else:
        x_train = bow.fit_transform(reviews['title_and_text'])
        y_train = reviews['rating']

        m.fit(x_train, y_train)

    joblib.dump((m, bow), model)
    click.echo(f"model saved in {model}")


@click.command()
@click.option('--data', required=True, help='path to file with data or text with review')
@click.option('--model', required=True, help='name of file with model in pickle format')
def predict(data, model):
    if not os.path.exists(model):
        raise FileNotFoundError(f"'{model}' does not exist")

    loaded_model, bow = joblib.load(model)

    if os.path.exists(data):
        df = pd.read_csv(data)
        reviews = preprocess_data(df)
        reviews = bow.transform(reviews['title_and_text'])
    else:
        reviews = bow.transform([data])

    prediction = loaded_model.predict(reviews)
    click.echo(f"predicted rating is")
    for p in prediction:
        click.echo(f"{p}")


train_and_predict.add_command(train)
train_and_predict.add_command(predict)

if __name__ == '__main__':
    train_and_predict()
