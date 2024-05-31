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


def splt(reviews, random_seed, test_size):
    return train_test_split(reviews, random_state=random_seed, test_size=test_size)

def clean(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\d+', '', s)
    common_symbols = {'=', '+', '-', '*', '/', '%', '>', '<', '&', '|', '^', '(', ')', '[', ']', '{', '}', ',',  ':'}
    s = ''.join([c for c in s if c not in common_symbols])
    return s

import click


@click.command()
@click.argument('input_text')
def main(input_text):
    loaded_model, bow = joblib.load('model_languages.pkl')

    if os.path.exists(input_text):
        with open(input_text, 'r') as file:
            text = file.read()
        text = clean(text)
        txt = bow.transform([text])
        prediction = loaded_model.predict(txt)
        click.echo(f"{prediction[0]}")

if __name__ == '__main__':
    main()
