import numpy as np
import pytest
import os
import pandas as pd
import math

# - функцию обучения (что оно корректно запускается на корректных данных,
# корректно реагирует на некорректные данные и что на выходе появляется sklearn-подобный файлик)
# - лучше запустить на каких-то небольших фиксированных данных и на маленькое количество итераций.
# после того, как тест отработает, файлик с моделью нужно удалить.

from click.testing import CliRunner
from main import train, predict, splt, preprocess_text


# проверим, что файл с моделью появился
@pytest.mark.parametrize("data, model, test, split", [
    ('../data/singapore_airlines_reviews.csv', 'm.pkl', None, None),
    ('reviews.csv', 'm.pkl', None, None)
])
def test_train_with_valid_data(data, model, test, split):
    runner = CliRunner()
    result = runner.invoke(train, ['--data', data, '--model', model, '--test', test, '--split', split])

    assert result.exit_code == 0
    assert os.path.isfile(model)
    os.remove(model)


# проверим, что если подать несущетсвующий файл с data или test, то будет ошибка
@pytest.mark.parametrize("data, model, test, split", [
    ('../data/fake_reviews.csv', 'm.pkl', None, None),
    ('fake_reviews.csv', 'm.pkl', None, None),
    ('../data/singapore_airlines_reviews.csv', 'm.pkl', 'fake_test.csv', None),
])
def test_train_with_invalid_data(data, model, test, split):
    runner = CliRunner()
    result = runner.invoke(train, ['--data', data, '--model', model, '--test', test, '--split', split],
                           standalone_mode=False)
    assert result.exception is not None
    assert isinstance(result.exception, FileNotFoundError)
    assert not os.path.exists('m.pkl')


# проверим, что если подать и test и split в аргументах, то будет ошибка
@pytest.mark.parametrize("data, model, test, split", [
    ('../data/singapore_airlines_reviews.csv', 'm.pkl', 'fake_test.csv', 0.2),
])
def test_train_with_test_and_split(data, model, test, split):
    runner = CliRunner()
    result = runner.invoke(train, ['--data', data, '--model', model, '--test', test, '--split', split],
                           standalone_mode=False)
    assert result.exception is not None
    assert isinstance(result.exception, ValueError)
    assert not os.path.exists('m.pkl')


# - функцию предикта, и что на выходе ничего не появляется, кроме того, что ожидается


# проверим, что вывод такой, как ожидается
@pytest.mark.parametrize("data, model, cnt", [
    ('reviews.csv', 'model.pkl', 5), ('worst flight', 'model.pkl', 1)
])
def test_predict_with_valid_data(data, model, cnt):
    runner = CliRunner()
    result = runner.invoke(predict, ['--data', data, '--model', model])

    output = result.output.splitlines()
    assert "predicted rating is" in output
    output.remove("predicted rating is")
    assert len(output) == cnt
    for line in output:
        rating = int(line)
        assert 1 <= rating <= 5


# проверим, что если в аргументах подать несуществующую модель, то будет ошибка
@pytest.mark.parametrize("data, model", [
    ('reviews.csv', 'model_model.pkl')
])
def test_predict_with_invalid_data(data, model):
    runner = CliRunner()
    result = runner.invoke(predict, ['--data', data, '--model', model])

    assert result.exception is not None
    assert isinstance(result.exception, FileNotFoundError)


# - функцию разбиения данных - что это происходит в правильной пропорции и что происходит перемешивание.
# не забываем, что можно контроллировать random seed.

df = pd.DataFrame({'text': ['hi', 'hello', 'hello, world', 'hello, world!', 'hi, world', ' bye', 'goodbye',
                            'bye, world', 'bye, world']})


@pytest.mark.parametrize("data, random_seed, test_size", [
    (df, 42, 0.6), (df, 42, 0.2)
])
def test_splt(data, random_seed, test_size):
    if 'type' in data.columns:
        data = data.drop(columns=['type'])
    length = len(data['text'])
    result, res = splt(data, random_seed, test_size)
    assert len(res) == math.ceil(test_size * length) # проверяем, что происходит в правильной пропорции
    values = data['text'].values
    res_values = []
    for r in result.values:
        res_values.append(r[0])
    res_values = np.array(res_values)
    assert not np.array_equal(values[:len(result)], res_values) # проверяем, что данные перемешались
