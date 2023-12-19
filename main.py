import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from collections import Counter, defaultdict
import re
from pymorphy3 import MorphAnalyzer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_ru = set(stopwords.words('russian'))
patterns = "[«»A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
morph = MorphAnalyzer()


def read_csv_to_dataframe(path: str) -> pd.DataFrame:
    """
    Чтение данных из CSV-файла и преобразование их в DataFrame.

    Args:
    - path (str): Путь к CSV-файлу.

    Returns:
    pd.DataFrame: DataFrame с колонками 'review' и 'rating'.
    """
    df_csv = pd.read_csv(path)
    texts = [(open(absolute_path, 'r', encoding='utf-8').read(), rating)
             for absolute_path, rating in zip(df_csv['absolute_path'], df_csv['rating'])]
    return pd.DataFrame(texts, columns=['review', 'rating'])


def drop_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаление строк с отсутствующими значениями.

    Args:
    - df (pd.DataFrame): Исходный DataFrame.

    Returns:
    pd.DataFrame: DataFrame без отсутствующих значений.
    """
    print(df.isnull().sum())
    return df.dropna()


def add_word_count_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавление колонки 'word_count' с количеством слов в каждом отзыве.

    Args:
    - df (pd.DataFrame): Исходный DataFrame.

    Returns:
    pd.DataFrame: DataFrame с новой колонкой 'word_count'.
    """
    df['word_count'] = df['review'].apply(lambda text: len(text.split()))
    return df


def filter_by_word_count(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Фильтрация DataFrame по количеству слов в отзыве.

    Args:
    - df (pd.DataFrame): Исходный DataFrame.
    - threshold (int): Пороговое значение количества слов.

    Returns:
    pd.DataFrame: Отфильтрованный DataFrame.
    """
    return df[df['word_count'] >= threshold]


def filter_by_rating(df: pd.DataFrame, rating_value: int) -> pd.DataFrame:
    """
    Фильтрация DataFrame по рейтингу.

    Args:
    - df (pd.DataFrame): Исходный DataFrame.
    - rating_value (int): Значение рейтинга для фильтрации.

    Returns:
    pd.DataFrame: Отфильтрованный DataFrame.
    """
    return df[df['rating'] == rating_value]


def lemmatize_text(review: str) -> List[str]:
    """
    Лемматизация текста.

    Args:
    - review (str): Исходный текст.

    Returns:
    List[str]: Список лемм.
    """
    review = re.sub(patterns, ' ', review)
    tokens = review.lower().split()
    preprocessed_text = [morph.parse(token)[0].normal_form for token in tokens if token not in stopwords_ru]
    return preprocessed_text


def get_most_common_words(df: pd.DataFrame, rating: int, top_n: int = 10) -> List[tuple[str, int]]:
    """
    Определение самых популярных слов для заданного рейтинга.

    Args:
    - df (pd.DataFrame): Исходный DataFrame.
    - rating (int): Значение рейтинга.
    - top_n (int): Количество самых популярных слов для вывода.

    Returns:
    List[tuple[str, int]]: Список кортежей (слово, частота).
    """
    data = df[df['rating'] == rating]['review'].apply(lemmatize_text)
    words = Counter(word for txt in data for word in txt)
    return words.most_common(top_n)


def plot_word_histogram(hist_list: List[tuple[str, int]]) -> None:
    """
    Построение гистограммы для списка слов.

    Args:
    - hist_list (List[tuple[str, int]]): Список кортежей (слово, частота).

    Returns:
    None
    """
    words, count = zip(*hist_list)
    plt.bar(words, count)
    plt.ylabel('Количество')
    plt.title('Гистограмма самых популярных слов')
    plt.show()


if __name__ == '__main__':
    path = r"/Users/wardonne/Desktop/Lab_python/PP_3_Lab/annotation1.csv"
    df = read_csv_to_dataframe(path)
    df = drop_null_rows(df)
    df = add_word_count_column(df)
    plot_word_histogram(get_most_common_words(df, 1))

    print(df.loc[df['word_count'] == 6])
    print(df.head())
    print('----')
    print(df.describe())
    filtered_by_word_count = filter_by_word_count(df, 1600)
    print(filtered_by_word_count)
    filtered_by_rating = filter_by_rating(df, 3)
    print(filtered_by_rating)
    print('------')
    rating_stats = df.groupby('rating').agg({'word_count': ['min', 'max', 'mean']})
    print(rating_stats)
    word_freq = Counter(word for tokens in df[df['rating'] == 1]['review'].apply(lemmatize_text) for word in tokens)
    print(len(word_freq))
    print(word_freq.most_common(10))
