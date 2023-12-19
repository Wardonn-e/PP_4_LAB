import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from collections import Counter
import re
from pymorphy3 import MorphAnalyzer
from collections import defaultdict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords_ru = set(stopwords.words('russian'))
patterns = "[«»A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
morph = MorphAnalyzer()


def read_csv_in_data_frame(path: str) -> pd.core.frame.DataFrame:
    """
    Чтение данных из CSV-файла и преобразование их в DataFrame.

    Args:
    - path (str): Путь к CSV-файлу.

    Returns:
    pd.core.frame.DataFrame: DataFrame с колонками 'review' и 'rating'.
    """
    df_csv = pd.read_csv(path)
    texts = []
    for absolute_path, rating in zip(df_csv['absolute_path'], df_csv['rating']):
        with open(absolute_path, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append((text, rating))

    df = pd.DataFrame(texts, columns=['review', 'rating'])
    return df


def delete_none(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Удаление строк с отсутствующими значениями.

    Args:
    - df (pd.core.frame.DataFrame): Исходный DataFrame.

    Returns:
    pd.core.frame.DataFrame: DataFrame без отсутствующих значений.
    """
    print(df.isnull().sum())
    df.dropna()
    return df


def count_word(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Добавление колонки 'count_word' с количеством слов в каждом отзыве.

    Args:
    - df (pd.core.frame.DataFrame): Исходный DataFrame.

    Returns:
    pd.core.frame.DataFrame: DataFrame с новой колонкой 'count_word'.
    """
    df['count_word'] = df['review'].apply(lambda word: len(word.split()))
    return df


def filter_by_words(df: pd.core.frame.DataFrame, count_words: int) -> pd.core.frame.DataFrame:
    """
    Фильтрация DataFrame по количеству слов в отзыве.

    Args:
    - df (pd.core.frame.DataFrame): Исходный DataFrame.
    - count_words (int): Пороговое значение количества слов.

    Returns:
    pd.core.frame.DataFrame: Отфильтрованный DataFrame.
    """
    return df[df.count_word >= count_words]


def filter_by_rating(df: pd.core.frame.DataFrame, count_rating: int) -> pd.core.frame.DataFrame:
    """
    Фильтрация DataFrame по рейтингу.

    Args:
    - df (pd.core.frame.DataFrame): Исходный DataFrame.
    - count_rating (int): Значение рейтинга для фильтрации.

    Returns:
    pd.core.frame.DataFrame: Отфильтрованный DataFrame.
    """
    return df[df.rating == count_rating]


def lemmatize(review: str) -> List[str]:
    """
    Лемматизация текста.

    Args:
    - review (str): Исходный текст.

    Returns:
    List[str]: Список лемм.
    """
    review = re.sub(patterns, ' ', review)
    tokens = review.lower().split()
    preprocessed_text = []
    for token in tokens:
        lemma = morph.parse(token)[0].normal_form
        if lemma not in stopwords_ru:
            preprocessed_text.append(lemma)
    return preprocessed_text


def most_popular_words(df: pd.core.frame.DataFrame, rating: int) -> List[tuple[str, int]]:
    """
    Определение самых популярных слов для заданного рейтинга.

    Args:
    - df (pd.core.frame.DataFrame): Исходный DataFrame.
    - rating (int): Значение рейтинга.

    Returns:
    List[tuple[str, int]]: Список кортежей (слово, частота).
    """
    data = df[df['rating'] == rating]['review'].apply(lemmatize)
    words = Counter()
    for txt in data:
        words.update(txt)
    return words.most_common(10)


def graph_build(hist_list: List[tuple[str, int]]) -> None:
    """
    Построение гистограммы для списка слов.

    Args:
    - hist_list (List[tuple[str, int]]): Список кортежей (слово, частота).

    Returns:
    None
    """
    words = []
    count = []
    for i in range(len(hist_list)):
        words.append(hist_list[i][0])
        count.append(hist_list[i][1])

    fig, ax = plt.subplots()

    ax.bar(words, count)
    ax.set_ylabel('Количество')
    ax.set_title('Гистограмма самых популярных слов')
    plt.show()

if __name__ == '__main__':
    path = r"/Users/wardonne/Desktop/Lab_python/PP_3_Lab/annotation1.csv"
    df = read_csv_in_data_frame(path)
    df = count_word(df)
    graph_build(most_popular_words(df, 1))

    print(df.loc[df['count_word'] == 6])
    print(df.head())
    print('----')
    print(df.describe())
    ab = filter_by_words(df, 1600)
    print(ab)
    cd = filter_by_rating(df, 3)
    print(cd)
    print('------')
    a = df.groupby('rating').agg({'count_word': ['min', 'max', 'mean']})
    print(a)
    stats_rating = df['rating'].describe()
    str = df.loc[df['count_word'] == 8]
    data = df[df['rating'] == 1]['review'].apply(lemmatize)
    word_freq = defaultdict(int)
    for tokens in data.iloc[:]:
        for token in tokens:
            word_freq[token] += 1
    print(len(word_freq))
    print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])
