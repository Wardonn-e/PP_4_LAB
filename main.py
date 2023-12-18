import pandas as pd


def read_csv_in_data_frame(path: str) -> pd.core.frame.DataFrame:
    df_csv = pd.read_csv(path)
    texts = []
    for absolute_path, rating in zip(df_csv['absolute_path'], df_csv['rating']):
        with open(absolute_path, 'r', encoding='utf-8') as file:
            text = file.read()
            texts.append((text, rating))

    df = pd.DataFrame(texts, columns=['review', 'rating'])
    return df


def delete_none(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    print(df.isnull().sum())
    df.dropna()
    return df




if __name__ == '__main__':
    path = r"/Users/wardonne/Desktop/Lab_python/PP_3_Lab/annotation1.csv"
    df = read_csv_in_data_frame(path)
    print(df)
