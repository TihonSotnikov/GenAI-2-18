import nltk
from nltk.corpus import brown
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.vocabulary import Vocabulary
from typing import Tuple, Iterable


def ensure_nltk_data() -> None:
    """
    Проверяет и при отсутствии скачивает необходимые данные NLTK.
    """
    resources = {
        "corpora/brown": "brown",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"nltk Resource '{name}' not found. Downloading...")
            nltk.download(name)


def load_and_prepare_data(n_gram_order: int = 2) -> Tuple[Iterable, Vocabulary]:
    """
    Загружает корпус brown, токенизирует его и подготавливает для обучения n-gram модели.

    Parameters
    ----------
    category : str
        Категория текстов из корпуса brown.
    n_gram_order : int
        Порядок n (для n-граммной модели).

    Returns
    -------
    Tuple[Iterable, Vocabulary]
        Кортеж, содержащий подготовленные данные для обучения и объект словаря (Vocabulary).

    Raises
    ------
    ValueError
        Указанная категория не найдена в корпусе brown.
    """

    ensure_nltk_data()

    print("Loading the entire brown corpus...")

    text_sents = [[word.lower() for word in sent] for sent in brown.sents()]
    vocab = Vocabulary((word for sent in text_sents for word in sent), unk_cutoff=2)
    train_data, _ = padded_everygram_pipeline(n_gram_order, text_sents)

    print("Data preparation complete.")
    return train_data, vocab

load_and_prepare_data()