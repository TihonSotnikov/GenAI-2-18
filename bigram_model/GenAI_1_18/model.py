from nltk.lm import Lidstone
from nltk.lm.vocabulary import Vocabulary
from typing import List, Iterable


class BigramModel:
    """
    Класс-обертка для биграммной статистической языковой модели на основе NLTK.
    """

    def __init__(self, n_gram_order: int = 2, gamma: float = 0.1) -> None:
        """
        Инициализирует модель с указанным порядком n-грамм.

        Parameters
        ----------
        n_gram_order : int
            Порядок n для n-граммной модели.

        Raises
        ------
        ValueError
            Если подан порядок n < 2.
        """

        if n_gram_order < 2:
            raise ValueError("N-gram order must be at least 2 for this model.")

        self.order = n_gram_order
        self.model = Lidstone(order=self.order, gamma=gamma)
        self.trained = False

    def train(self, train_data: Iterable, vocab: Vocabulary) -> None:
        """
        Обучает модель на предоставленных данных.

        Parameters
        ----------
        train_data : Iterable
            Подготовленные n-граммы для обучения (обычно генератор).
        vocab : nltk.lm.vocabulary.Vocabulary
            Объект словаря, созданный NLTK.

        Raises
        ------
        RuntimeError
            Если модель уже была обучена.
        """

        if self.trained:
            raise RuntimeError("Model is already trained. Create a new instance to retrain.")

        print("Training the model...\n")
        self.model.fit(train_data, vocab)
        self.trained = True
        print("Model training complete.\n")

    def generate_text(self, num_words: int, text_seed: List[str], filter_service_tokins: bool = True) -> List[str]:
        """
        Генерирует текст с помощью обученной модели.

        Parameters
        ----------
        num_words : int
            Количество слов для генерации.
        text_seed : list[str]
            Начальная последовательность слов. Длина должна быть >= (order - 1).
        filter_service_tokens : bool
            Производить ли фильтрацию системных токенов в сгенерированном тексте. 

        Returns
        -------
        list[str]
            Список сгенерированных токенов.

        Raises
        ------
        RuntimeError
            Если модель не обучена.
        ValueError
            Если `text_seed` содержит недостаточно токенов для контекста.
        """

        if not self.trained:
            raise RuntimeError("Model must be trained before text generation.")
        if len(text_seed) < self.order - 1:
            raise ValueError(f"text_seed must contain at least {self.order - 1} tokens for a model of order {self.order}.")

        print(f"Generating {num_words} words with seed: \"{' '.join(text_seed)}\"...\n")
        return [
            word for word in self.model.generate(num_words, text_seed=text_seed)
            if not filter_service_tokins or word not in ["<UNK>", "<s>", "</s>"]
        ]
        
    def calculate_log_probability(self, sentence_tokens: List[str]) -> float:
        """
        Вычисляет логарифм вероятности предложения по основанию 2.

        Parameters
        ----------
        sentence_tokens : list[str]
            Предложение в виде списка токенов.

        Returns
        -------
        float
            Значение log_2(P(sentence)).

        Raises
        ------
        RuntimeError
            Если модель не обучена.
        """

        if not self.trained:
            raise RuntimeError("Model must be trained before probability calculation.")

        return self.model.logscore(sentence_tokens)
