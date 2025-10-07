import math
from typing import List
from ..GenAI_1_18.model import BigramModel


def calculate_perplexity(model: BigramModel, sentence_tokens: List[str]) -> float:
    """
    Вычисляет перплексию для заданного предложения с использованием обученной модели.

    Parameters
    ----------
    model : BigramModel
        Обученный экземпляр биграммной модели.
    sentence_tokens : List[str]
        Предложение, представленное в виде списка токенов.

    Returns
    -------
    float
        Значение перплексии. Возвращает float('inf'), если вероятность равна нулю.
    """

    print("Start measuring the perplexity of the model\n")

    log_prob = model.calculate_log_probability(sentence_tokens)

    N = len(sentence_tokens)
    if N == 0:
        return 0.0

    perplexity = math.pow(2, (-1/N) * log_prob)

    print("Model perplexity measured\n")

    return perplexity
