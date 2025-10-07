from pathlib import Path
import nltk
from ..GenAI_1_18.model import BigramModel
from ..data_utils import load_and_prepare_data
from .evaluation import calculate_perplexity

OUTPUT_FILE = Path(__file__).resolve().parents[2] / "figures" / "genai_2_results.txt"


def run_task(path_to_save_results=OUTPUT_FILE) -> None:
    """
    Функция для выполнения GenAI-2-18.
    """

    try:
        train_data, vocab = load_and_prepare_data()
    except ValueError as e:
        print(f"Error with preparing data: {e}")
        return

    # --- Создание и обучение моели ---
    model = BigramModel(2)
    model.train(train_data, vocab)

    # --- Подготовка данных для теста ---
    meaningful_sentence_str = "the price of crude oil has risen sharply"
    random_sentence_str = "colorless green ideas sleep furiously"

    meaningful_tokens = [word.lower() for word in nltk.word_tokenize(meaningful_sentence_str)]
    random_tokens = [word.lower() for word in nltk.word_tokenize(random_sentence_str)]

    # --- Вычисление перплексии ---
    perplexity_meaningful = calculate_perplexity(model, meaningful_tokens)
    perplexity_random = calculate_perplexity(model, random_tokens)

    # --- Вывод и сохранение результатов ---
    results_output = f"""============================================================
      Результаты оценки perplexity биграммной модели
============================================================

1. Осмысленное предложение
------------------------------------------------------------
- Предложение: "{meaningful_sentence_str}"
- Перплексия: {perplexity_meaningful:.4f}

2. Бессмысленное (но грамматичное) предложение
------------------------------------------------------------
- Предложение: "{random_sentence_str}"
- Перплексия: {perplexity_random:.4f}

============================================================"""

    print(results_output)

    try:
        with open(path_to_save_results, 'w', encoding="utf-8") as f:
            f.write(results_output)
            print(f"Results successfully saved to \"{path_to_save_results}\"\n")
    except FileExistsError or FileNotFoundError as e:
        print(f"Error writing result to file: {e}")


if __name__ == '__main__':
    run_task()
