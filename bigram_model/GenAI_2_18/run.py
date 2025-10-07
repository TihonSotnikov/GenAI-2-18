from pathlib import Path
from typing import List
import nltk
from ..GenAI_1_18.model import BigramModel
from ..data_utils import load_and_prepare_data
from .evaluation import calculate_perplexity

OUTPUT_FILE = Path(__file__).resolve().parents[2] / "results" / "genai_2_results.txt"

MEANINGFUL_SENTENCE = "the price of crude oil has risen sharply"
RANDOM_SENTENCE = "colorless green ideas sleep furiously"


def prepare_and_train_model() -> BigramModel:
    """Загрузка данных и обучение модели."""

    try:
        train_data, vocab = load_and_prepare_data()
    except ValueError as e:
        raise RuntimeError(f"Error while preparing data: {e}")

    model = BigramModel(2)
    model.train(train_data, vocab)
    return model


def format_results(
    meaningful: str,
    random: str,
    perplex_meaningful: float,
    perplex_random: float
) -> str:
    """Формирует человекочитаемый отчёт."""

    lines = [
        "=" * 60,
        "  Результаты оценки perplexity биграммной модели",
        "=" * 60,
        "",
        "1. Осмысленное предложение",
        "-" * 60,
        f'- Предложение: "{meaningful}"',
        f"- Перплексия: {perplex_meaningful:.4f}",
        "",
        "2. Бессмысленное (но грамматичное) предложение",
        "-" * 60,
        f'- Предложение: "{random}"',
        f"- Перплексия: {perplex_random:.4f}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def run_task(path_to_save_results: Path = OUTPUT_FILE) -> None:
    """Функция для выполнения GenAI-2-18."""

    # --- Подготовка модели ---
    model = prepare_and_train_model()

    # --- Подготовка данных ---
    meaningful_tokens: List[str] = [w.lower() for w in nltk.word_tokenize(MEANINGFUL_SENTENCE)]
    random_tokens: List[str] = [w.lower() for w in nltk.word_tokenize(RANDOM_SENTENCE)]

    # --- Вычисление перплексии ---
    print("Start measuring the perplexity of the model\n")

    perplexity_meaningful: float = calculate_perplexity(model, meaningful_tokens)
    perplexity_random: float = calculate_perplexity(model, random_tokens)

    print("Model perplexity measured\n")
    
    # --- Формирование и вывод результатов ---
    results_output: str = format_results(
        MEANINGFUL_SENTENCE,
        RANDOM_SENTENCE,
        perplexity_meaningful,
        perplexity_random
    )
    print(results_output)

    try:
        with open(path_to_save_results, "w", encoding="utf-8") as f:
            f.write(results_output)
            print(f'Results successfully saved to "{path_to_save_results}"\n')
    except (FileExistsError, FileNotFoundError) as e:
        print(f"Error writing result to file: {e}")


if __name__ == "__main__":
    run_task()
