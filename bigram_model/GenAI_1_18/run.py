import math
import nltk
from pathlib import Path
from .model import BigramModel
from ..data_utils import load_and_prepare_data

OUTPUT_FILE = Path(__file__).resolve().parents[2] / "results" / "genai_1_results.txt"

SEED_WORDS = "in the"
TEST_SENTENCE = "detectives with his picture in hand were on the trail of cal barco."
GENERATED_WORDS_COUNT = 10


def prepare_and_train_model():
    """Загрузка данных и обучение модели."""

    try:
        train_data, vocab = load_and_prepare_data()
    except ValueError as e:
        raise RuntimeError(f"Error while preparing data: {e}")

    model = BigramModel(2)
    model.train(train_data, vocab)
    return model


def format_results(seed_words, generated_text, test_sentence, log_prob, prob):
    """Формирует человекочитаемый отчёт."""

    lines = [
        "=" * 60,
        "  Результаты работы биграммной языковой модели",
        "=" * 60,
        "",
        "1. Генерация текста",
        "-" * 60,
        f'- Начальная фраза: "{seed_words}"',
        f'- Сгенерированная последовательность: "{generated_text}"',
        "",
        "2. Оценка вероятности предложения",
        "-" * 60,
        f'- Тестовое предложение: "{test_sentence}"',
        f"- Логарифмическая вероятность: {log_prob:.4f}",
        f"- Вероятность: {prob:.4e}",
        "",
        "=" * 60,
        ""
    ]
    return "\n".join(lines)


def run_task(path_to_save_results=OUTPUT_FILE) -> None:
    """Функция для выполнения GenAI-1-18."""

    # --- Подготовка модели ---
    model = prepare_and_train_model()

    # --- Генерация текста ---
    seed_tokens = SEED_WORDS.split()
    generated_words = model.generate_text(num_words=GENERATED_WORDS_COUNT, text_seed=seed_tokens)
    full_generated_sequence = seed_tokens + generated_words
    generated_text_str = " ".join(full_generated_sequence)

    # --- Оценка вероятности ---
    test_tokens = [w.lower() for w in nltk.word_tokenize(TEST_SENTENCE)]
    log_prob = model.calculate_log_probability(test_tokens)
    try:
        prob = math.pow(2, log_prob)
    except Exception:
        prob = 0.0

    # --- Формирование и вывод результатов ---
    results_output = format_results(SEED_WORDS, generated_text_str, TEST_SENTENCE, log_prob, prob)
    print(results_output)

    try:
        with open(path_to_save_results, "w", encoding="utf-8") as f:
            f.write(results_output)
            print(f'Results successfully saved to "{path_to_save_results}"\n')
    except (FileExistsError, FileNotFoundError) as e:
        print(f"Error writing result to file: {e}")


if __name__ == "__main__":
    run_task()
