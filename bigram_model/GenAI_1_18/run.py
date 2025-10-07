import math
from pathlib import Path
import nltk
from .model import BigramModel
from ..data_utils import load_and_prepare_data

OUTPUT_FILE = Path(__file__).resolve().parents[2] / "figures" / "genai_1_results.txt"
SEED_WORDS = "in the"


def run_task() -> None:
    """
    Функция для выполнения GenAI-1-18.
    """

    try:
        train_data, vocab = load_and_prepare_data()
    except ValueError as e:
        print(f"Error with preparing data: {e}")
    
    # --- Создание и обучение модели ---
    model = BigramModel(2)
    model.train(train_data, vocab)

    # --- Генерация 10 слов ---
    seed_words = SEED_WORDS.split()
    generated_words = model.generate_text(num_words=10, text_seed=seed_words)
    full_generated_sequence = seed_words + generated_words

    # --- Оценка вероятности предложения ---
    test_sentence_str = ("detectives with his picture in hand were on the trail of cal barco.")
    test_sentence_tokens = [word.lower() for word in nltk.word_tokenize(test_sentence_str)]

    log_prob = model.calculate_log_probability(test_sentence_tokens)
    try:
        prob = math.pow(2, log_prob)
    except Exception as e:
        prob = 0.0

    # --- Вывод и сохранение результатов ---
    generated_text_str = " ".join(full_generated_sequence)
    results_output = f"""============================================================
      Результаты работы биграммной языковой модели
============================================================

1. Генерация текста
------------------------------------------------------------
- Начальная фраза: "{SEED_WORDS}"
- Сгенерированная последовательность: "{generated_text_str}"

2. Оценка вероятности предложения
------------------------------------------------------------
- Тестовое предложение: 
  "{test_sentence_str}"

- Логарифмическая вероятность: {log_prob:.4f}
- Вероятность: {prob:.4e}

============================================================"""

    print(results_output)

    with open(OUTPUT_FILE, 'w', encoding="utf-8") as f:
        f.write(results_output)

    print(f"Results successfully saved to \"{OUTPUT_FILE}\"")

if __name__ == '__main__':
    run_task()