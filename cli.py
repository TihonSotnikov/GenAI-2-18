import argparse
import sys
from pathlib import Path
from typing import Callable
from bigram_model.GenAI_1_18.run import run_task as run_task_1
from bigram_model.GenAI_2_18.run import run_task as run_task_2

TASKS = {
    "GenAI_1_18": run_task_1,
    "GenAI_2_18": run_task_2,
}

DEFAULT_TASK_KEY = "GenAI_2_18"


def parse_arguments() -> argparse.Namespace:
    """
    Парсит аргументы командной строки.

    Returns
    -------
    argparse.Namespace
        Объект, содержащий разобранные аргументы командной строки.
        - GenAI_1_18 (bool): Флаг для запуска задачи 1.
        - GenAI_2_18 (bool): Флаг для запуска задачи 2.
        - output (str | None): Путь для сохранения файла с результатами.
    """

    parser = argparse.ArgumentParser(
        description="A CLI utility for running GenAI project tasks."
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to the output file for saving results.\nIf not specified, the default path from the task script is used."
    )

    task_group = parser.add_mutually_exclusive_group(required=False)

    task_group.add_argument(
        "-1", "--GenAI-1-18",
        action="store_true",
        help="Execute task GenAI-1-18 (bigram model)."
    )

    task_group.add_argument(
        "-2", "--GenAI-2-18",
        action="store_true",
        help="Execute task GenAI-2-18."
    )

    return parser.parse_args()


def run_cli() -> None:
    """
    Основная точка входа для интерфейса командной строки.
    Определяет, какую задачу выполнить на основе переданных аргументов,
    и вызывает соответствующую функцию с необходимыми параметрами.

    Note
    ----
    - Если флаг задачи (`-1` или `-2`) не указан, выполняется задача по умолчанию (`GenAI_2_18`).
    - Если путь для сохранения (`-o` или `--output`) не указан, используется путь по умолчанию (сохранение в папку figures)
    - Если директория для указанного пути сохранения не существует, она будет создана автоматически.

    Examples
    --------
    1. Запуск задачи по умолчанию (GenAI-2-18) с сохранением в файл по умолчанию:
       $ python cli.py
    2. Явный запуск задачи GenAI-1-18 с сохранением в файл по умолчанию:
       $ python cli.py -1
       или
       $ python cli.py --GenAI-1-18
    3. Запуск задачи GenAI-2-18 с сохранением результата в указанный файл:
       $ python cli.py -2 -o ./results/my_genai2_report.txt
       или
       $ python cli.py --GenAI-2-18 --output ./results/my_genai2_report.txt
    """

    args = parse_arguments()

    if args.GenAI_1_18:
        task_key = "GenAI_1_18"
    else:
        task_key = "GenAI_2_18"

    task_to_run = TASKS[task_key]

    kwargs = {}
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs['path_to_save_results'] = output_path

    try:
        if not args.GenAI_1_18 and not args.GenAI_2_18:
            print(f"No task specified. Running the default task: {task_key}...")
        else:
            print(f"Executing task: {task_key}...")
        task_to_run(**kwargs)
        print(f"Task {task_key} completed successfully.")
        sys.exit(0)

    except Exception as e:
        print(f"\nAn error occurred while executing task {task_key}:", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run_cli()
