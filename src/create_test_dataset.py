import random
from pathlib import Path

from PIL import Image


PARENT_PATH = Path(__file__).resolve().parent.parent
# Входы и выходы совпадают с расположением конвертера и обучающего скрипта.
SOURCE_FOLDER = PARENT_PATH / "data" / "dataset" / "DIV2K_train_HR"
TRAIN_FOLDER = PARENT_PATH / "data" / "dataset" / "convertered_dataset"
OUTPUT_FOLDER = PARENT_PATH / "data" / "dataset" / "test_dataset"

# Настройки небольшой выборки, которой удобно быстро проверить модель.
PATCH_SIZE = 128
TEST_GROUPS = 20
RANDOM_SEED = 42


def main():
    # Защищаем уже созданный набор: повторный запуск не удалит и не перезапишет его.
    if OUTPUT_FOLDER.exists():
        raise FileExistsError(
            f"Папка уже существует: {OUTPUT_FOLDER}. "
            "Чтобы не перезаписывать данные, удалите или переименуйте её вручную."
        )

    # Исключаем все исходные изображения, хотя бы один фрагмент которых
    # уже использовался при обучении.
    training_ids = {
        image.stem.split("_y", 1)[0]
        for image in TRAIN_FOLDER.glob("group_*/*.png")
    }
    if not training_ids:
        raise FileNotFoundError(
            f"Обучающие фрагменты не найдены: {TRAIN_FOLDER}. "
            "Сначала подготовьте обучающий набор."
        )

    # Один исходный файл целиком попадает только в train или test.
    source_files = [
        path
        for path in SOURCE_FOLDER.glob("*.png")
        if path.stem not in training_ids
    ]
    # Фиксированный seed делает выбор тестовых исходников повторяемым.
    random.Random(RANDOM_SEED).shuffle(source_files)

    selected_groups = []
    for source_path in source_files:
        try:
            with Image.open(source_path) as source_image:
                image = source_image.convert("L")
        except OSError as error:
            raise OSError(f"Не удалось прочитать изображение: {source_path}") from error

        # Берём первые четыре полных неперекрывающихся фрагмента 128x128.
        width, height = image.size
        patches = [
            (
                f"{source_path.stem}_y{y}_x{x}.png",
                image.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE)),
            )
            for y in range(0, height - PATCH_SIZE + 1, PATCH_SIZE)
            for x in range(0, width - PATCH_SIZE + 1, PATCH_SIZE)
        ][:4]

        if len(patches) == 4:
            selected_groups.append((source_path.stem, patches))
            if len(selected_groups) == TEST_GROUPS:
                break

    if len(selected_groups) < TEST_GROUPS:
        raise RuntimeError(
            f"Удалось подготовить только {len(selected_groups)} тестовых групп "
            f"из {TEST_GROUPS}. Проверьте набор исходных и обучающих изображений."
        )

    # Сохраняем формат group_N/имя_исходника_yY_xX.png,
    # чтобы ImageGroupsDataset мог читать тестовый набор без отдельного загрузчика.
    for group_index, (_, patches) in enumerate(selected_groups):
        group_folder = OUTPUT_FOLDER / f"group_{group_index}"
        group_folder.mkdir(parents=True)
        for filename, patch in patches:
            output_path = group_folder / filename
            patch.save(output_path)

    print(f"Создано тестовых групп: {len(selected_groups)}")
    print(f"Не использовано при обучении исходных файлов: {len(source_files)}")
    print(f"Тестовый набор сохранён в: {OUTPUT_FOLDER}")
    print("Каждая группа содержит 4 фрагмента одного ранее не использованного файла.")


if __name__ == "__main__":
    main()
