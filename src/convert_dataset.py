import os
import cv2
from pathlib import Path
import shutil
import random

# Подготавливает grayscale-фрагменты 128x128 из исходных изображений.
# Внимание: при каждом запуске папка результата удаляется и создаётся заново.
def convert_dataset():
    # Счётчики ограничивают общее количество фрагментов и формируют имена групп.
    count_img = 0
    count_group = 0

    PARENT_PATH = (Path(__file__).resolve().parent).parent

    input_folder = PARENT_PATH / "data" / "dataset" / "DIV2K_train_HR"
    output_folder = PARENT_PATH / "data" / "dataset" / "convertered_dataset"

    # Удаляем предыдущий результат конвертации, чтобы не смешивать его с новым.
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    # Ширина и высота каждого сохраняемого фрагмента в пикселях.
    patch_size = 128

    
    list_filles = os.listdir(input_folder)
    # Случайный порядок означает, что состав групп может отличаться между запусками.
    random.shuffle(list_filles)
    
    # Читаем исходники и останавливаемся после создания 40 000 фрагментов.
    for filename in list_filles:
        if count_img >= 40000:
            break

        print(f"\rФайл: {filename} Уже есть картинок 128x128: {count_img}")

        in_path = os.path.join(input_folder, filename)
        
        # UNCHANGED сохраняет исходную глубину и число каналов.
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Пропускаем файлы, которые OpenCV не смог прочитать.
            continue
            
        # Приводим цветные исходники к одному каналу яркости.
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # Если RGBA с альфа-каналом
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:  # Если обычный RGB/BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # После преобразования в grayscale матрица содержит только высоту и ширину.
        height, width = img.shape[:2]
        
        # Координаты исходного файла сохраняются в имени каждого фрагмента.
        name_without_ext = os.path.splitext(filename)[0]
        
        # Шаг равен размеру фрагмента, поэтому фрагменты не перекрываются.
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                
                # Правая и нижняя границы текущего фрагмента.
                y_end = y + patch_size
                x_end = x + patch_size
                
                # Неполные краевые фрагменты не дополняются, а пропускаются.
                if y_end > height or x_end > width:
                    continue
                    
                # Вырезаем участок изображения размером 128x128.
                patch = img[y:y_end, x:x_end]
                
                # По имени можно восстановить исходник и координаты фрагмента.
                patch_filename = f"{name_without_ext}_y{y}_x{x}.png"

                # Папка группы создаётся при сохранении её первого фрагмента.
                os.makedirs(output_folder / f"group_{count_group}", exist_ok=True)
                out_path = os.path.join(output_folder / f"group_{count_group}", patch_filename)
                
                # img уже grayscale, поэтому PNG сохраняется одноканальным.
                cv2.imwrite(out_path, patch)
                count_img += 1

                # Каждые четыре последовательных фрагмента получают один номер группы.
                # Одна группа может включать фрагменты из разных исходников.
                if (count_img % 4 == 0):
                    count_group += 1

# Раскомментируйте вызов, если нужно запустить конвертацию напрямую.
# if __name__ == "__main__":
#     convert_dataset()