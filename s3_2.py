import cv2        # Імпортуємо бібліотеку OpenCV для роботи із зображеннями та відео
import os         # Імпортуємо модуль os для роботи з файловою системою

class GrayContrastProcessor:   # Оголошуємо клас для обробки відео у відтінках сірого та підсилення контрасту
    def __init__(self, video_path="Autopilot.mp4", out_dir="gray_contrast_results", show=True, save_frames=False, max_frames=100, thumb_width=400, clahe_clip=2.0, clahe_grid=8):
        self.video_path = video_path            # Шлях до відеофайлу
        self.out_dir = out_dir                  # Папка для збереження результатів
        self.show = show                        # Показувати результати на екрані
        self.save_frames = save_frames          # Зберігати кадри чи ні
        self.max_frames = max_frames            # Максимальна кількість кадрів для обробки
        self.thumb_width = thumb_width          # Ширина міні-зображення
        self.clahe_clip = clahe_clip            # Максимальне значення контрасту для CLAHE
        self.clahe_grid = clahe_grid            # Розмір сітки для CLAHE

        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)    # Створюємо директорію для збереження, якщо треба

        # Підготовка CLAHE (об'єкт для адаптивного підсилення контрасту)
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))

    def process(self):
        print(f"Пробую відкрити відео: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)         # Відкриваємо відеофайл
        if not cap.isOpened():                         # Перевіряємо, чи вдалося відкрити відео
            print(f"[ОШИБКА] Не вдалося відкрити відеофайл: {self.video_path}")
            return

        frame_count = 0                                # Лічильник кадрів
        print("Старт обробки відео...")
        while True:
            ret, frame = cap.read()                    # Зчитуємо наступний кадр з відео
            if not ret or frame_count >= self.max_frames:   # Якщо кадрів більше немає або досягнуто ліміт
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break

            # 1. Перетворення кадра у відтінки сірого (GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 2. Підсилення контрасту методом CLAHE
            contrast = self.clahe.apply(gray)

            # 3. Створюємо результати для показу (оригінал, сірий, сірий+контраст)
            results = {
                "original": frame,
                "gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),              # Перетворюємо назад у BGR для відображення
                "gray_contrast": cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)  # Те саме для кадра з контрастом
            }

            if self.show:
                self.display_results(results)            # Відображаємо мозаїку з результатів

            if self.save_frames:
                self.save_results(results, frame_count)  # Зберігаємо результати у файли

            if cv2.waitKey(20) & 0xFF == ord('q'):      # Якщо натиснута 'q' — вихід
                print("Вихід за запитом користувача (q)")
                break

            frame_count += 1                            # Збільшуємо лічильник кадрів

        cap.release()                                   # Звільняємо відеоресурси
        cv2.destroyAllWindows()                         # Закриваємо всі вікна OpenCV
        print(f"Оброблено кадрів: {frame_count}")

    def display_results(self, results):
        # Масштабуємо всі картинки до потрібної ширини для мозаїки
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in ["original", "gray", "gray_contrast"]]
        mosaic = cv2.hconcat(thumbs)                    # З'єднуємо всі зображення в один ряд (мозаїка)
        cv2.imshow('GRAY+Контраст (CLAHE)', mosaic)     # Показуємо вікно з мозаїкою

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]                             # Беремо розміри зображення
        ratio = width / w                                # Розраховуємо масштабний коефіцієнт
        new_dim = (width, int(h * ratio))                # Нові розміри (ширина, висота)
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)   # Масштабуємо зображення

    def save_results(self, results, frame_idx):
        for name, img in results.items():                # Проходимо по всіх результатах (оригінал, сірий, сірий+контраст)
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")    # Формуємо шлях до файлу
            cv2.imwrite(filename, img)                   # Зберігаємо зображення

if __name__ == '__main__':      # Головна точка входу у програму
    print("==== Старт GRAY+Контраст ====")
    processor = GrayContrastProcessor(
        video_path="Autopilot.mp4",         # Шлях до відеофайлу
        out_dir="gray_contrast_results",    # Папка для збереження
        show=True,                         # Показувати результати
        save_frames=False,                 # Не зберігати окремі кадри
        max_frames=100,                    # Максимум 100 кадрів
        thumb_width=400,                   # Ширина мініатюр
        clahe_clip=2.0,                    # Значення контрасту для CLAHE
        clahe_grid=8                       # Розмір сітки для CLAHE
    )
    processor.process()                    # Запускаємо обробку відео
    print("==== Кінець програми ====")
