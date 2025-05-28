import cv2        # Імпортуємо бібліотеку OpenCV для роботи з зображеннями і відео
import os         # Імпортуємо модуль os для роботи з файловою системою

class VideoColorTransformer:    # Оголошуємо клас для обробки відео з кольоровими перетвореннями
    COLOR_MODES = {
        "gray": cv2.COLOR_BGR2GRAY,     # Чорно-білий режим
        "hsv": cv2.COLOR_BGR2HSV,       # Перетворення у простір HSV
        "lab": cv2.COLOR_BGR2LAB,       # Перетворення у простір LAB
        "ycrcb": cv2.COLOR_BGR2YCrCb,   # Перетворення у простір YCrCb
        "rgb": cv2.COLOR_BGR2RGB,       # Перетворення у простір RGB
    }

    def __init__(self, video_path="Autopilot.mp4", out_dir="results", show=True, save_frames=False, max_frames=100, thumb_width=300):
        self.video_path = video_path          # Шлях до відеофайлу
        self.out_dir = out_dir                # Папка для збереження результатів
        self.show = show                      # Показувати результати чи ні
        self.save_frames = save_frames        # Зберігати кадри чи ні
        self.max_frames = max_frames          # Максимальна кількість кадрів для обробки
        self.thumb_width = thumb_width        # Ширина міні-зображень

        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)   # Створюємо директорію для збереження, якщо треба

    def process(self):
        print(f"Пробую відкрити відео: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)        # Відкриваємо відеофайл
        if not cap.isOpened():                        # Перевіряємо, чи вдалося відкрити відео
            print(f"[ОШИБКА] Не вдалося відкрити відеофайл: {self.video_path}")
            return

        frame_count = 0                               # Лічильник кадрів
        print("Старт обробки відео...")
        while True:
            ret, frame = cap.read()                   # Зчитуємо черговий кадр з відео
            if not ret or frame_count >= self.max_frames:      # Якщо кадрів немає або досягнуто ліміт
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break

            results = self.apply_color_transforms(frame)       # Застосовуємо кольорові перетворення до кадра

            if self.show:
                self.display_results(results)                  # Показуємо результати на екрані

            if self.save_frames:
                self.save_results(results, frame_count)        # Зберігаємо результати у файли

            if cv2.waitKey(20) & 0xFF == ord('q'):            # Перевіряємо, чи натиснута клавіша 'q'
                print("Вихід за запитом користувача (q)")
                break

            frame_count += 1                                  # Збільшуємо лічильник кадрів

        cap.release()                                         # Звільняємо відео-ресурси
        cv2.destroyAllWindows()                               # Закриваємо всі вікна OpenCV
        print(f"Оброблено кадрів: {frame_count}")

    def apply_color_transforms(self, frame):
        results = {"original": frame}             # Зберігаємо оригінальний кадр
        for mode, code in self.COLOR_MODES.items():      # Проходимо по кожному кольоровому режиму
            converted = cv2.cvtColor(frame, code)        # Перетворюємо кадр у заданий колірний простір
            if mode == "gray":
                results[mode] = cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)   # Повертаємо сірий кадр у формат BGR для відображення
            else:
                results[mode] = self._to_bgr(converted, mode)                 # Інші простори переводимо назад у BGR
        return results                          # Повертаємо словник з усіма перетвореннями

    @staticmethod
    def _to_bgr(image, mode):
        # Перетворюємо зображення з відповідного простору назад у BGR
        if mode == "hsv":
            return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif mode == "lab":
            return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        elif mode == "ycrcb":
            return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        elif mode == "rgb":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def display_results(self, results):
        titles = ["original", "gray", "hsv", "lab", "ycrcb", "rgb"]           # Всі режими для показу
        imgs = [results[k] for k in titles if k in results]                   # Вибираємо наявні картинки
        # Масштабуємо всі зображення до thumb_width
        thumbs = [self.resize_to_width(img, self.thumb_width) for img in imgs]
        # Розбиваємо на два рядки (по 3 зображення)
        row1 = cv2.hconcat(thumbs[:3])                                        # Перший ряд
        row2 = cv2.hconcat(thumbs[3:])                                        # Другий ряд
        mosaic = cv2.vconcat([row1, row2])                                    # Об’єднуємо в мозаїку
        cv2.imshow('Color Transformations', mosaic)                           # Показуємо мозаїку

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]                          # Визначаємо розміри зображення
        ratio = width / w                             # Розраховуємо масштабний коефіцієнт
        new_dim = (width, int(h * ratio))             # Нові розміри
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)   # Масштабуємо зображення

    def save_results(self, results, frame_idx):
        for name, img in results.items():           # Проходимо по всіх результатах
            filename = os.path.join(
                self.out_dir,                      # Папка для збереження
                f"{name}_{frame_idx:04d}.jpg"      # Ім'я файлу (тип_кадр.jpg)
            )
            cv2.imwrite(filename, img)             # Зберігаємо зображення

if __name__ == '__main__':     # Точка входу в програму
    print("==== Старт програми ====")
    transformer = VideoColorTransformer(
        video_path="Autopilot.mp4",    # Шлях до відеофайлу
        out_dir="results",             # Папка для збереження
        show=True,                     # Показувати результати
        save_frames=False,             # Не зберігати окремі кадри
        max_frames=100,                # Максимум 100 кадрів
        thumb_width=300                # Ширина міні-зображень
    )
    transformer.process()              # Запускаємо процес обробки
    print("==== Кінець програми ====")
