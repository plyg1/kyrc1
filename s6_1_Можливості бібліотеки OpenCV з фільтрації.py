import cv2             # Імпортуємо OpenCV для обробки зображень і відео
import numpy as np     # Імпортуємо NumPy для роботи з матрицями та масивами
import os              # os — для роботи з файлами та папками

class FilteringDemo:    # Клас для демонстрації фільтрації зображень у відео
    def __init__(self, video_path="Autopilot.mp4", out_dir="filtering_results", show=True, save_frames=False, max_frames=60, thumb_width=280):
        self.video_path = video_path      # Шлях до відеофайлу
        self.out_dir = out_dir            # Папка для збереження результатів
        self.show = show                  # Чи показувати результати на екрані
        self.save_frames = save_frames    # Чи зберігати результати у файли
        self.max_frames = max_frames      # Максимальна кількість кадрів для обробки
        self.thumb_width = thumb_width    # Ширина мініатюри для мозаїки

        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)   # Створюємо папку, якщо треба зберігати

    def process(self):
        print(f"Пробую відкрити відео: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)        # Відкриваємо відеофайл
        if not cap.isOpened():                         # Якщо не вдалося відкрити файл
            print(f"[ОШИБКА] Не вдалося відкрити відеофайл: {self.video_path}")
            return

        frame_count = 0                                # Лічильник кадрів
        while True:
            ret, frame = cap.read()                    # Зчитуємо черговий кадр
            if not ret or frame_count >= self.max_frames:   # Кінець відео або досягнуто ліміт
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break

            results = self.apply_filters(frame)        # Застосовуємо всі фільтри

            if self.show:
                self.display_results(results)          # Показуємо результати на екрані

            if self.save_frames:
                self.save_results(results, frame_count) # Зберігаємо результати у файли

            if cv2.waitKey(20) & 0xFF == ord('q'):    # Вихід при натисканні 'q'
                print("Вихід за запитом користувача (q)")
                break

            frame_count += 1                          # Збільшуємо лічильник

        cap.release()                                 # Звільняємо відеоресурси
        cv2.destroyAllWindows()                       # Закриваємо всі вікна
        print(f"Оброблено кадрів: {frame_count}")

    def apply_filters(self, frame):
        # 1. Оригінал
        original = frame

        # 2. Gaussian Blur — розмиття по Гаусу
        gauss = cv2.GaussianBlur(frame, (11, 11), 0)

        # 3. Median Blur — медіанний фільтр
        median = cv2.medianBlur(frame, 9)

        # 4. Bilateral — бiлатеральний фільтр (зберігає краї)
        bilateral = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # 5. Box Filter (average) — просте середнє розмиття (бокс-фільтр)
        box = cv2.blur(frame, (9, 9))

        # 6. Sharpening (custom kernel) — фільтр різкості (саморобне ядро)
        kernel_sharpen = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened = cv2.filter2D(frame, -1, kernel_sharpen)

        return {
            "original": original,
            "gauss": gauss,
            "median": median,
            "bilateral": bilateral,
            "box": box,
            "sharpened": sharpened
        }

    def display_results(self, results):
        titles = ["original", "gauss", "median", "bilateral", "box", "sharpened"]  # Які картинки показуємо
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in titles if k in results]
        # Два рядки по три зображення
        row1 = self.safe_hconcat(thumbs[:3])
        row2 = self.safe_hconcat(thumbs[3:])
        # Вирівнювання по ширині
        if row1.shape[1] > row2.shape[1]:
            diff = row1.shape[1] - row2.shape[1]
            row2 = cv2.copyMakeBorder(row2, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
        elif row2.shape[1] > row1.shape[1]:
            diff = row2.shape[1] - row1.shape[1]
            row1 = cv2.copyMakeBorder(row1, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
        # Вирівнювання по висоті
        if row1.shape[0] != row2.shape[0]:
            maxh = max(row1.shape[0], row2.shape[0])
            if row1.shape[0] < maxh:
                row1 = cv2.copyMakeBorder(row1, 0, maxh-row1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            if row2.shape[0] < maxh:
                row2 = cv2.copyMakeBorder(row2, 0, maxh-row2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        mosaic = cv2.vconcat([row1, row2])  # Об'єднуємо два рядки у мозаїку
        cv2.imshow('OpenCV Filtering Demo', mosaic)   # Показуємо мозаїку у вікні

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]                      # Дізнаємось розміри
        ratio = width / w                         # Розраховуємо коефіцієнт масштабування
        new_dim = (width, int(h * ratio))         # Нові розміри
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)  # Масштабуємо зображення

    @staticmethod
    def safe_hconcat(img_list):
        """Горизонтальна конкатенація з вирівнюванням по висоті."""
        img_list = [img for img in img_list if img is not None]
        if not img_list:
            return None
        max_height = max(img.shape[0] for img in img_list)   # Максимальна висота серед усіх
        safe_imgs = []
        for img in img_list:
            if img.shape[0] < max_height:                    # Додаємо чорний бордер знизу, якщо треба
                diff = max_height - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            safe_imgs.append(img)
        return cv2.hconcat(safe_imgs)                        # Об'єднуємо горизонтально

    def save_results(self, results, frame_idx):
        for name, img in results.items():                             # Зберігаємо всі зображення для кадру
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, img)                                # Записуємо у файл

if __name__ == '__main__':    # Головна точка входу у програму
    print("==== Старт OpenCV Filtering Demo ====")
    demo = FilteringDemo(
        video_path="Autopilot.mp4",        # Шлях до відеофайлу
        out_dir="filtering_results",       # Папка для збереження
        show=True,                        # Показувати на екрані
        save_frames=False,                # Не зберігати файли
        max_frames=60,                    # Максимум 60 кадрів
        thumb_width=280                   # Ширина мініатюри для мозаїки
    )
    demo.process()                        # Запускаємо обробку відео
    print("==== Кінець програми ====")
