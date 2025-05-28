import cv2           # Імпортуємо бібліотеку OpenCV для роботи з комп’ютерним зором
import numpy as np    # Імпортуємо NumPy для обробки масивів і матричних операцій
import os            # Імпортуємо os для роботи з файловою системою

class OpenCVOperationsDemo:    # Клас для демонстрації різних операцій OpenCV над відео
    def __init__(self, video_path="Autopilot.mp4", out_dir="opencv_ops_results", show=True, save_frames=False, max_frames=60, thumb_width=270):
        self.video_path = video_path       # Шлях до відеофайлу
        self.out_dir = out_dir             # Папка для збереження результатів
        self.show = show                   # Чи показувати результати на екрані
        self.save_frames = save_frames     # Чи зберігати результати у файли
        self.max_frames = max_frames       # Максимальна кількість кадрів для обробки
        self.thumb_width = thumb_width     # Ширина міні-зображень для мозаїки

        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)   # Створюємо папку, якщо треба зберігати результати

    def process(self):
        print(f"Пробую відкрити відео: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)        # Відкриваємо відеофайл
        if not cap.isOpened():                         # Якщо не вдалося відкрити файл
            print(f"[ОШИБКА] Не вдалося відкрити відеофайл: {self.video_path}")
            return

        frame_count = 0                                # Лічильник кадрів
        while True:
            ret, frame = cap.read()                    # Зчитуємо наступний кадр
            if not ret or frame_count >= self.max_frames:   # Кінець відео або досягнуто ліміт
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break

            results = self.apply_operations(frame)      # Застосовуємо всі OpenCV-операції

            if self.show:
                self.display_results(results)           # Показуємо результати на екрані

            if self.save_frames:
                self.save_results(results, frame_count) # Зберігаємо результати у файли

            if cv2.waitKey(20) & 0xFF == ord('q'):      # Вихід при натисканні 'q'
                print("Вихід за запитом користувача (q)")
                break

            frame_count += 1                            # Збільшуємо лічильник

        cap.release()                                   # Звільняємо ресурси відео
        cv2.destroyAllWindows()                         # Закриваємо всі вікна
        print(f"Оброблено кадрів: {frame_count}")

    def apply_operations(self, frame):
        # 1. Переводимо кадр у відтінки сірого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Бінаризація (порогове перетворення)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # 3. Морфологічні операції: ерозія (звуження) та дилатація (розширення)
        kernel = np.ones((5,5), np.uint8)          # Створюємо ядро (матриця 5х5 з одиниць)
        eroded = cv2.erode(binary, kernel, iterations=1)    # Ерозія (стиснення білих областей)
        dilated = cv2.dilate(binary, kernel, iterations=1)  # Дилатація (розширення білих областей)

        # 4. Розмиття зображення (Гаусовий Blur)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        # 5. Інверсія (негатив кадру)
        inverted = cv2.bitwise_not(frame)

        # 6. Виділення контурів (малюємо контури на копії кадру)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoured = frame.copy()
        cv2.drawContours(contoured, contours, -1, (0, 255, 0), 2)

        # 7. Анотація — малюємо фігури та текст
        annotated = frame.copy()
        cv2.rectangle(annotated, (30, 30), (200, 120), (0, 0, 255), 3)                 # Прямокутник
        cv2.circle(annotated, (100, 220), 40, (255, 0, 0), 3)                          # Коло
        cv2.putText(annotated, "OpenCV Demo", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2) # Текст

        # Функція для переведення зображення у 3-канальний формат (якщо потрібно)
        def to3(img):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

        # Повертаємо результати у словнику
        return {
            "original": frame,
            "gray": to3(gray),
            "binary": to3(binary),
            "eroded": to3(eroded),
            "dilated": to3(dilated),
            "blurred": blurred,
            "inverted": inverted,
            "contours": contoured,
            "annotated": annotated
        }

    def display_results(self, results):
        titles = ["original", "gray", "binary", "eroded", "dilated", "blurred", "inverted", "contours", "annotated"]  # Які картинки показуємо
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in titles if k in results]                 # Масштабуємо до однакової ширини
        # Мозаїка: перший ряд — 5 зображень, другий — 4
        row1 = self.safe_hconcat(thumbs[:5])
        row2 = self.safe_hconcat(thumbs[5:])
        # Вирівнюємо по ширині (додаємо чорний бордер якщо треба)
        if row1.shape[1] > row2.shape[1]:
            diff = row1.shape[1] - row2.shape[1]
            row2 = cv2.copyMakeBorder(row2, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
        elif row2.shape[1] > row1.shape[1]:
            diff = row2.shape[1] - row1.shape[1]
            row1 = cv2.copyMakeBorder(row1, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
        # Вирівнюємо по висоті
        if row1.shape[0] != row2.shape[0]:
            maxh = max(row1.shape[0], row2.shape[0])
            if row1.shape[0] < maxh:
                row1 = cv2.copyMakeBorder(row1, 0, maxh-row1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            if row2.shape[0] < maxh:
                row2 = cv2.copyMakeBorder(row2, 0, maxh-row2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        mosaic = cv2.vconcat([row1, row2])  # Вертикально склеюємо два рядки
        cv2.imshow('OpenCV Operations Demo', mosaic)   # Показуємо у вікні

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]                  # Визначаємо розміри
        ratio = width / w                     # Розрахунок коефіцієнта масштабування
        new_dim = (width, int(h * ratio))     # Нові розміри (ширина, висота)
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)  # Масштабуємо зображення

    @staticmethod
    def safe_hconcat(img_list):
        """Горизонтальна конкатенація з вирівнюванням по висоті."""
        if not img_list:
            return None
        max_height = max(img.shape[0] for img in img_list)   # Шукаємо максимальну висоту
        safe_imgs = []
        for img in img_list:
            if img.shape[0] < max_height:    # Додаємо чорний бордер знизу, якщо зображення нижче
                diff = max_height - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            safe_imgs.append(img)
        return cv2.hconcat(safe_imgs)        # Об'єднуємо горизонтально

    def save_results(self, results, frame_idx):
        for name, img in results.items():                             # Зберігаємо всі зображення з цього кадру
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, img)                                # Записуємо у файл

if __name__ == '__main__':    # Головна точка входу у програму
    print("==== Старт демонстрації OpenCV-операцій ====")
    demo = OpenCVOperationsDemo(
        video_path="Autopilot.mp4",       # Шлях до відеофайлу
        out_dir="opencv_ops_results",     # Папка для збереження результатів
        show=True,                       # Показувати результат на екрані
        save_frames=False,               # Не зберігати файли
        max_frames=60,                   # Максимум 60 кадрів
        thumb_width=400                  # Ширина мініатюри для мозаїки
    )
    demo.process()                       # Запускаємо обробку
    print("==== Кінець програми ====")
