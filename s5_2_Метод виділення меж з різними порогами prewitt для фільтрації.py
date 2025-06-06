import cv2             # Імпортуємо OpenCV для роботи з відео/зображеннями
import numpy as np     # Імпортуємо NumPy для матричних операцій
import os              # Модуль os для роботи з файловою системою

class PrewittEdgeDemo:  # Клас для демонстрації виявлення контурів оператором Prewitt
    def __init__(self, video_path="Autopilot.mp4", out_dir="prewitt_results", show=True, save_frames=False, max_frames=60, thumb_width=300, thresholds=(50, 100, 150)):
        self.video_path = video_path           # Шлях до відеофайлу
        self.out_dir = out_dir                 # Папка для збереження результатів
        self.show = show                       # Чи показувати результати на екрані
        self.save_frames = save_frames         # Чи зберігати результати у файли
        self.max_frames = max_frames           # Максимальна кількість кадрів для обробки
        self.thumb_width = thumb_width         # Ширина мініатюри для мозаїки
        self.thresholds = thresholds           # Набір порогів для бінаризації

        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)   # Створюємо папку, якщо треба зберігати

    def process(self):
        print(f"Пробую відкрити відео: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)      # Відкриваємо відеофайл
        if not cap.isOpened():                      # Якщо не вдалося відкрити
            print(f"[ОШИБКА] Не вдалося відкрити відеофайл: {self.video_path}")
            return

        frame_count = 0                             # Лічильник кадрів
        while True:
            ret, frame = cap.read()                 # Зчитуємо наступний кадр
            if not ret or frame_count >= self.max_frames:    # Кінець відео або досягнуто ліміт
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break

            results = self.apply_prewitt(frame)     # Застосовуємо оператор Prewitt

            if self.show:
                self.display_results(results)       # Показуємо результат

            if self.save_frames:
                self.save_results(results, frame_count)   # Зберігаємо результат у файли

            if cv2.waitKey(20) & 0xFF == ord('q'):        # Вихід при натисканні 'q'
                print("Вихід за запитом користувача (q)")
                break

            frame_count += 1                        # Збільшуємо лічильник кадрів

        cap.release()                               # Звільняємо відеоресурси
        cv2.destroyAllWindows()                     # Закриваємо всі вікна
        print(f"Оброблено кадрів: {frame_count}")

    def apply_prewitt(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # Перетворюємо у відтінки сірого

        # Оператор Prewitt: ядра для обчислення градієнта по X і Y
        kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)
        kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)

        # Градієнт по X та Y (згортка з ядрами)
        grad_x = cv2.filter2D(gray, -1, kernelx)
        grad_y = cv2.filter2D(gray, -1, kernely)

        # Об'єднуємо X і Y для отримання повного зображення градієнта
        prewitt = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        
        # Бінаризація Prewitt-фільтра по різних порогах
        edges = {}
        for t in self.thresholds:
            _, threshed = cv2.threshold(prewitt, t, 255, cv2.THRESH_BINARY)
            edges[f'prewitt_thr{t}'] = threshed    # Зберігаємо результат з кожним порогом

        # Функція для переведення у 3-канальний формат (для відображення)
        def to3(img): return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        results = {
            "original": frame,        # Оригінал
            "gray": to3(gray),        # Сіре зображення
            "prewitt": to3(prewitt),  # Градієнт Prewitt
        }
        # Додаємо бінаризовані варіанти
        for k, v in edges.items():
            results[k] = to3(v)
        return results

    def display_results(self, results):
        # Формуємо список назв для показу: оригінал, gray, prewitt, prewitt_thrXX...
        titles = ["original", "gray", "prewitt"] + [k for k in results if k.startswith("prewitt_thr")]
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in titles if k in results]
        # Мозаїка: перші 3 в перший рядок, решта у другий
        row1 = self.safe_hconcat(thumbs[:3])
        row2 = self.safe_hconcat(thumbs[3:])
        # Вирівнювання по ширині/висоті (додаємо чорний бордер якщо треба)
        if row2 is not None:
            if row1.shape[1] > row2.shape[1]:
                diff = row1.shape[1] - row2.shape[1]
                row2 = cv2.copyMakeBorder(row2, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
            elif row2.shape[1] > row1.shape[1]:
                diff = row2.shape[1] - row1.shape[1]
                row1 = cv2.copyMakeBorder(row1, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
            # По висоті
            if row1.shape[0] != row2.shape[0]:
                maxh = max(row1.shape[0], row2.shape[0])
                if row1.shape[0] < maxh:
                    row1 = cv2.copyMakeBorder(row1, 0, maxh-row1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                if row2.shape[0] < maxh:
                    row2 = cv2.copyMakeBorder(row2, 0, maxh-row2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            mosaic = cv2.vconcat([row1, row2])  # Склеюємо два рядки
        else:
            mosaic = row1
        cv2.imshow('Prewitt Edge Detection (Різні пороги)', mosaic)   # Показуємо мозаїку у вікні

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]                      # Беремо розміри
        ratio = width / w                         # Обраховуємо коефіцієнт масштабування
        new_dim = (width, int(h * ratio))         # Нові розміри (ширина, висота)
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)   # Масштабуємо зображення

    @staticmethod
    def safe_hconcat(img_list):
        """Горизонтальна конкатенація з вирівнюванням по висоті."""
        img_list = [img for img in img_list if img is not None]
        if not img_list:
            return None
        max_height = max(img.shape[0] for img in img_list)   # Шукаємо максимальну висоту
        safe_imgs = []
        for img in img_list:
            if img.shape[0] < max_height:   # Додаємо чорний бордер якщо треба
                diff = max_height - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            safe_imgs.append(img)
        return cv2.hconcat(safe_imgs)   # Склеюємо всі картинки

    def save_results(self, results, frame_idx):
        for name, img in results.items():  # Зберігаємо всі зображення для цього кадру
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, img)     # Записуємо у файл

if __name__ == '__main__':   # Головна точка входу у програму
    print("==== Старт Prewitt edge detection ====")
    demo = PrewittEdgeDemo(
        video_path="Autopilot.mp4",     # Шлях до відеофайлу
        out_dir="prewitt_results",      # Папка для збереження
        show=True,                      # Показувати результати
        save_frames=False,              # Не зберігати у файли
        max_frames=60,                  # Максимум 60 кадрів
        thumb_width=300,                # Ширина мініатюри
        thresholds=(50, 100, 150)       # Пороги для бінаризації
    )
    demo.process()                      # Запускаємо обробку
    print("==== Кінець програми ====")
