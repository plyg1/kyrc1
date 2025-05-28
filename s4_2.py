import cv2           # Імпортуємо бібліотеку OpenCV для роботи із зображеннями/відео
import numpy as np    # Імпортуємо NumPy для матричних операцій
import os            # Модуль os для роботи з файловою системою

class ShearFilterDemo:    # Клас для демонстрації геометричного скосу (shear)
    def __init__(self, video_path="Autopilot.mp4", out_dir="shear_results", show=True, save_frames=False, max_frames=60, thumb_width=500, shear_x=0.5, shear_y=0.0):
        self.video_path = video_path      # Шлях до відеофайлу
        self.out_dir = out_dir            # Папка для збереження результатів
        self.show = show                  # Чи показувати результати на екрані
        self.save_frames = save_frames    # Чи зберігати результати у файли
        self.max_frames = max_frames      # Максимум кадрів для обробки
        self.thumb_width = thumb_width    # Ширина мініатюр для мозаїки
        self.shear_x = shear_x            # Коефіцієнт скосу по X
        self.shear_y = shear_y            # Коефіцієнт скосу по Y

        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)   # Створюємо директорію для збереження, якщо треба

    def process(self):
        print(f"Пробую відкрити відео: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)      # Відкриваємо відеофайл
        if not cap.isOpened():                      # Якщо не вдалося відкрити
            print(f"[ОШИБКА] Не вдалося відкрити відеофайл: {self.video_path}")
            return

        frame_count = 0                             # Лічильник кадрів
        while True:
            ret, frame = cap.read()                 # Зчитуємо наступний кадр
            if not ret or frame_count >= self.max_frames:   # Якщо відео закінчилось або ліміт кадрів
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break

            sheared = self.apply_shear(frame, self.shear_x, self.shear_y)   # Застосовуємо shear (скіс)

            results = {
                "original": frame,      # Оригінальний кадр
                "shear": sheared        # Кадр після скосу
            }

            if self.show:
                self.display_results(results)       # Показуємо результати на екрані

            if self.save_frames:
                self.save_results(results, frame_count)   # Зберігаємо результати у файли

            if cv2.waitKey(20) & 0xFF == ord('q'):        # Якщо натиснули 'q' — вихід
                print("Вихід за запитом користувача (q)")
                break

            frame_count += 1                            # Збільшуємо лічильник кадрів

        cap.release()                                   # Звільняємо ресурси відео
        cv2.destroyAllWindows()                         # Закриваємо всі вікна
        print(f"Оброблено кадрів: {frame_count}")

    def apply_shear(self, img, shear_x=0.5, shear_y=0.0):
        h, w = img.shape[:2]              # Висота і ширина зображення
        # Формуємо матрицю скосу (shear)
        M = np.float32([
            [1, shear_x, 0],              # [1, shear_x, 0] — X напрямок
            [shear_y, 1, 0]               # [shear_y, 1, 0] — Y напрямок
        ])
        # Вираховуємо нову ширину і висоту з урахуванням скосу
        new_w = int(w + abs(shear_x) * h)
        new_h = int(h + abs(shear_y) * w)
        # Застосовуємо скос до зображення (borderMode: повторювати крайні пікселі)
        sheared = cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
        return sheared

    def display_results(self, results):
        # Масштабуємо кадри до однієї ширини для красивої мозаїки
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in ["original", "shear"]]
        mosaic = self.safe_hconcat(thumbs)   # З'єднуємо горизонтально
        cv2.imshow('Shear Transform (Скос)', mosaic)   # Показуємо результат

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]                     # Отримуємо розміри
        ratio = width / w                        # Розраховуємо масштабний коефіцієнт
        new_dim = (width, int(h * ratio))        # Нові розміри
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)   # Масштабуємо

    @staticmethod
    def safe_hconcat(img_list):
        """Горизонтальна конкатенація з вирівнюванням висоти."""
        if not img_list:
            return None
        max_height = max(img.shape[0] for img in img_list)   # Знаходимо максимальну висоту
        safe_imgs = []
        for img in img_list:
            if img.shape[0] < max_height:        # Якщо зображення нижче — додаємо чорний "бордер" знизу
                diff = max_height - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            safe_imgs.append(img)
        return cv2.hconcat(safe_imgs)            # З'єднуємо всі картинки в один ряд

    def save_results(self, results, frame_idx):
        for name, img in results.items():                          # Зберігаємо всі результати
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")   # Формуємо ім'я файлу
            cv2.imwrite(filename, img)                             # Зберігаємо картинку

if __name__ == '__main__':    # Головна точка входу у програму
    print("==== Старт методу скосу для фільтрації ====")
    demo = ShearFilterDemo(
        video_path="Autopilot.mp4",      # Шлях до відеофайлу
        out_dir="shear_results",         # Папка для збереження результатів
        show=True,                       # Показувати на екрані
        save_frames=False,               # Не зберігати результати у файли
        max_frames=60,                   # Максимум 60 кадрів
        thumb_width=500,                 # Ширина мініатюр для мозаїки
        shear_x=0.5,                     # Коефіцієнт скосу по X
        shear_y=0.0                      # Коефіцієнт скосу по Y
    )
    demo.process()                       # Запускаємо обробку відео
    print("==== Кінець програми ====")
