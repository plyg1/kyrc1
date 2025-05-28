import cv2              # Імпортуємо OpenCV для роботи з відео та зображеннями
import numpy as np      # NumPy для роботи з масивами (у цьому коді прямо не використовується)
import os               # Модуль os для роботи з файловою системою

class GaussianFilterDemo:   # Клас для демонстрації фільтрації Гаусовим фільтром із різними розмірами ядра
    def __init__(self, video_path="Autopilot.mp4", out_dir="gauss_results", show=True, save_frames=False, max_frames=60, thumb_width=320, kernel_sizes=(3, 15, 31, 61)):
        self.video_path = video_path        # Шлях до відеофайлу
        self.out_dir = out_dir              # Папка для збереження результатів
        self.show = show                    # Показувати результати на екрані чи ні
        self.save_frames = save_frames      # Зберігати результати у файли чи ні
        self.max_frames = max_frames        # Максимальна кількість кадрів для обробки
        self.thumb_width = thumb_width      # Ширина кожної мініатюри для мозаїки
        self.kernel_sizes = kernel_sizes    # Кортеж розмірів ядер для Гаусового фільтра

        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)   # Створюємо папку для результатів, якщо треба

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

            results = self.apply_gauss_filters(frame)  # Застосовуємо Гаусові фільтри з різними ядрами

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

    def apply_gauss_filters(self, frame):
        results = {"original": frame}                 # Додаємо оригінальний кадр
        for k in self.kernel_sizes:                   # Проходимо по всіх розмірах ядра
            blurred = cv2.GaussianBlur(frame, (k, k), 0)           # Застосовуємо Гаусів фільтр із ядром kxk
            results[f"gauss_{k}x{k}"] = blurred                    # Додаємо результат у словник
        return results

    def display_results(self, results):
        titles = ["original"] + [f"gauss_{k}x{k}" for k in self.kernel_sizes]  # Назви картинок для мозаїки
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in titles if k in results] # Масштабуємо всі до однієї ширини
        row1 = self.safe_hconcat(thumbs[:3])         # Перший рядок — перші три картинки
        row2 = self.safe_hconcat(thumbs[3:])         # Другий рядок — решта

        # Вирівнюємо по ширині/висоті (додаємо чорний бордер, якщо потрібно)
        if row2 is not None:
            if row1.shape[1] > row2.shape[1]:
                diff = row1.shape[1] - row2.shape[1]
                row2 = cv2.copyMakeBorder(row2, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
            elif row2.shape[1] > row1.shape[1]:
                diff = row2.shape[1] - row1.shape[1]
                row1 = cv2.copyMakeBorder(row1, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
            if row1.shape[0] != row2.shape[0]:
                maxh = max(row1.shape[0], row2.shape[0])
                if row1.shape[0] < maxh:
                    row1 = cv2.copyMakeBorder(row1, 0, maxh-row1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                if row2.shape[0] < maxh:
                    row2 = cv2.copyMakeBorder(row2, 0, maxh-row2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            mosaic = cv2.vconcat([row1, row2])        # Склеюємо два рядки вертикально
        else:
            mosaic = row1
        cv2.imshow('Gaussian Filter: різні великі маски (кольорове)', mosaic)   # Показуємо мозаїку у вікні

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]                        # Беремо розміри картинки
        ratio = width / w                           # Коефіцієнт масштабування
        new_dim = (width, int(h * ratio))           # Нові розміри (ширина, висота)
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)  # Масштабуємо

    @staticmethod
    def safe_hconcat(img_list):
        img_list = [img for img in img_list if img is not None]        # Видаляємо None-елементи
        if not img_list:
            return None
        max_height = max(img.shape[0] for img in img_list)             # Знаходимо максимальну висоту
        safe_imgs = []
        for img in img_list:
            if img.shape[0] < max_height:                              # Додаємо чорний бордер знизу, якщо потрібно
                diff = max_height - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            safe_imgs.append(img)
        return cv2.hconcat(safe_imgs)                                  # Склеюємо горизонтально

    def save_results(self, results, frame_idx):
        for name, img in results.items():                             # Зберігаємо всі картинки для цього кадру
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, img)                                # Записуємо у файл

if __name__ == '__main__':    # Головна точка входу у програму
    print("==== Старт Gaussian фільтрації (кольорова) ====")
    demo = GaussianFilterDemo(
        video_path="Autopilot.mp4",      # Шлях до відеофайлу
        out_dir="gauss_results",         # Папка для збереження
        show=True,                       # Показувати результат
        save_frames=False,               # Не зберігати файли
        max_frames=60,                   # Максимум 60 кадрів
        thumb_width=320,                 # Ширина мініатюри
        kernel_sizes=(3, 15, 31, 61)     # Розміри ядер для фільтрації
    )
    demo.process()                       # Запускаємо обробку відео
    print("==== Кінець програми ====")
