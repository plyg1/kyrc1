import cv2           # Імпортуємо OpenCV для роботи із зображеннями та відео
import numpy as np    # Імпортуємо NumPy для числових операцій та роботи з масивами
import os            # Імпортуємо os для роботи з файловою системою

class GeometricTransformsDemo:   # Оголошення класу для демонстрації геометричних перетворень
    def __init__(self, video_path="Autopilot.mp4", out_dir="geom_results", show=True, save_frames=False, max_frames=60, thumb_width=260):
        self.video_path = video_path        # Шлях до відеофайлу
        self.out_dir = out_dir              # Папка для збереження результатів
        self.show = show                    # Показувати результати чи ні
        self.save_frames = save_frames      # Зберігати кадри чи ні
        self.max_frames = max_frames        # Максимальна кількість кадрів
        self.thumb_width = thumb_width      # Ширина мініатюри для мозаїки

        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)   # Створюємо директорію для збереження, якщо треба

    def process(self):
        print(f"Пробую відкрити відео: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)        # Відкриваємо відеофайл
        if not cap.isOpened():                         # Якщо не вдалося відкрити
            print(f"[ОШИБКА] Не вдалося відкрити відеофайл: {self.video_path}")
            return

        frame_count = 0                                # Лічильник кадрів
        while True:
            ret, frame = cap.read()                    # Зчитуємо кадр
            if not ret or frame_count >= self.max_frames:   # Якщо кінець відео або ліміт кадрів
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break

            results = self.apply_geometric_transforms(frame)  # Застосовуємо всі геометричні перетворення

            if self.show:
                self.display_results(results)              # Показуємо результати на екрані

            if self.save_frames:
                self.save_results(results, frame_count)    # Зберігаємо результати у файли

            if cv2.waitKey(20) & 0xFF == ord('q'):         # Вихід за натисканням 'q'
                print("Вихід за запитом користувача (q)")
                break

            frame_count += 1                               # Збільшуємо лічильник

        cap.release()                                      # Звільняємо ресурси відео
        cv2.destroyAllWindows()                            # Закриваємо всі вікна
        print(f"Оброблено кадрів: {frame_count}")

    def apply_geometric_transforms(self, frame):
        h, w = frame.shape[:2]          # Отримуємо висоту та ширину кадра

        # 1. Масштабування (зменшуємо в 2 рази)
        scaled = cv2.resize(frame, (w//2, h//2), interpolation=cv2.INTER_AREA)

        # 2. Поворот на 30 градусів відносно центру
        angle = 30
        center = (w//2, h//2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)    # Матриця повороту
        rotated = cv2.warpAffine(frame, M_rot, (w, h))         # Застосовуємо поворот

        # 3. Віддзеркалення по горизонталі
        flipped = cv2.flip(frame, 1)

        # 4. Зсув (на 10% ширини і висоти вправо-вниз)
        tx, ty = int(w * 0.1), int(h * 0.1)                   # Розраховуємо зміщення
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])        # Матриця зсуву
        translated = cv2.warpAffine(frame, M_trans, (w, h))   # Застосовуємо зсув

        # 5. Аффінне перетворення (зміщуємо три точки)
        pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])                # Три точки з оригіналу
        pts2 = np.float32([[w*0.0, h*0.1], [w*0.9, h*0.2], [w*0.2, h*0.9]])  # Їх нові координати
        M_aff = cv2.getAffineTransform(pts1, pts2)                  # Матриця аффінного перетворення
        affine = cv2.warpAffine(frame, M_aff, (w, h))               # Застосовуємо аффінне перетворення

        # 6. Перспективне перетворення (імітація "нахилу")
        pts1 = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])       # 4 кути оригіналу
        pts2 = np.float32([[w*0.1, h*0.2], [w*0.9, h*0.05], [w*0.95, h*0.9], [w*0.15, h*0.95]])  # Нові точки
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)             # Матриця перспективного перетворення
        persp = cv2.warpPerspective(frame, M_persp, (w, h))           # Застосовуємо перспективне перетворення

        return {
            "original": frame,       # Оригінал
            "scaled": scaled,        # Масштабування
            "rotated": rotated,      # Поворот
            "flipped": flipped,      # Дзеркало
            "translated": translated,# Зсув
            "affine": affine,        # Аффінне перетворення
            "perspective": persp     # Перспектива
        }

    def display_results(self, results):
        titles = ["original", "scaled", "rotated", "flipped", "translated", "affine", "perspective"] # Список всіх картинок
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in titles if k in results] # Масштабуємо до однакової ширини

        # Мозаїка: два рядки (4 + 3)
        row1 = self.safe_hconcat(thumbs[:4])   # Перший ряд (4 зображення)
        row2 = self.safe_hconcat(thumbs[4:])   # Другий ряд (3 зображення)

        # Вирівнюємо по ширині (додаємо чорну смугу, якщо ряди різні за шириною)
        if row1.shape[1] > row2.shape[1]:
            diff = row1.shape[1] - row2.shape[1]
            row2 = cv2.copyMakeBorder(row2, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
        elif row2.shape[1] > row1.shape[1]:
            diff = row2.shape[1] - row1.shape[1]
            row1 = cv2.copyMakeBorder(row1, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
        # Вирівнюємо по висоті (додаємо чорну смугу, якщо ряди різні за висотою)
        if row1.shape[0] != row2.shape[0]:
            maxh = max(row1.shape[0], row2.shape[0])
            if row1.shape[0] < maxh:
                row1 = cv2.copyMakeBorder(row1, 0, maxh-row1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            if row2.shape[0] < maxh:
                row2 = cv2.copyMakeBorder(row2, 0, maxh-row2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        mosaic = cv2.vconcat([row1, row2])  # Об’єднуємо два рядки вертикально
        cv2.imshow('Geometric Transformations', mosaic) # Показуємо мозаїку у вікні

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]                   # Беремо розміри зображення
        ratio = width / w                      # Розрахунок масштабу
        new_dim = (width, int(h * ratio))      # Нові розміри
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)   # Масштабуємо

    @staticmethod
    def safe_hconcat(img_list):
        """Універсальна горизонтальна конкатенація зображень різної висоти."""
        if not img_list:
            return None
        max_height = max(img.shape[0] for img in img_list)  # Максимальна висота серед усіх
        safe_imgs = []
        for img in img_list:
            if img.shape[0] < max_height:  # Якщо зображення нижче — додаємо чорну смугу знизу
                diff = max_height - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            safe_imgs.append(img)
        return cv2.hconcat(safe_imgs)      # Конкатенуємо всі зображення горизонтально

    def save_results(self, results, frame_idx):
        for name, img in results.items():                       # Зберігаємо всі картинки з поточного кадру
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")   # Формуємо ім'я файлу
            cv2.imwrite(filename, img)                          # Зберігаємо у файл

if __name__ == '__main__':    # Головна точка входу у програму
    print("==== Старт геометричних перетворень ====")
    demo = GeometricTransformsDemo(
        video_path="Autopilot.mp4",       # Шлях до відеофайлу
        out_dir="geom_results",           # Папка для збереження результатів
        show=True,                        # Показувати результат
        save_frames=False,                # Не зберігати кадри
        max_frames=60,                    # Максимум 60 кадрів
        thumb_width=460                   # Ширина мініатюри (для мозаїки)
    )
    demo.process()                        # Запускаємо обробку
    print("==== Кінець програми ====")
