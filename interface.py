import cv2
import numpy as np
import os
import sys

# --- GrayContrastProcessor ---
class GrayContrastProcessor:
    def __init__(self, video_path="Autopilot.mp4", out_dir="gray_contrast_results", show=True, save_frames=False, max_frames=100, thumb_width=400, clahe_clip=2.0, clahe_grid=8):
        self.video_path = video_path
        self.out_dir = out_dir
        self.show = show
        self.save_frames = save_frames
        self.max_frames = max_frames
        self.thumb_width = thumb_width
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(self.clahe_grid, self.clahe_grid))

    def process(self):
        print(f"Пробую відкрити відео: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ОШИБКА] Не вдалося відкрити відеофайл: {self.video_path}")
            return

        frame_count = 0
        print("Старт обробки відео...")
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= self.max_frames:
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrast = self.clahe.apply(gray)

            results = {
                "original": frame,
                "gray": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                "gray_contrast": cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
            }

            if self.show:
                self.display_results(results)
            if self.save_frames:
                self.save_results(results, frame_count)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                print("Вихід за запитом користувача (q)")
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"Оброблено кадрів: {frame_count}")

    def display_results(self, results):
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in ["original", "gray", "gray_contrast"]]
        mosaic = cv2.hconcat(thumbs)
        cv2.imshow('GRAY+Контраст', mosaic)

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]
        ratio = width / w
        new_dim = (width, int(h * ratio))
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

    def save_results(self, results, frame_idx):
        for name, img in results.items():
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, img)

# --- GeometricTransformsDemo ---
class GeometricTransformsDemo:
    def __init__(self, video_path="Autopilot.mp4", out_dir="geom_results", show=True, save_frames=False, max_frames=60, thumb_width=260):
        self.video_path = video_path
        self.out_dir = out_dir
        self.show = show
        self.save_frames = save_frames
        self.max_frames = max_frames
        self.thumb_width = thumb_width
        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)

    def process(self):
        print(f"Пробую відкрити відео: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ОШИБКА] Не вдалося відкрити відеофайл: {self.video_path}")
            return
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= self.max_frames:
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break
            results = self.apply_geometric_transforms(frame)
            if self.show:
                self.display_results(results)
            if self.save_frames:
                self.save_results(results, frame_count)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                print("Вихід за запитом користувача (q)")
                break
            frame_count += 1
        cap.release()
        cv2.destroyAllWindows()
        print(f"Оброблено кадрів: {frame_count}")

    def apply_geometric_transforms(self, frame):
        h, w = frame.shape[:2]
        scaled = cv2.resize(frame, (w//2, h//2), interpolation=cv2.INTER_AREA)
        angle = 30
        center = (w//2, h//2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M_rot, (w, h))
        flipped = cv2.flip(frame, 1)
        tx, ty = int(w * 0.1), int(h * 0.1)
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(frame, M_trans, (w, h))
        pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])
        pts2 = np.float32([[w*0.0, h*0.1], [w*0.9, h*0.2], [w*0.2, h*0.9]])
        M_aff = cv2.getAffineTransform(pts1, pts2)
        affine = cv2.warpAffine(frame, M_aff, (w, h))
        pts1 = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
        pts2 = np.float32([[w*0.1, h*0.2], [w*0.9, h*0.05], [w*0.95, h*0.9], [w*0.15, h*0.95]])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        persp = cv2.warpPerspective(frame, M_persp, (w, h))
        return {
            "original": frame,
            "scaled": scaled,
            "rotated": rotated,
            "flipped": flipped,
            "translated": translated,
            "affine": affine,
            "perspective": persp
        }

    def display_results(self, results):
        titles = ["original", "scaled", "rotated", "flipped", "translated", "affine", "perspective"]
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in titles if k in results]
        row1 = self.safe_hconcat(thumbs[:4])
        row2 = self.safe_hconcat(thumbs[4:])
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
        mosaic = cv2.vconcat([row1, row2])
        cv2.imshow('Geometric Transformations', mosaic)

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]
        ratio = width / w
        new_dim = (width, int(h * ratio))
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def safe_hconcat(img_list):
        if not img_list:
            return None
        max_height = max(img.shape[0] for img in img_list)
        safe_imgs = []
        for img in img_list:
            if img.shape[0] < max_height:
                diff = max_height - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            safe_imgs.append(img)
        return cv2.hconcat(safe_imgs)

    def save_results(self, results, frame_idx):
        for name, img in results.items():
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, img)

# --- GaussianFilterDemo ---
class GaussianFilterDemo:
    def __init__(self, video_path="Autopilot.mp4", out_dir="gauss_results", show=True, save_frames=False, max_frames=60, thumb_width=320, kernel_sizes=(3, 15, 31, 61)):
        self.video_path = video_path
        self.out_dir = out_dir
        self.show = show
        self.save_frames = save_frames
        self.max_frames = max_frames
        self.thumb_width = thumb_width
        self.kernel_sizes = kernel_sizes

        if self.save_frames:
            os.makedirs(self.out_dir, exist_ok=True)

    def process(self):
        print(f"Пробую відкрити відео: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ОШИБКА] Не вдалося відкрити відеофайл: {self.video_path}")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= self.max_frames:
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break

            results = self.apply_gauss_filters(frame)

            if self.show:
                self.display_results(results)

            if self.save_frames:
                self.save_results(results, frame_count)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                print("Вихід за запитом користувача (q)")
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        print(f"Оброблено кадрів: {frame_count}")

    def apply_gauss_filters(self, frame):
        results = {"original": frame}
        for k in self.kernel_sizes:
            blurred = cv2.GaussianBlur(frame, (k, k), 0)
            results[f"gauss_{k}x{k}"] = blurred
        return results

    def display_results(self, results):
        titles = ["original"] + [f"gauss_{k}x{k}" for k in self.kernel_sizes]
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in titles if k in results]
        row1 = self.safe_hconcat(thumbs[:3])
        row2 = self.safe_hconcat(thumbs[3:])
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
            mosaic = cv2.vconcat([row1, row2])
        else:
            mosaic = row1
        cv2.imshow('Gaussian Filter: різні великі маски (кольорове)', mosaic)

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]
        ratio = width / w
        new_dim = (width, int(h * ratio))
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def safe_hconcat(img_list):
        img_list = [img for img in img_list if img is not None]
        if not img_list:
            return None
        max_height = max(img.shape[0] for img in img_list)
        safe_imgs = []
        for img in img_list:
            if img.shape[0] < max_height:
                diff = max_height - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            safe_imgs.append(img)
        return cv2.hconcat(safe_imgs)

    def save_results(self, results, frame_idx):
        for name, img in results.items():
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, img)

# --- OpenCVOperationsDemо ---
class OpenCVOperationsDemo:
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
        titles = ["original", "gray", "binary", "eroded", "dilated", "blurred", "inverted", "contours", "annotated"]
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in titles if k in results]
        row1 = self.safe_hconcat(thumbs[:5])
        row2 = self.safe_hconcat(thumbs[5:])
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
        mosaic = cv2.vconcat([row1, row2])
        cv2.imshow('OpenCV Operations Demo', mosaic)

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]
        ratio = width / w
        new_dim = (width, int(h * ratio))
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def safe_hconcat(img_list):
        if not img_list:
            return None
        max_height = max(img.shape[0] for img in img_list)
        safe_imgs = []
        for img in img_list:
            if img.shape[0] < max_height:
                diff = max_height - img.shape[0]
                img = cv2.copyMakeBorder(img, 0, diff, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            safe_imgs.append(img)
        return cv2.hconcat(safe_imgs)

    def save_results(self, results, frame_idx):
        for name, img in results.items():
            filename = os.path.join(self.out_dir, f"{name}_{frame_idx:04d}.jpg")
            cv2.imwrite(filename, img)

# ІНТЕРФЕЙС
def main():
    while True:
        print("\n--- Меню ---")    # Виводимо заголовок меню
        print("1. Gray+Контраст")      # Варіант 1: контрастування зображення
        print("2. Геометричні перетворення")   # Варіант 2: різні геометричні трансформації
        print("3. Gaussian ")# Варіант 3: розмиття Гаусса з різними розмірами
        print("4. prewitt") # Варіант 4: універсальні операції
        print("5. Вийти")                      # Варіант 5: вихід з програми
        choice = input("Оберіть дію (1-5): ")  # Запитуємо вибір користувача

        if choice == "1":
            # Запуск GrayContrastProcessor
            demo = GrayContrastProcessor(
                video_path="Autopilot.mp4",
                out_dir="gray_contrast_results",
                show=True,
                save_frames=False,
                max_frames=100,
                thumb_width=400,
                clahe_clip=2.0,
                clahe_grid=8
            )
            demo.process() # Запуск обробки
        elif choice == "2":
            # Запуск GeometricTransformsDemo
            demo = GeometricTransformsDemo(
                video_path="Autopilot.mp4",
                out_dir="geom_results",
                show=True,
                save_frames=False,
                max_frames=60,
                thumb_width=260
            )
            demo.process()
        elif choice == "3":
            # Запуск GaussianFilterDemo
            demo = GaussianFilterDemo(
                video_path="Autopilot.mp4",
                out_dir="gauss_results",
                show=True,
                save_frames=False,
                max_frames=60,
                thumb_width=320,
                kernel_sizes=(3, 15, 31, 61)
            )
            demo.process()
        elif choice == "4":
            # Запуск OpenCVOperationsDemo
            demo = OpenCVOperationsDemo(
                video_path="Autopilot.mp4",
                out_dir="opencv_ops_results",
                show=True,
                save_frames=False,
                max_frames=60,
                thumb_width=400
            )
            demo.process()
        elif choice == "5":
            print("Вихід із програми.")      # Повідомлення про завершення
            sys.exit(0)                      # Завершення програми
        else:
            print("Некоректний вибір. Спробуйте ще раз.") # Обробка некоректного вводу

if __name__ == '__main__':
    main()
