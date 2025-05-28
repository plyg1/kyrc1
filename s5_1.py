import cv2
import numpy as np
import os

class OpenCVOperationsDemo:
    def __init__(self, video_path="Autopilot.mp4", out_dir="opencv_ops_results", show=True, save_frames=False, max_frames=60, thumb_width=270):
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

            results = self.apply_operations(frame)

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

    def apply_operations(self, frame):
        # 1. Переводимо в відтінки сірого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Бінаризація (поріг)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # 3. Морфологія (ерозія і дилатація)
        kernel = np.ones((5,5), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=1)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # 4. Blur (розмиття)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        # 5. Інверсія (арифметична операція)
        inverted = cv2.bitwise_not(frame)

        # 6. Виділення контурів
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoured = frame.copy()
        cv2.drawContours(contoured, contours, -1, (0, 255, 0), 2)

        # 7. Анотація (текст + фігури)
        annotated = frame.copy()
        cv2.rectangle(annotated, (30, 30), (200, 120), (0, 0, 255), 3)
        cv2.circle(annotated, (100, 220), 40, (255, 0, 0), 3)
        cv2.putText(annotated, "OpenCV Demo", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # Конвертуємо всі до 3 каналів для конкатенації
        def to3(img):
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

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
        # Мозаїка: 5+4
        row1 = self.safe_hconcat(thumbs[:5])
        row2 = self.safe_hconcat(thumbs[5:])
        # Вирівнюємо по ширині
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
        """Горизонтальна конкатенація з вирівнюванням висоти."""
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

if __name__ == '__main__':
    print("==== Старт демонстрації OpenCV-операцій ====")
    demo = OpenCVOperationsDemo(
        video_path="Autopilot.mp4",
        out_dir="opencv_ops_results",
        show=True,
        save_frames=False,     
        max_frames=60,
        thumb_width=400        
    )
    demo.process()
    print("==== Кінець програми ====")
