import cv2
import numpy as np
import os

class FilteringDemo:
    def __init__(self, video_path="Autopilot.mp4", out_dir="filtering_results", show=True, save_frames=False, max_frames=60, thumb_width=280):
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

            results = self.apply_filters(frame)

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

    def apply_filters(self, frame):
        # 1. Оригінал
        original = frame

        # 2. Gaussian Blur
        gauss = cv2.GaussianBlur(frame, (11, 11), 0)

        # 3. Median Blur
        median = cv2.medianBlur(frame, 9)

        # 4. Bilateral
        bilateral = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        # 5. Box Filter (average)
        box = cv2.blur(frame, (9, 9))

        # 6. Sharpening (custom kernel)
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
        titles = ["original", "gauss", "median", "bilateral", "box", "sharpened"]
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in titles if k in results]
        # Два рядки по три
        row1 = self.safe_hconcat(thumbs[:3])
        row2 = self.safe_hconcat(thumbs[3:])
        # Вирівнювання по ширині/висоті
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
        cv2.imshow('OpenCV Filtering Demo', mosaic)

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]
        ratio = width / w
        new_dim = (width, int(h * ratio))
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def safe_hconcat(img_list):
        """Горизонтальна конкатенація з вирівнюванням висоти."""
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

if __name__ == '__main__':
    print("==== Старт OpenCV Filtering Demo ====")
    demo = FilteringDemo(
        video_path="Autopilot.mp4",
        out_dir="filtering_results",
        show=True,
        save_frames=False,    
        max_frames=60,
        thumb_width=280
    )
    demo.process()
    print("==== Кінець програми ====")
