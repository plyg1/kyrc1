import cv2
import os

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

        # Підготовка CLAHE
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

            # 1. Перетворення в GRAY
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 2. Підсилення контрасту (CLAHE)
            contrast = self.clahe.apply(gray)

            # 3. Мозаїка для показу
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
        # Масштабуємо для мозаїки
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in ["original", "gray", "gray_contrast"]]
        mosaic = cv2.hconcat(thumbs)
        cv2.imshow('GRAY+Контраст (CLAHE)', mosaic)

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

if __name__ == '__main__':
    print("==== Старт GRAY+Контраст ====")
    processor = GrayContrastProcessor(
        video_path="Autopilot.mp4",
        out_dir="gray_contrast_results",
        show=True,
        save_frames=False,     
        max_frames=100,
        thumb_width=400,       
        clahe_clip=2.0,
        clahe_grid=8
    )
    processor.process()
    print("==== Кінець програми ====")
