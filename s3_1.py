import cv2
import os

class VideoColorTransformer:
    COLOR_MODES = {
        "gray": cv2.COLOR_BGR2GRAY,
        "hsv": cv2.COLOR_BGR2HSV,
        "lab": cv2.COLOR_BGR2LAB,
        "ycrcb": cv2.COLOR_BGR2YCrCb,
        "rgb": cv2.COLOR_BGR2RGB,
    }

    def __init__(self, video_path="Autopilot.mp4", out_dir="results", show=True, save_frames=False, max_frames=100, thumb_width=300):
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
        print("Старт обробки відео...")
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= self.max_frames:
                print(f"Завершення: ret={ret}, кадрів={frame_count}")
                break

            results = self.apply_color_transforms(frame)

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

    def apply_color_transforms(self, frame):
        results = {"original": frame}
        for mode, code in self.COLOR_MODES.items():
            converted = cv2.cvtColor(frame, code)
            if mode == "gray":
                results[mode] = cv2.cvtColor(converted, cv2.COLOR_GRAY2BGR)
            else:
                results[mode] = self._to_bgr(converted, mode)
        return results

    @staticmethod
    def _to_bgr(image, mode):
        if mode == "hsv":
            return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif mode == "lab":
            return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        elif mode == "ycrcb":
            return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
        elif mode == "rgb":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def display_results(self, results):
        titles = ["original", "gray", "hsv", "lab", "ycrcb", "rgb"]
        imgs = [results[k] for k in titles if k in results]
        # Масштабуємо всі зображення до thumb_width
        thumbs = [self.resize_to_width(img, self.thumb_width) for img in imgs]
        # Розбиваємо на два рядки (по 3 зображення)
        row1 = cv2.hconcat(thumbs[:3])
        row2 = cv2.hconcat(thumbs[3:])
        mosaic = cv2.vconcat([row1, row2])
        cv2.imshow('Color Transformations', mosaic)

    @staticmethod
    def resize_to_width(img, width):
        h, w = img.shape[:2]
        ratio = width / w
        new_dim = (width, int(h * ratio))
        return cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)

    def save_results(self, results, frame_idx):
        for name, img in results.items():
            filename = os.path.join(
                self.out_dir,
                f"{name}_{frame_idx:04d}.jpg"
            )
            cv2.imwrite(filename, img)

if __name__ == '__main__':
    print("==== Старт програми ====")
    transformer = VideoColorTransformer(
        video_path="Autopilot.mp4",
        out_dir="results",
        show=True,
        save_frames=False,
        max_frames=100,
        thumb_width=300    # Ширина кожного міні-зображення (250, 200)
    )
    transformer.process()
    print("==== Кінець програми ====")
