import cv2
import numpy as np
import os

class ShearFilterDemo:
    def __init__(self, video_path="Autopilot.mp4", out_dir="shear_results", show=True, save_frames=False, max_frames=60, thumb_width=500, shear_x=0.5, shear_y=0.0):
        self.video_path = video_path
        self.out_dir = out_dir
        self.show = show
        self.save_frames = save_frames
        self.max_frames = max_frames
        self.thumb_width = thumb_width
        self.shear_x = shear_x
        self.shear_y = shear_y

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

            sheared = self.apply_shear(frame, self.shear_x, self.shear_y)

            results = {
                "original": frame,
                "shear": sheared
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

    def apply_shear(self, img, shear_x=0.5, shear_y=0.0):
        h, w = img.shape[:2]
        M = np.float32([
            [1, shear_x, 0],
            [shear_y, 1, 0]
        ])
        new_w = int(w + abs(shear_x) * h)
        new_h = int(h + abs(shear_y) * w)
        sheared = cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
        return sheared

    def display_results(self, results):
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in ["original", "shear"]]
        mosaic = self.safe_hconcat(thumbs)
        cv2.imshow('Shear Transform (Скос)', mosaic)

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
    print("==== Старт методу скосу для фільтрації ====")
    demo = ShearFilterDemo(
        video_path="Autopilot.mp4",
        out_dir="shear_results",
        show=True,
        save_frames=False,
        max_frames=60,
        thumb_width=500,
        shear_x=0.5,
        shear_y=0.0
    )
    demo.process()
    print("==== Кінець програми ====")
