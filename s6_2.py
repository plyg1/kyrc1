import cv2
import numpy as np
import os

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

if __name__ == '__main__':
    print("==== Старт Gaussian фільтрації (кольорова) ====")
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
    print("==== Кінець програми ====")
