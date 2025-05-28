import cv2
import numpy as np
import os

class PrewittEdgeDemo:
    def __init__(self, video_path="Autopilot.mp4", out_dir="prewitt_results", show=True, save_frames=False, max_frames=60, thumb_width=300, thresholds=(50, 100, 150)):
        self.video_path = video_path
        self.out_dir = out_dir
        self.show = show
        self.save_frames = save_frames
        self.max_frames = max_frames
        self.thumb_width = thumb_width
        self.thresholds = thresholds  # Набір порогів

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

            results = self.apply_prewitt(frame)

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

    def apply_prewitt(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Prewitt kernels
        kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=np.float32)
        kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=np.float32)

        # Фільтрація по X та Y
        grad_x = cv2.filter2D(gray, -1, kernelx)
        grad_y = cv2.filter2D(gray, -1, kernely)

        # Сумарне зображення градієнта
        prewitt = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        
        # Бінаризація по різних порогах
        edges = {}
        for t in self.thresholds:
            _, threshed = cv2.threshold(prewitt, t, 255, cv2.THRESH_BINARY)
            edges[f'prewitt_thr{t}'] = threshed

        # Конвертуємо для відображення (BGR)
        def to3(img): return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        results = {
            "original": frame,
            "gray": to3(gray),
            "prewitt": to3(prewitt),
        }
        # Додаємо всі бінаризовані варіанти
        for k, v in edges.items():
            results[k] = to3(v)
        return results

    def display_results(self, results):
        # Формуємо мозаїку: оригінал, gray, prewitt, prewitt_thr1, prewitt_thr2, prewitt_thr3
        titles = ["original", "gray", "prewitt"] + [k for k in results if k.startswith("prewitt_thr")]
        thumbs = [self.resize_to_width(results[k], self.thumb_width) for k in titles if k in results]
        # Мозаїка: перші 3 в 1 рядок, інші — в другий
        row1 = self.safe_hconcat(thumbs[:3])
        row2 = self.safe_hconcat(thumbs[3:])
        # Вирівнювання по ширині/висоті
        if row2 is not None:
            if row1.shape[1] > row2.shape[1]:
                diff = row1.shape[1] - row2.shape[1]
                row2 = cv2.copyMakeBorder(row2, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
            elif row2.shape[1] > row1.shape[1]:
                diff = row2.shape[1] - row1.shape[1]
                row1 = cv2.copyMakeBorder(row1, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
            # По висоті
            if row1.shape[0] != row2.shape[0]:
                maxh = max(row1.shape[0], row2.shape[0])
                if row1.shape[0] < maxh:
                    row1 = cv2.copyMakeBorder(row1, 0, maxh-row1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                if row2.shape[0] < maxh:
                    row2 = cv2.copyMakeBorder(row2, 0, maxh-row2.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            mosaic = cv2.vconcat([row1, row2])
        else:
            mosaic = row1
        cv2.imshow('Prewitt Edge Detection (Різні пороги)', mosaic)

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
    print("==== Старт Prewitt edge detection ====")
    demo = PrewittEdgeDemo(
        video_path="Autopilot.mp4",
        out_dir="prewitt_results",
        show=True,
        save_frames=False,      # True якщо треба зберігати
        max_frames=60,
        thumb_width=300,        
        thresholds=(50, 100, 150)
    )
    demo.process()
    print("==== Кінець програми ====")
