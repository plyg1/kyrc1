import cv2
import numpy as np
import os

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

        # 1. Масштабування (зменшення у 2 рази)
        scaled = cv2.resize(frame, (w//2, h//2), interpolation=cv2.INTER_AREA)

        # 2. Поворот на 30 градусів (з центру)
        angle = 30
        center = (w//2, h//2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(frame, M_rot, (w, h))

        # 3. Віддзеркалення по горизонталі
        flipped = cv2.flip(frame, 1)

        # 4. Зсув (10% ширини і висоти вправо-вниз)
        tx, ty = int(w * 0.1), int(h * 0.1)
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(frame, M_trans, (w, h))

        # 5. Аффінне перетворення (зміщуємо три точки)
        pts1 = np.float32([[0,0], [w-1,0], [0,h-1]])
        pts2 = np.float32([[w*0.0, h*0.1], [w*0.9, h*0.2], [w*0.2, h*0.9]])
        M_aff = cv2.getAffineTransform(pts1, pts2)
        affine = cv2.warpAffine(frame, M_aff, (w, h))

        # 6. Перспективне перетворення (імітація "нахилу" кадру)
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
        # Мозаїка у два рядки (4+3)
        row1 = self.safe_hconcat(thumbs[:4])
        row2 = self.safe_hconcat(thumbs[4:])
        # Вирівнюємо по ширині
        if row1.shape[1] > row2.shape[1]:
            diff = row1.shape[1] - row2.shape[1]
            row2 = cv2.copyMakeBorder(row2, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
        elif row2.shape[1] > row1.shape[1]:
            diff = row2.shape[1] - row1.shape[1]
            row1 = cv2.copyMakeBorder(row1, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
        # Вирівнюємо по висоті (додатково, на всяк випадок)
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
        """Універсальна горизонтальна конкатенація зображень різної висоти."""
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
    print("==== Старт геометричних перетворень ====")
    demo = GeometricTransformsDemo(
        video_path="Autopilot.mp4",
        out_dir="geom_results",
        show=True,
        save_frames=False,     
        max_frames=60,
        thumb_width=460        
    )
    demo.process()
    print("==== Кінець програми ====")
