import matplotlib.pyplot as plt   # Vẽ biểu đồ, hiển thị ảnh
import os                         # Quản lý đường dẫn, thư mục
import cv2                        # Thư viện xử lý ảnh (OpenCV)
import numpy as np                # Tính toán ma trận, xử lý pixel

# 1. LIÊN KẾT MODULE
try:
    from preprocessing import BRIGHT_DIR, is_very_bright, BRIGHT_PIXEL_THRESHOLD
except ImportError:
    print("Cảnh báo: Không tìm thấy file 'preprocessing.py'.")
    BRIGHT_DIR = "../data/bright_images"
    BRIGHT_PIXEL_THRESHOLD = 245
    def is_very_bright(img): return False

# 2. HÀM HIỂN THỊ EDA
def show_random_bright_images_train(cls_name, max_show=5):
    """
    Hiển thị các ảnh bị lỗi 'Quá sáng' trong một lớp cụ thể (lấy từ BRIGHT_DIR).
    - cls_name: tên lớp (thư mục con trong BRIGHT_DIR)
    - max_show: số lượng ảnh tối đa hiển thị
    """
    cls_path = os.path.join(BRIGHT_DIR, cls_name)

    if not os.path.exists(cls_path):
        print(f"Lỗi: Không tìm thấy thư mục lớp tại: {cls_path}")
        return

    files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nĐang quét lớp '{cls_name}' (Tổng: {len(files)} ảnh)...")

    found_images = []
    for img_name in files:
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Ảnh trong BRIGHT_DIR vốn đã được lọc là quá sáng, nhưng ta vẫn tính tỷ lệ để hiển thị
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ratio = np.sum(gray >= BRIGHT_PIXEL_THRESHOLD) / gray.size

        found_images.append((img_name, img, ratio))
        if len(found_images) >= max_show:
            break

    if not found_images:
        print(f"Không tìm thấy ảnh nào quá sáng trong lớp {cls_name}.")
        return

    print(f"Phát hiện {len(found_images)} ảnh quá sáng. Đang hiển thị...")

    plt.figure(figsize=(15, 5))
    for i, (fname, img, ratio) in enumerate(found_images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, max_show, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"{fname}\nTỷ lệ chói: {ratio:.2f}", fontsize=9, color='red')
        plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if os.path.exists(BRIGHT_DIR):
        all_classes = os.listdir(BRIGHT_DIR)
        if all_classes:
            first_class = all_classes[0]
            show_random_bright_images_train(first_class, max_show=5)