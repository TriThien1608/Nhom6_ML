

import os                   # Thao tác đường dẫn và thư mục
import cv2                  # Xử lý ảnh (đọc/ghi, chuyển màu, lọc, v.v.)
import shutil               # Sao chép file giữa thư mục
import numpy as np          # Tính toán ma trận, xử lý pixel
import tensorflow as tf     # Tạo dataset ảnh và các layer augmentation



CURRENT_FILE_PATH = os.path.abspath(__file__)

SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)

PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_ROOT_NAME = "New Plant Diseases Dataset(Augmented)"

DATA_DIR = os.path.join(PROJECT_ROOT, "data", DATA_ROOT_NAME, DATA_ROOT_NAME)

TRAIN_DIR = os.path.join(DATA_DIR, "train")

# Đường dẫn thư mục train đã làm sạch (ảnh dùng để huấn luyện)
TRAIN_CLEAN_DIR = os.path.join(PROJECT_ROOT, "data", "train_clean1")

# Thư mục lưu ảnh quá sáng để kiểm tra
BRIGHT_DIR = os.path.join(PROJECT_ROOT, "data", "bright_images")

# Thư mục lưu ảnh đã chỉnh sửa (fixed) để kiểm tra
FIXED_DIR = os.path.join(PROJECT_ROOT, "data", "fixed_images")

# Kích thước ảnh chuẩn dùng cho mô hình (width, height)
IMG_SIZE = (224, 224)

# Kích thước batch khi tạo dataset
BATCH_SIZE = 32

# In ra để kiểm tra cấu hình đường dẫn train gốc
print(f"Cấu hình đường dẫn TRAIN_DIR: {TRAIN_DIR}")


# Ngưỡng pixel (0–255) để coi là pixel sáng
BRIGHT_PIXEL_THRESHOLD = 245

# Tỷ lệ pixel sáng trên toàn ảnh để coi là ảnh quá sáng (ví dụ >30%)
BRIGHT_RATIO_THRESHOLD = 0.3

# Ngưỡng độ biến thiên Laplacian để coi là ảnh mờ (thấp => mờ)
BLUR_THRESHOLD = 120

#Hàm Xử lý ảnh

def is_very_bright(img):
    # Trả về True nếu ảnh có tỷ lệ pixel sáng vượt ngưỡng quy định
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Tính tỷ lệ số pixel có giá trị >= ngưỡng sáng trên tổng số pixel
    ratio = np.sum(gray >= BRIGHT_PIXEL_THRESHOLD) / gray.size
  
    return ratio > BRIGHT_RATIO_THRESHOLD

def is_medium_bright(img):
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ratio = np.sum(gray >= BRIGHT_PIXEL_THRESHOLD) / gray.size
    return 0.05 < ratio <= BRIGHT_RATIO_THRESHOLD

def is_blurry(img, thresh=BLUR_THRESHOLD):
    # Trả về True nếu ảnh bị mờ dựa trên độ biến thiên Laplacian
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.Laplacian đo độ sắc nét theo biên; giá trị nhỏ => ít biên => mờ
    return cv2.Laplacian(gray, cv2.CV_64F).var() < thresh

def fix_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Giảm nhẹ độ sáng tổng thể để tránh cháy sáng
    l = np.clip(l * 0.85, 0, 255).astype(np.uint8)
    # Tạo bộ cân bằng histogram cục bộ (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l = clahe.apply(l)
    # Gộp lại các kênh và chuyển về BGR
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    # Làm mịn nhẹ bằng Gaussian Blur
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    # Tăng độ sắc nét bằng cách cộng trộn ảnh gốc với ảnh mịn
    img = cv2.addWeighted(img, 1.3, blur, -0.3, 0)
    return img


# 3. HÀM LÀM SẠCH DỮ LIỆU (DATA CLEANING)

def clean_data():
    # Tạo các thư mục đầu ra nếu chưa tồn tại
    os.makedirs(TRAIN_CLEAN_DIR, exist_ok=True)
    os.makedirs(BRIGHT_DIR, exist_ok=True)
    os.makedirs(FIXED_DIR, exist_ok=True)
    
    # Kiểm tra sự tồn tại của thư mục train gốc; nếu không có thì dừng
    if not os.path.exists(TRAIN_DIR):
        print(f"Lỗi: Không tìm thấy thư mục gốc '{TRAIN_DIR}'")
        return

    # Thông báo bắt đầu quá trình làm sạch dữ liệu
    print("Đang chạy quá trình lọc và sửa ảnh...")
    # Biến thống kê số lượng ảnh theo từng loại xử lý
    stats = {"too_bright": 0, "fixed": 0, "normal": 0}

    # Duyệt qua từng class (thư mục con) trong train gốc
    for cls in os.listdir(TRAIN_DIR):
        cls_path = os.path.join(TRAIN_DIR, cls)
        # Bỏ qua nếu không phải thư mục (tránh file lẻ)
        if not os.path.isdir(cls_path):
            continue

        # Chuẩn bị các thư mục đầu ra cho class hiện tại
        out_bright = os.path.join(BRIGHT_DIR, cls)
        out_fixed  = os.path.join(FIXED_DIR, cls)
        out_clean  = os.path.join(TRAIN_CLEAN_DIR, cls)

        # Tạo thư mục nếu chưa có
        os.makedirs(out_bright, exist_ok=True)
        os.makedirs(out_fixed, exist_ok=True)
        os.makedirs(out_clean, exist_ok=True)

        # Duyệt qua từng ảnh trong class
        for img_name in os.listdir(cls_path):
            # Chỉ xử lý file ảnh (jpg/jpeg/png)
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            # Đường dẫn file ảnh nguồn
            src = os.path.join(cls_path, img_name)
            # Đọc ảnh bằng OpenCV (BGR)
            img = cv2.imread(src)
            # Bỏ qua nếu đọc lỗi
            if img is None:
                continue

            # Phân loại và xử lý theo chất lượng ảnh
            if is_very_bright(img):
                # Ảnh quá sáng: sao chép sang thư mục bright để kiểm tra
                shutil.copy(src, os.path.join(out_bright, img_name))
                stats["too_bright"] += 1
            elif is_medium_bright(img) or is_blurry(img):
                # Ảnh sáng vừa/mờ: sửa bằng fix_image rồi lưu vào fixed và clean
                img_fixed = fix_image(img)
                cv2.imwrite(os.path.join(out_fixed, img_name), img_fixed)
                cv2.imwrite(os.path.join(out_clean, img_name), img_fixed)
                stats["fixed"] += 1
            else:
                # Ảnh bình thường: sao chép nguyên bản sang clean
                shutil.copy(src, os.path.join(out_clean, img_name))
                stats["normal"] += 1

    # In thống kê cuối cùng về số ảnh dùng để train
    print(f"Tổng ảnh dùng để train: {stats['fixed'] + stats['normal']} (trong {TRAIN_CLEAN_DIR})")


# 4. HÀM LOAD DỮ LIỆU (DATA LOADING)


def load_datasets():
    print("\nĐang nạp dữ liệu...")

    if not os.path.exists(TRAIN_CLEAN_DIR):
        print("Lỗi: Chưa thấy thư mục ảnh sạch. Đang tự động chạy clean_data()...")
        clean_data()

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_CLEAN_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_CLEAN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    class_names = train_ds.class_names
    print(f"Đã tìm thấy {len(class_names)} lớp bệnh.")

    # Test dataset: chỉ kiểm tra số ảnh, không tạo dataset từ directory
    TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test", "test")
    test_ds = None
    if os.path.exists(TEST_DIR):
        test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if test_files:
            print(f"Thư mục Test có {len(test_files)} ảnh, sẽ dùng cho demo trực quan.")
        else:
            print("Thư mục Test tồn tại nhưng không có ảnh hợp lệ")
    else:
        print(f"Cảnh báo: Không tìm thấy thư mục Test tại '{TEST_DIR}'")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names



# 5. HÀM DATA AUGMENTATION (TĂNG CƯỜNG DỮ LIỆU)


def get_data_augmentation():
    
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")



if __name__ == "__main__":
   
    clean_data()