import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# 1. LIÊN KẾT MODULE

# Import các hàm từ preprocessing.py để đồng bộ cấu hình và augmentation
# Nếu không tìm thấy file preprocessing.py thì định nghĩa mặc định
try:
    from preprocessing import get_data_augmentation, IMG_SIZE
except ImportError:
    print("Không tìm thấy preprocessing.py, dùng cấu hình mặc định.")
    def get_data_augmentation(): return tf.keras.Sequential()
    IMG_SIZE = (224, 224)


# 2. HÀM XÂY DỰNG MÔ HÌNH RESNET50 (TRANSFER LEARNING)

def build_model(num_classes):
    """
    Xây dựng kiến trúc model dựa trên ResNet50 (Transfer Learning).
    
    Args:
        num_classes (int): Số lượng lớp đầu ra (số loại bệnh).
        
    Returns:
        model: tf.keras.Model đã được xây dựng kiến trúc (chưa compile optimizer).
    """
    print(f"Khởi tạo Model ResNet50 cho {num_classes} phân lớp...")
    print("   - Chế độ: Transfer Learning (Đóng băng Base Model)")

    # 1. TẢI BASE MODEL (Pre-trained on ImageNet)
    # include_top=False: bỏ lớp Dense 1000 class cuối cùng của ImageNet
    base_model = ResNet50(
        input_shape=IMG_SIZE + (3,),  # (224, 224, 3)
        include_top=False,
        weights="imagenet"
    )
    
    # Đóng băng toàn bộ các lớp của Base Model để không train lại
    base_model.trainable = False

    # 2. XÂY DỰNG KIẾN TRÚC (Functional API)
    inputs = Input(shape=IMG_SIZE + (3,), name="input_layer")

    # Bước A: Data Augmentation (tăng cường dữ liệu ngay trong model)
    # Lợi ích: tận dụng GPU để xoay/lật ảnh trong quá trình train
    data_augmentation = get_data_augmentation()
    x = data_augmentation(inputs)

    # Bước B: Preprocessing (chuẩn hóa dữ liệu theo chuẩn ResNet)
    # Chuyển pixel từ [0-255] sang định dạng zero-centered (chuẩn caffe)
    x = preprocess_input(x)

    # Bước C: Truyền qua Base Model
    # training=False: giữ BatchNormalization ở chế độ inference
    x = base_model(x, training=False)

    # Bước D: Phần đầu mới (Custom Head)
    x = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
    x = layers.Dropout(0.5)(x)  # Dropout để giảm overfitting
    x = layers.Dense(256, activation='relu', name="dense_256")(x)
    x = layers.Dropout(0.2)(x)

    # Lớp Output: số node bằng số lớp bệnh, activation softmax
    outputs = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)

    # 3. KẾT HỢP
    model = models.Model(inputs, outputs, name="ResNet50_TransferLearning")
    
    return model


# 3. KHỐI TEST CODE (CHẠY KHI DEBUG FILE NÀY)

if __name__ == "__main__":
    # Test thử build model với 10 classes giả định
    model = build_model(num_classes=10)
    model.summary()
    print("Build model thành công!")