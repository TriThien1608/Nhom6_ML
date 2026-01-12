import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import random
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


try:
    from preprocessing import IMG_SIZE
except ImportError:
    print("Không tìm thấy preprocessing.py, dùng kích thước mặc định (224,224)")
    IMG_SIZE = (224, 224)


# 2. VẼ BIỂU ĐỒ HUẤN LUYỆN

def plot_history(history):
    """
    Vẽ biểu đồ quá trình huấn luyện:
    - Training vs Validation Accuracy
    - Training vs Validation Loss
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 6))

    # Biểu đồ độ chính xác
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linestyle='--')
    plt.legend(loc='lower right')
    plt.title('Độ chính xác (Accuracy)')
    plt.xlabel('Epochs')
    plt.grid(True, alpha=0.3)

    # Biểu đồ hàm mất mát
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss', linestyle='--')
    plt.legend(loc='upper right')
    plt.title('Hàm mất mát (Loss)')
    plt.xlabel('Epochs')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 3. ĐÁNH GIÁ MÔ HÌNH

def evaluate_model(model, test_ds, class_names):
    """
    Đánh giá mô hình trên tập Test:
    - In classification report (precision, recall, f1-score)
    - Tính accuracy tổng thể
    - Vẽ confusion matrix trực quan
    """
    print("Đang dự đoán trên tập Test...")
    y_true, y_pred = [], []

    # Duyệt qua từng batch trong test dataset
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # In báo cáo chi tiết theo từng class
    print("\nClassification Report")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Tính accuracy tổng thể
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

    # Vẽ confusion matrix
    plt.figure(figsize=(20, 18))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Acc: {acc*100:.1f}%)')
    plt.xticks(rotation=90)
    plt.show()

# ==========================================
# 4. DEMO TRỰC QUAN
# ==========================================
def run_demo(model, class_names, demo_folder_path=None):
    """
    Chạy demo trực quan:
    - Lấy ngẫu nhiên 5 ảnh từ thư mục test
    - Dự đoán class và hiển thị kết quả trên ảnh
    - Màu chữ xanh cho cây khỏe mạnh, đỏ cho cây bệnh
    """
    # Nếu không truyền đường dẫn test thì dùng mặc định
    if demo_folder_path is None:
        demo_folder_path = os.path.join("..", "data", "New Plant Diseases Dataset(Augmented)",
                                        "New Plant Diseases Dataset(Augmented)", "test")

    if not os.path.exists(demo_folder_path):
        print(f"Không tìm thấy folder: {demo_folder_path}")
        return

    # Thu thập tất cả đường dẫn ảnh trong thư mục test
    all_image_paths = []
    for root, dirs, files in os.walk(demo_folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))

    if not all_image_paths:
        print("Folder test rỗng hoặc không chứa ảnh!")
        return

    # Lấy ngẫu nhiên tối đa 5 ảnh để demo
    random_samples = random.sample(all_image_paths, min(len(all_image_paths), 5))
    plt.figure(figsize=(20, 6))

    for i, img_path in enumerate(random_samples):
        # Load ảnh và resize về IMG_SIZE
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)

        # Dự đoán class
        pred = model.predict(img_arr, verbose=0)
        score = np.max(pred[0])
        predicted_index = np.argmax(pred[0])
        label_name = class_names[predicted_index]

        # Tách tên cây và bệnh từ nhãn
        parts = label_name.split("___")
        if len(parts) > 1:
            plant_name = parts[0]
            disease_part = parts[1].replace("_", " ")
        else:
            plant_name = label_name
            disease_part = ""

        # Xác định cây khỏe mạnh hay bệnh
        is_healthy = "healthy" in label_name.lower()
        text_color = 'green' if is_healthy else 'red'

        # Chuẩn bị text hiển thị
        if is_healthy:
            status_text = f"Cây: {plant_name}\nTrạng thái: KHỎE MẠNH\n(Tin cậy: {score*100:.1f}%)"
        else:
            status_text = f"Cây: {plant_name}\nBệnh: {disease_part}\n(Tin cậy: {score*100:.1f}%)"

        # Vẽ ảnh và kết quả dự đoán
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(status_text, color=text_color, fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.show()