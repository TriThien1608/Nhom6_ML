import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


def integrated_gradients(model, image, target_class, baseline=None, steps=50):
    """
    Sử dụng Integrated Gradients để giải thích dự đoán của mô hình.

    Tham số
    ----------
    model : tf.keras.Model
        Mô hình TensorFlow/Keras đã được huấn luyện.
    image : tf.Tensor
        Ảnh đầu vào để giải thích (shape: (1, H, W, C)).
    target_class : int
        Lớp mục tiêu để giải thích dự đoán.
    baseline : tf.Tensor, optional
        Ảnh cơ sở để so sánh (mặc định là ảnh đen).
    steps : int
        Số bước để xấp xỉ tích phân (mặc định là 50).

    Trả về
    -------
    ig : tf.Tensor
        Tích phân Gradients (shape: (1, H, W, C)) đại diện cho tầm quan trọng của mỗi pixel.

    Ghi chú
    -----
    Integrated Gradients được tính toán theo công thức:
        IG = (x - x') * ∫ (∂F(x') / ∂x') dα

    Trong đó:   x là ảnh đầu vào, 
                x' là ảnh cơ sở, 
                F là hàm dự đoán của mô hình
                α là tham số tích phân từ 0 đến 1.
    """
    if baseline is None:
        baseline = tf.zeros_like(image)

    alphas = tf.linspace(0.0, 1.0, steps)
    ig = tf.zeros_like(image)

    for alpha in alphas:
        with tf.GradientTape() as tape:
            interpolated = baseline + alpha * (image - baseline)
            tape.watch(interpolated)
            preds = model(interpolated)
            loss = preds[:, target_class]

        grads = tape.gradient(loss, interpolated)
        ig += grads

    ig /= steps
    ig *= (image - baseline)
    return ig


def visualize_ig(ig, img_path):
    """
    Hiển thị heap map của Integrated Gradients trên ảnh gốc.

    Tham số
    ----------
    ig : tf.Tensor
        Tích phân Gradients (shape: (1, H, W, C)).
    img_path : str
        Đường dẫn đến ảnh gốc.
    
    Trả về
    -------
    None
    
    Ghi chú
    -----
    - Chuẩn hóa và làm mờ tích phân Gradients.
    - Tạo heatmap và chồng lên ảnh gốc để trực quan hóa tầm quan trọng của các pixel.
    """
    ig = tf.reduce_mean(tf.abs(ig), axis=-1)[0].numpy()
    ig = ig / (ig.max() + 1e-8)
    ig = cv2.GaussianBlur(ig, (7, 7), 0)

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    ig = cv2.resize(ig, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * ig), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Integrated Gradients")
    plt.show()
