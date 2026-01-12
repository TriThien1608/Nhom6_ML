"""
main.py
Main experimental pipeline.
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import clean_dataset
from eda import evaluate_model
from explain import integrated_gradients, visualize_ig

from models import (
    MobileNetV3_HuaKhanhDuy,
    EfficientNetB0_DinhVanNghia,
    ResNet50_LeTriThien
)

# =========================
# CONFIG
# =========================
RAW_DATA_DIR = "../data/train"
CLEAN_DATA_DIR = "../data/data_clean"
VAL_DIR = "../data/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2


# =========================
# DATA PIPELINE
# =========================
def prepare_datasets():
    """
    Load datasets AFTER preprocessing.
    """
    datagen = ImageDataGenerator(
        validation_split=VAL_SPLIT,
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input
    )

    train_ds = datagen.flow_from_directory(
        CLEAN_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset="training",
        class_mode="sparse",
        shuffle=True
    )

    val_ds = datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset="validation",
        class_mode="sparse",
        shuffle=False
    )

    return train_ds, val_ds


# =========================
# MAIN
# =========================
def main():
    print("Step 1: Preprocessing dataset...")
    clean_dataset(RAW_DATA_DIR, CLEAN_DATA_DIR)

    print("Step 2: Loading datasets...")
    train_ds, val_ds = prepare_datasets()
    num_classes = train_ds.num_classes

    models = {
        "MobileNetV3_HuaKhanhDuy": MobileNetV3_HuaKhanhDuy,
        "EfficientNetB0_DinhVanNghia": EfficientNetB0_DinhVanNghia,
        "ResNet50_LeTriThien": ResNet50_LeTriThien
    }

    for name, module in models.items():
        print(f"\n🚀 Training {name}")

        if name == "MobileNetV3_HuaKhanhDuy":
            model, backbone = module.build_model(num_classes)
            module.train_model(model, train_ds, val_ds)
            module.fine_tune_model(model, backbone, train_ds, val_ds)
        else:
            model = module.build_model(num_classes)
            module.train_model(model, train_ds, val_ds)
            module.fine_tune_model(model, train_ds, val_ds)

        print(f"\n📊 Evaluating {name}")
        evaluate_model(model, val_ds, name)

        # Giải thích với Integrated Gradients cho một ảnh mẫu
        img_path = "path_to_sample_image.jpg"  # Thay đổi đường dẫn ảnh mẫu
        sample_img, sample_label = val_ds[0][0][0:1], val_ds[0][1][0]
        ig = integrated_gradients(model, sample_img, int(sample_label))
        visualize_ig(ig, img_path)

    print("\nHoàn thành tất cả các bước!")


if __name__ == "__main__":
    main()
