import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
"""
MobileNetV3_HuaKhanhDuy.py
Proposed model based on MobileNetV3Large with fine-tuning strategy
"""

def build_model(num_classes):
    """
    Build MobileNetV3-based classification model.

    Args:
        num_classes (int): Number of output classes

    Returns:
        model (tf.keras.Model): Compiled model
        base_model (tf.keras.Model): Backbone for fine-tuning
    """
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def train_model(model, train_ds, valid_ds, epochs=20):
    """
    Train model with frozen backbone.
    """
    callbacks = [
        ModelCheckpoint("MobileNetV3_HuaKhanhDuy_best.keras",
                        save_best_only=True),
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.3)
    ]

    return model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )


def fine_tune_model(model, base_model, train_ds, valid_ds, epochs=10):
    """
    Fine-tune các lớp.
    """
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    for layer in base_model.layers[-30:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return train_model(model, train_ds, valid_ds, epochs)