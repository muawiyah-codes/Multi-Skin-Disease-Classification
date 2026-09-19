import tensorflow as tf

def build_data_augmentation_layer():
    """Builds GPU-accelerated Keras Sequential Data Augmentation pipeline."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42, name="aug_flip"),
        tf.keras.layers.RandomRotation(0.15, fill_mode="reflect", seed=42, name="aug_rotation"),
        tf.keras.layers.RandomZoom((-0.1, 0.1), fill_mode="reflect", seed=42, name="aug_zoom"),
        tf.keras.layers.RandomContrast(0.10, seed=42, name="aug_contrast")
    ], name="data_augmentation_pipeline")

def build_transfer_learning_model(num_classes, input_shape=(224, 224, 3)):
    """
    Builds Transfer Learning Architecture using EfficientNetV2B0 base + Custom Classification Head.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # Data Augmentation (Training active only)
    aug = build_data_augmentation_layer()
    x = aug(inputs)
    
    # Pre-trained Base Model
    base_model = tf.keras.applications.EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        include_preprocessing=True
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batch_norm")(x)
    x = tf.keras.layers.Dense(256, activation="swish", name="head_dense_256")(x)
    x = tf.keras.layers.Dropout(0.3, seed=42, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="output_softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SkinDisease_EfficientNetV2B0")
    return model, base_model
