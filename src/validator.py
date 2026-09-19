import tensorflow as tf

def build_lightweight_validator_model(input_shape=(224, 224, 3)):
    """
    Builds lightweight MobileNetV2 binary OOD Image Validator model.
    Output: Sigmoid probability (0 = INVALID, 1 = VALID).
    """
    inputs = tf.keras.Input(shape=input_shape, name="validator_input")
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False
    
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="validator_output")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="SkinDomain_Validator")
