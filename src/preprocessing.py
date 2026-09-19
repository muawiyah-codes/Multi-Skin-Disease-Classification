import os
import hashlib
from pathlib import Path
import tensorflow as tf
from PIL import Image

def get_file_hash(filepath, chunk_size=65536):
    """Calculates SHA256 hash of a file for duplicate detection."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()

def parse_image(filepath, label, img_size=(224, 224)):
    """Decodes JPEG/PNG image, resizes to target size, and returns float32 tensor."""
    image_raw = tf.io.read_file(filepath)
    image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, img_size, method='bilinear')
    return tf.cast(image, tf.float32), label

def create_tf_dataset(df, class_map, batch_size=32, img_size=(224, 224), is_training=False):
    """Constructs high-performance tf.data.Dataset pipeline."""
    filepaths = df['filepath'].values
    labels = [class_map[c] for c in df['class'].values]
    
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=42)
        
    dataset = dataset.map(lambda x, y: parse_image(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
