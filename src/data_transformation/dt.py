import tensorflow as tf
from Mylib import tf_myfuncs
import os


def load_data(train_ds_path):
    train_ds = tf.data.Dataset.load(train_ds_path)

    return train_ds


def create_transformer(transformer, image_size, channels):
    input_layer = tf.keras.Input(
        (image_size, image_size, channels)
    )  # Tạo sẵn layer Input trước
    x = input_layer

    for layer in transformer:
        x = layer(x)

    # Tạo model thôi, chưa cần compile
    model = tf.keras.Model(inputs=input_layer, outputs=x)

    return model


def transform_and_save_data(data_transformation_path, transformer, train_ds):
    # Transform data
    train_ds_transformed = train_ds.map(lambda x, y: (transformer(x, training=True), y))

    # Cache và save model
    train_ds_transformed = tf_myfuncs.cache_prefetch_tfdataset_2(train_ds_transformed)
    train_ds_transformed.save(os.path.join(data_transformation_path, "train_ds"))
