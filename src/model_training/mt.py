import tensorflow as tf
from Mylib import tf_myclasses, tf_myfuncs, myfuncs
import os
import time
from src.utils import classes


def load_train_ds_and_val_ds(train_ds_path, val_ds_path):
    train_ds = tf.data.Dataset.load(train_ds_path)
    val_ds = tf.data.Dataset.load(val_ds_path)

    return train_ds, val_ds


def create_model_from_layers_optimizer(
    model, model_optimizer, loss, metrics, image_size, channels
):
    input_layer = tf.keras.Input(
        (image_size, image_size, channels)
    )  # Tạo sẵn layer Input trước
    x = input_layer

    for layer in model:
        x = layer(x)

    model = tf.keras.Model(inputs=input_layer, outputs=x)

    model.compile(
        optimizer=model_optimizer,
        loss=loss,
        metrics=metrics,
    )

    return model


def create_and_save_models_before_training(
    model_training_path,
    model_indices,
    models,
    optimizer,
    loss,
    metrics,
    image_size,
    channels,
):
    for model_index, model in zip(model_indices, models):
        model_path = f"{model_training_path}/{model_index}.keras"
        model_optimizer = tf_myfuncs.copy_one_optimizer(optimizer)
        model = create_model_from_layers_optimizer(
            model, model_optimizer, loss, metrics, image_size, channels
        )

        # Save model
        model.save(model_path)


def create_callbacks(callbacks, model_path, target_score, model_checkpoint_monitor):
    callbacks = [tf_myfuncs.copy_one_callback(callback) for callback in callbacks]

    callbacks = [
        classes.CustomisedModelCheckpoint(
            filepath=model_path,
            monitor=model_checkpoint_monitor,
            indicator=target_score,
        ),
    ] + callbacks

    return callbacks


def train_and_save_models(
    model_training_path,
    model_indices,
    num_models,
    train_ds,
    val_ds,
    epochs,
    callbacks,
    model_name,
    target_score,
    model_checkpoint_monitor,
    scoring,
    plot_dir,
):
    tf.config.run_functions_eagerly(True)  # Bật eager execution
    tf.data.experimental.enable_debug_mode()  # Bật chế độ eager cho tf.data

    print(
        f"===========Bắt đầu train {num_models} model name = {model_name}===============\n"
    )

    start_time = time.time()  # Bắt đầu tính thời gian train model
    for model_index in zip(model_indices):
        # Load model
        model_path = os.path.join(model_training_path, f"{model_index}.keras")
        model = tf.keras.models.load_model(model_path)

        # Create callbacks cho model
        model_callbacks = create_callbacks(
            callbacks, model_path, target_score, model_checkpoint_monitor
        )

        # Train model
        print(f"Bắt đầu train model index {model_name} - {model_index}")
        history = model.fit(
            train_ds,
            epochs=epochs,
            verbose=1,
            validation_data=val_ds,
            callbacks=model_callbacks,
        )
        print(f"Kết thúc train model index {model_name} - {model_index}")
        num_epochs_before_stopping = len(history.history["loss"])

        # Đánh giá model
        ## Load model với epoch tốt nhất được lưu bởi CustomisedModelCheckpoint !!!!
        best_model_among_epochs = tf.keras.models.load_model(model_path)
        train_scoring = best_model_among_epochs.evaluate(train_ds, verbose=0)[
            1
        ]  # chỉ số đầu luôn là loss, sau đó là scoring
        val_scoring = best_model_among_epochs.evaluate(val_ds, verbose=0)[1]

        ## In kết quả
        print("Kết quả của model")
        print(
            f"Model index {model_name} - {model_index}\n -> Train {scoring}: {train_scoring}, Val {scoring}: {val_scoring}, Epochs: {num_epochs_before_stopping}/{epochs}\n"
        )

        # Lưu dữ liệu để vẽ biểu đồ
        model_name_in_plot = f"{model_name}_{model_index}"

        myfuncs.save_python_object(
            os.path.join(plot_dir, f"{model_name_in_plot}.pkl"),
            (model_name_in_plot, train_scoring, val_scoring),
        )
    all_model_end_time = time.time()  # Kết thúc tính thời gian train model
    true_all_models_train_time = (all_model_end_time - start_time) / 60

    print(f"Thời gian chạy tất cả: {true_all_models_train_time} (mins)")
    print(
        f"===========Kết thúc train {num_models} model name = {model_name}===============\n"
    )
