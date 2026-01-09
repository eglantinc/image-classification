import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_data(data_dir, img_height, img_width, batch_size):
    """
    Load training and validation datasets from a directory.
    80% training, 20% validation split.
    """
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    return train_ds, val_ds


def build_custom_cnn(img_height, img_width, num_classes):
    """
    Builds a custom CNN with 4 convolutional blocks,
    BatchNorm, Dropout, and fully-connected layers.
    """
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.5),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def compile_and_train(model, train_ds, val_ds, epochs, learning_rate):
    """
    Compiles with Adam optimizer and sparse categorical crossentropy loss.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    return history


def evaluate_model(model, val_ds):
    """
    Print loss and accuracy on the validation set.
    """
    loss, accuracy = model.evaluate(val_ds)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    return loss, accuracy


def calculate_metrics(model, val_ds):
    """
    Calculate accuracy, F1-score, precision and recall on the validation set.
    """
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        predictions = model.predict(images)
        predicted_labels = np.argmax(predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    print("Accuracy:", acc)
    print("F1-Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    return acc, f1, precision, recall


def write_report(dataset_name, acc, f1, precision, recall):
    """
    Save metrics to a text file per dataset (in 'report' folder).
    """
    os.makedirs("report", exist_ok=True)
    report_path = f"report/report_{dataset_name}.txt"
    with open(report_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write("-" * 40 + "\n")
    print(f"[INFO] Metrics written to {report_path}")


def plot_cnn_metrics(metrics_dict, dataset_name="Result", save_folder="report"):
    """
    Generates and saves a bar chart of metrics (accuracy, precision, recall, F1-score).
    """
    os.makedirs(save_folder, exist_ok=True)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [metrics_dict[m] for m in metrics]
    colors = ['#FFB7C5', '#C3B1E1', '#B5EAD7', '#FFF1B7']

    plt.figure(figsize=(6, 4))
    bars = plt.bar(metrics, values, color=colors)
    plt.ylim(0, 1)
    plt.title(f"Classification Metrics for {dataset_name}")
    plt.ylabel("Score")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_folder, f"metrics_{dataset_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Chart saved to {save_path}")


def plot_learning_curves(history, dataset_name="Result", save_folder="learning_curves"):
    """
    Generates and saves learning curves (accuracy and loss) in pastel colors.
    """
    os.makedirs(save_folder, exist_ok=True)
    pink = '#FFB7C5'
    lavender = '#C3B1E1'
    mint = '#B5EAD7'
    yellow = '#FFF1B7'
    purple = '#B39EB5'

    plt.figure(figsize=(6,4))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color=pink, marker='o', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color=lavender, marker='o', linewidth=2)
    plt.title(f"Learning Curve - Accuracy ({dataset_name})", fontsize=13, color=purple)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(alpha=0.15)
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(save_folder, f"accuracy_curve_{dataset_name}.png")
    plt.savefig(acc_path, dpi=180)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='Train Loss', color=mint, marker='o', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color=yellow, marker='o', linewidth=2)
    plt.title(f"Learning Curve - Loss ({dataset_name})", fontsize=13, color=purple)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(alpha=0.15)
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(save_folder, f"loss_curve_{dataset_name}.png")
    plt.savefig(loss_path, dpi=180)
    plt.close()

    print(f"[INFO] Learning curves saved in {save_folder}:")
    print(f"       - {os.path.basename(acc_path)}")
    print(f"       - {os.path.basename(loss_path)}")


def main():
    data_dir = "./Wildfire" 
    img_height = 150
    img_width = 150
    batch_size = 32
    epochs = 15
    learning_rate = 1e-3

    dataset_name = os.path.basename(os.path.normpath(data_dir))
    train_ds, val_ds = load_data(data_dir, img_height, img_width, batch_size)
    class_names = train_ds.class_names
    print("Detected classes:", class_names)
    print(f"Number of classes: {len(class_names)}")

    model = build_custom_cnn(img_height, img_width, len(class_names))
    model.summary()

    history = compile_and_train(model, train_ds, val_ds, epochs, learning_rate)
    plot_learning_curves(history, dataset_name=dataset_name)

    evaluate_model(model, val_ds)
    acc, f1, precision, recall = calculate_metrics(model, val_ds)
    write_report(dataset_name, acc, f1, precision, recall)
    metrics_dict = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }
    plot_cnn_metrics(metrics_dict, dataset_name=dataset_name)

if __name__ == "__main__":
    main()