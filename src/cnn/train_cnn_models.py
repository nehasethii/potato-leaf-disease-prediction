import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = r"D:\CROP DISEASE PREDICTION MODEL USING DEEP LEARNING"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "processed_potato", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "processed_potato", "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.h5")
CONF_MATRIX_PATH = os.path.join(BASE_DIR, "results","cnn", "confusion_matrix.png")
REPORT_PATH = os.path.join(BASE_DIR, "results","cnn", "classification_report.txt")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.0001
NUM_CLASSES = 3

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

def create_cnn_model(input_shape=(224, 224, 3), learning_rate=0.0001):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_cnn_model()

print("\n Starting training...\n")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)

model.save(MODEL_PATH)
print(f"\n Model saved as: {MODEL_PATH}")

train_loss, train_acc = model.evaluate(train_generator, verbose=0)
test_loss, test_acc = model.evaluate(test_generator, verbose=0)

print(f"\n Final Training Accuracy: {train_acc * 100:.2f}%")
print(f" Final Test Accuracy: {test_acc * 100:.2f}%")
print(f" Final Training Loss: {train_loss:.4f}")
print(f" Final Test Loss: {test_loss:.4f}")

y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

class_labels = list(test_generator.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=class_labels)
print("\n Classification Report:")
print(report)

with open(REPORT_PATH, "w") as f:
    f.write("Classification Report\n")
    f.write("=====================\n\n")
    f.write(report)
print(f"\n Classification report saved at: {REPORT_PATH}")

print(" Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.show()
print(f"\n Confusion matrix saved at: {CONF_MATRIX_PATH}")
