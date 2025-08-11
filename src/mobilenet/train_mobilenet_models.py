import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

BASE_DIR = r"D:\CROP DISEASE PREDICTION MODEL USING DEEP LEARNING"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "processed_potato", "train")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "processed_potato", "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenet_model.h5")
CONF_MATRIX_PATH = os.path.join(BASE_DIR, "results", "mobilenet","confusion_matrix.png")
REPORT_PATH = os.path.join(BASE_DIR, "results", "mobilenet","classification_report.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

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

class_labels = list(test_generator.class_indices.keys())

def create_mobilenetv2_model(input_shape=(224, 224, 3), learning_rate=LEARNING_RATE):
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))
    
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_mobilenetv2_model()

print("\n Starting MobileNetV2 training...\n")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1
)

model.save(MODEL_PATH)
print(f"\n Model saved at: {MODEL_PATH}")

train_loss, train_acc = model.evaluate(train_generator, verbose=0)
test_loss, test_acc = model.evaluate(test_generator, verbose=0)

print(f"\n Final Training Accuracy: {train_acc * 100:.2f}%")
print(f" Final Test Accuracy: {test_acc * 100:.2f}%")
print(f" Training Loss: {train_loss:.4f}")
print(f" Test Loss: {test_loss:.4f}")

print("\n Generating evaluation reports on test set...")

y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)
with open(REPORT_PATH, "w") as f:
    f.write("Classification Report\n")
    f.write("=====================\n\n")
    f.write(report)
print(f"\n Classification report saved at: {REPORT_PATH}")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('MobileNetV2 Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.show()

print(f"\n Confusion matrix saved at: {CONF_MATRIX_PATH}")
