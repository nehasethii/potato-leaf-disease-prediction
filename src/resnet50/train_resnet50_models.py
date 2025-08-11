import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint # type: ignore

from utils.load_data import train_generator, val_generator, test_generator, class_weights

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIGURATION
LEARNING_RATE = 0.001
EPOCHS = 50
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 3

# Output directories
BASE_DIR = r"D:\CROP DISEASE PREDICTION MODEL USING DEEP LEARNING"
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
CONF_MATRIX_PATH = os.path.join(BASE_DIR, "results" , "resnet50" , "confusion_matrix.png")
REPORT_PATH = os.path.join(BASE_DIR, "results" , "resnet50" , "classification_report.txt")

# ResNet50 model
def create_resnet50_model(input_shape, learning_rate):
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_resnet50_model(INPUT_SHAPE, LEARNING_RATE)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'resnet50_best_model.h5'), monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Training
print("\n Starting ResNet50 training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# Final model
final_model_path = os.path.join(MODEL_DIR, 'resnet50_final_model.h5')
model.save(final_model_path)
print(f"\n Final model saved at: {final_model_path}")

# Accuracy/Loss metrics
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

print(f"\n Final Training Accuracy: {train_acc:.4f}")
print(f" Final Validation Accuracy: {val_acc:.4f}")
print(f" Final Training Loss: {train_loss:.4f}")
print(f" Final Validation Loss: {val_loss:.4f}")

# Evaluation on Test Set
print("\n Evaluating on test set...")
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

class_labels = list(test_generator.class_indices.keys())

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_labels)
print("\n Classification Report:")
print(report)

with open(REPORT_PATH, "w") as f:
    f.write("Classification Report\n")
    f.write("=====================\n\n")
    f.write(report)
print(f"\n Classification report saved at: {REPORT_PATH}")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.show()
print(f"\n Confusion matrix saved at: {CONF_MATRIX_PATH}")
