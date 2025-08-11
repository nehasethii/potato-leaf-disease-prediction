import os
import random
from PIL import Image, ImageEnhance
import numpy as np

DATASET_PATH = r"D:\CROP DISEASE PREDICTION MODEL USING DEEP LEARNING\dataset\potato"
PROCESSED_PATH = r"D:\CROP DISEASE PREDICTION MODEL USING DEEP LEARNING\dataset\processed_potato"
IMAGE_SIZE = (224, 224)
TRAIN_RATIO = 0.8 

def create_dirs():
    """Create train/test directories for each class"""
    for split in ["train", "test"]:
        for class_name in os.listdir(DATASET_PATH):
            os.makedirs(os.path.join(PROCESSED_PATH, split, class_name), exist_ok=True)

def augment_image(img):
    """Apply realistic augmentations for plant leaves"""
    
    img = img.rotate(random.uniform(-15, 15))
    
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    return img

def balance_classes():
    """Ensure all classes have equal number of samples"""
    class_counts = {}
    for class_name in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_name)
        class_counts[class_name] = len(os.listdir(class_path))
    
    max_count = max(class_counts.values())
    
    for class_name, count in class_counts.items():
        if count < max_count:
            class_path = os.path.join(DATASET_PATH, class_name)
            images = os.listdir(class_path)
            
            while len(images) < max_count:
                img_path = os.path.join(class_path, random.choice(images))
                img = Image.open(img_path)
                img = augment_image(img)
                new_name = f"aug_{random.randint(1000,9999)}.jpg"
                img.save(os.path.join(class_path, new_name))
                images.append(new_name)

def preprocess_dataset():
    """Main preprocessing pipeline"""
    print("🔍 Preprocessing dataset...")
    
    balance_classes()
    create_dirs()
    for class_name in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)
        train_size = int(len(images) * TRAIN_RATIO)

        for i, img_name in enumerate(images):
            src_path = os.path.join(class_path, img_name)
            img = Image.open(src_path).resize(IMAGE_SIZE)
            
            split = "train" if i < train_size else "test"
            dest_path = os.path.join(PROCESSED_PATH, split, class_name, img_name)
            img.save(dest_path)

    print(f" Preprocessing complete. Data saved to:\n{PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess_dataset()