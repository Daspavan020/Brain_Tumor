import os
import random
import cv2
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

MODEL_PATH = "files_h5/best_model.h5"
DATA_DIR = "data"
SAVE_DIR = "results_h5"

IMG_H, IMG_W = 240, 240
NUM_CLASSES = 3

os.makedirs(SAVE_DIR, exist_ok=True)

def load_h5(path):
    with h5py.File(path, "r") as f:
        img = f["image"][:]
        msk = f["mask"][:]
    img = cv2.resize(img, (IMG_W, IMG_H)).astype(np.float32)
    img /= img.max()
    return img, msk

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".h5")])
sample_files = random.sample(files, min(20, len(files)))  # save only 20 examples

for idx, file in enumerate(sample_files):
    img, true_mask = load_h5(os.path.join(DATA_DIR, file))
    
    pred = model.predict(np.expand_dims(img, 0))[0]
    pred_mask = tf.argmax(pred, axis=-1).numpy()

    # Save side-by-side visualization
    fig, axes = plt.subplots(1, 3, figsize=(12,4))
    axes[0].imshow(img[...,0], cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(true_mask)
    axes[1].set_title("Ground Truth Mask")
    axes[2].imshow(pred_mask)
    axes[2].set_title("Predicted Mask")
    for a in axes: a.axis('off')

    output_path = os.path.join(SAVE_DIR, f"prediction_{idx}.png")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

print(f"\n📁 Done! Saved {len(sample_files)} results to: {SAVE_DIR}\n")
