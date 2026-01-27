"""
Final U-Net Pipeline for BraTS2021
- Directly reads BraTS folders (.nii.gz)
- Auto converts to .h5
- Trains U-Net
- Exports EfficientNet-ready images
- DirectML compatible (Windows)
"""

import os
os.environ["TF_USE_DIRECTML"] = "1"

# ---------------------- IMPORTS ----------------------
import glob, random
import numpy as np
import h5py
import cv2
import nibabel as nib
from tqdm import tqdm
import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPool2D, Conv2DTranspose, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau,
    CSVLogger, EarlyStopping
)

# ---------------------- GPU SETTINGS ----------------------
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

try:
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy("mixed_float16")
    print("⚙️ Mixed Precision Enabled\n")
except:
    print("⚠️ Mixed Precision Not Supported\n")

# ---------------------- CONFIG ----------------------
BRATS_ROOT = r"D:\FinalProject\UNet\data\BraTS2021_Training_Data"
DATA_DIR   = "data"
OUT_DIR    = "files_h5"
RESULTS_DIR = "results_h5"

EFF_DIR  = os.path.join(RESULTS_DIR, "efficientnet_inputs")
MASK_DIR = os.path.join(RESULTS_DIR, "raw_masks")

IMG_H, IMG_W = 240, 240
INPUT_CHANNELS = 4
NUM_CLASSES = 3

LR = 1e-4
EPOCHS = 25
BATCH_SIZE = 16

USE_FILE_LIMIT = True
FILE_LIMIT = 6000

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EFF_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------- BraTS → H5 ----------------------
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def mid_slice(vol):
    return vol[:, :, vol.shape[2] // 2]

def convert_brats_to_h5():
    patients = [
        os.path.join(BRATS_ROOT, p)
        for p in os.listdir(BRATS_ROOT)
        if p.startswith("BraTS")
    ]

    if USE_FILE_LIMIT:
        patients = random.sample(patients, min(len(patients), FILE_LIMIT))

    print(f"🔄 Converting {len(patients)} BraTS folders to .h5")

    for p in tqdm(patients):
        pid = os.path.basename(p)
        try:
            t1    = nib.load(os.path.join(p, pid+"_t1.nii.gz")).get_fdata()
            t1ce  = nib.load(os.path.join(p, pid+"_t1ce.nii.gz")).get_fdata()
            t2    = nib.load(os.path.join(p, pid+"_t2.nii.gz")).get_fdata()
            flair = nib.load(os.path.join(p, pid+"_flair.nii.gz")).get_fdata()
            seg   = nib.load(os.path.join(p, pid+"_seg.nii.gz")).get_fdata()

            img = np.stack([
                normalize(mid_slice(t1)),
                normalize(mid_slice(t1ce)),
                normalize(mid_slice(t2)),
                normalize(mid_slice(flair))
            ], axis=-1)

            mask = mid_slice(seg)
            mask[mask == 4] = 3

            img  = cv2.resize(img, (IMG_W, IMG_H))
            mask = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)

            with h5py.File(os.path.join(DATA_DIR, pid+".h5"), "w") as f:
                f.create_dataset("image", data=img)
                f.create_dataset("mask", data=mask)

        except Exception as e:
            print(f"❌ Skipped {pid}: {e}")

    print("✅ Conversion completed\n")

# ---------------------- DATA PIPELINE ----------------------
def read_h5(path):
    with h5py.File(path, "r") as f:
        img = f["image"][:]
        mask = f["mask"][:]

    mask = tf.one_hot(mask.astype(np.int32), NUM_CLASSES)
    return img.astype(np.float32), mask.numpy().astype(np.float32)

def tf_load(path):
    img, mask = tf.numpy_function(read_h5, [path], [tf.float32, tf.float32])
    img.set_shape([IMG_H, IMG_W, INPUT_CHANNELS])
    mask.set_shape([IMG_H, IMG_W, NUM_CLASSES])
    return img, mask

def build_datasets():
    files = glob.glob(os.path.join(DATA_DIR, "*.h5"))
    random.shuffle(files)

    n = len(files)
    t = int(n * 0.15)
    v = int(n * 0.15)

    test = files[:t]
    val  = files[t:t+v]
    train = files[t+v:]

    print(f"📁 Loaded {n} files → Train={len(train)}, Val={len(val)}, Test={len(test)}")

    def make_ds(lst):
        return tf.data.Dataset.from_tensor_slices(lst)\
            .map(tf_load, tf.data.AUTOTUNE)\
            .batch(BATCH_SIZE)\
            .prefetch(tf.data.AUTOTUNE)

    return make_ds(train), make_ds(val), make_ds(test)

# ---------------------- MODEL ----------------------
def conv_block(x, f):
    x = Conv2D(f, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(f, 3, padding="same")(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)

def build_unet():
    inp = Input((IMG_H, IMG_W, INPUT_CHANNELS))

    s1 = conv_block(inp, 64); p1 = MaxPool2D()(s1)
    s2 = conv_block(p1, 128); p2 = MaxPool2D()(s2)
    s3 = conv_block(p2, 256); p3 = MaxPool2D()(s3)
    s4 = conv_block(p3, 512); p4 = MaxPool2D()(s4)

    b = conv_block(p4, 1024)

    d1 = conv_block(Concatenate()([Conv2DTranspose(512, 2, 2, "same")(b), s4]), 512)
    d2 = conv_block(Concatenate()([Conv2DTranspose(256, 2, 2, "same")(d1), s3]), 256)
    d3 = conv_block(Concatenate()([Conv2DTranspose(128, 2, 2, "same")(d2), s2]), 128)
    d4 = conv_block(Concatenate()([Conv2DTranspose(64, 2, 2, "same")(d3), s1]), 64)

    out = Conv2D(NUM_CLASSES, 1, activation="softmax", dtype="float32")(d4)
    return Model(inp, out)

# ---------------------- LOSS ----------------------
def dice(y_true, y_pred):
    num = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2))
    den = tf.reduce_sum(y_true + y_pred, axis=(1,2))
    return tf.reduce_mean((num + 1e-7) / (den + 1e-7))

def loss_fn(y_true, y_pred):
    return tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred) + (1 - dice(y_true, y_pred))

# ---------------------- EXPORT EfficientNet INPUTS ----------------------
def export_efficientnet_inputs(model, test_ds):
    idx = 0
    for imgs, _ in test_ds:
        preds = model.predict(imgs)
        masks = np.argmax(preds, axis=-1)
        imgs = imgs.numpy()

        for i in range(len(imgs)):
            flair = imgs[i, :, :, 3]
            binary = (masks[i] > 0).astype(np.float32)

            masked = flair * binary
            masked = cv2.normalize(masked, None, 0, 255, cv2.NORM_MINMAX)
            masked = cv2.resize(masked, (224, 224))
            rgb = np.stack([masked]*3, axis=-1).astype(np.uint8)

            cv2.imwrite(os.path.join(EFF_DIR, f"img_{idx}.png"), rgb)
            np.save(os.path.join(MASK_DIR, f"mask_{idx}.npy"), masks[i])

            idx += 1

    print(f"✅ Saved {idx} EfficientNet-ready images\n")

# ---------------------- TRAIN ----------------------
def train():
    if len(glob.glob(os.path.join(DATA_DIR, "*.h5"))) == 0:
        convert_brats_to_h5()

    train_ds, val_ds, test_ds = build_datasets()

    model = build_unet()
    model.compile(Adam(LR), loss_fn, metrics=[dice])
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint(os.path.join(OUT_DIR, "best_unet.h5"), save_best_only=True),
            ReduceLROnPlateau(patience=2),
            EarlyStopping(patience=5, restore_best_weights=True),
            CSVLogger(os.path.join(OUT_DIR, "log.csv"))
        ]
    )

    print("\n🎉 U-Net Training Completed")
    print("📤 Exporting EfficientNet inputs...")
    export_efficientnet_inputs(model, test_ds)

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    train()
